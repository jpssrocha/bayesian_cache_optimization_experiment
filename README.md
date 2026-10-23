# Tuning a tiled matrix multiply with Optuna

A small experiment that uses **Bayesian optimisation** (Optuna's TPE sampler)
to tune a C implementation of tiled matrix multiplication. The C kernel is
compiled as a shared library, loaded from Python with **cffi**, and called once
per Optuna trial with a candidate `(tile_size, num_threads)` pair. The value
returned to Optuna is the kernel's wall-clock time, so the search looks for the
configuration that finishes the workload fastest.

```
matmul.c ──make/gcc──▶ matmul.so ──cffi/dlopen──▶ main.py ──trial──▶ Optuna
                                                      │
                                                      └──▶ db.sqlite3 ──▶ dashboard
```

## What each file does

| File | Role |
| --- | --- |
| `matmul.c` | Allocates three 4096×4096 `float` matrices, fills `A` and `B`, runs the tiled `i-k-j` kernel twice under OpenMP, and returns the time of the last run (seconds). |
| `Makefile` | Builds `matmul.so` (`make`), runs the search (`make run`), opens the dashboard (`make dashboard`), and removes the binary (`make clean`). |
| `matmul.so` | Generated shared library exposing `double matmul_tiled(int tile_size, int num_threads)`. Not shipped; build it locally. |
| `main.py` | Declares the C signature, warms the library up, defines the Optuna objective, creates the study, runs 200 trials and prints the best parameters. |
| `db.sqlite3` | Created on the first run: Optuna's SQLite storage, one row per trial (parameters, value, state). |
| `pyproject.toml` / `uv.lock` | Python dependencies: `cffi`, `optuna`, `optuna-dashboard`. |

## Requirements

- Linux (the example is built and measured on Linux; use WSL or adapt the
  compile command on other systems)
- GCC with OpenMP support (`gcc`, typically from `build-essential`)
- GNU Make
- [uv](https://docs.astral.sh/uv/) (recommended) or Python ≥ 3.13 with `pip`

## Quick start

From this directory:

```bash
# 1. Build the shared library (the Makefile passes -O3, -fPIC and -fopenmp)
make

# 2. Install the Python environment
uv sync

# 3. Run the search
uv run main.py
```

or simply `make run` after `uv sync` (it builds first if needed).

The Makefile compiles `matmul.c` into `matmul.so` with:

```
gcc -O3 -fPIC -fopenmp -shared -o matmul.so matmul.c
```

`-fopenmp` enables OpenMP at compile and link time, `-fPIC` makes the code
position-independent so it can be loaded as a shared object, and `-shared`
produces the `.so` that cffi opens. To compile manually without Make, use the
same command.

Without `uv`:

```bash
make
python3 -m venv .venv
source .venv/bin/activate
pip install "cffi>=2.1.0" "optuna>=4.9.0" "optuna-dashboard>=0.20.0"
python main.py
```

The final line printed is the best configuration, for example:

```python
{'tile_size': 64, 'num_threads': 8}
```

> **Runtime warning:** one trial performs two 4096³ matrix multiplies
> (~275 GFLOP in total). 200 trials can easily take tens of minutes and can
> saturate every core of the machine. Reduce the number of trials (or `N` in
> `matmul.c`) for a first smoke test.

## What is actually being measured

`matmul_tiled` does the following, in order:

1. Allocates `A`, `B`, `C` on the heap (~192 MiB total).
2. Fills `A` and `B` with pseudo-random values using a fixed seed (`srand(0)`),
   so every trial multiplies the same matrices.
3. Runs the tiled kernel `NUM_EXECUCOES = 2` times. Before each run `C` is
   reset to zero.
4. Times only the kernel with `CLOCK_MONOTONIC`, returning the time of the
   **second** repetition (the first one acts as a cache warm-up).

Consequences worth keeping in mind:

- Allocation and data generation are *not* part of the score, but they still
  cost real time per trial.
- The score is a single noisy sample of wall-clock time, not a CPU-time or
  FLOP/s measurement. Frequency scaling, other processes, and NUMA effects can
  move it by several percent.
- The first call in `main.py` is a deliberate warm-up outside the study, so
  library/OpenMP start-up costs are not attributed to the first trial.

## The two search dimensions

**`tile_size` (2–600)** — the cache-blocking factor. The loops process blocks
of `ts × ts` elements so that a working set fits in the CPU caches before it is
reused. Too small and loop/index overhead dominates; too large and the blocks
no longer fit in cache. The C code clamps the last blocks with `i_max`,
`k_max`, `j_max`, so sizes that do not divide 4096 are legal.

**`num_threads` (1–12)** — passed to `omp_set_num_threads`. Note that the
`#pragma omp parallel for` in the kernel parallelises only the outermost `ii`
loop. With a large tile there are only `ceil(4096 / ts)` chunks (e.g. 7 chunks
for `ts = 600`), so extra threads may sit idle regardless of `num_threads`.
Comparisons above the number of physical cores also invite oversubscription.

Optuna's default TPESampler uses the results of previous trials to decide where
to sample next, rather than sweeping the grid blindly. The default direction is
**minimisation**, which is why the objective simply returns the time.

## Inspecting the results

Launch the dashboard against the same SQLite file:

```bash
make dashboard
# equivalent to: uv run optuna-dashboard sqlite:///db.sqlite3
```

Then open <http://127.0.0.1:8080>. Useful views: optimisation history, the
parameter importances, and the 2-D contour of `tile_size` vs `num_threads`.

From Python, without the dashboard:

```bash
uv run python -c "
import optuna
s = optuna.load_study(study_name='search_tilling_threads', storage='sqlite:///db.sqlite3')
print('best:', s.best_params, s.best_value)
print('trials:', len(s.trials))
"
```

## Re-running and resuming

`optuna.create_study` uses `load_if_exists=False` by default. `db.sqlite3` is
created on the first run and is ignored by git. Once it already contains a
study named `search_tilling_threads`, running `main.py` again raises
`DuplicateStudyError`. Choose one of:

- **Start over:** delete `db.sqlite3` (`rm -f db.sqlite3`) and run again.
- **Resume/extend:** pass `load_if_exists=True` to `create_study`. Calling
  `study.optimize(objective, n_trials=50)` then adds 50 *new* trials to the
  existing history.
- **Keep both:** change `study_name`.

## Ideas for further experiments

- Compare the search against a plain grid or random search: does TPE find a
  good region with fewer trials?
- Increase `NUM_EXECUCOES`, or record the minimum/median instead of the last
  run, to reduce measurement noise. How does that change the best point?
- Try `collapse(2)` on the `ii`/`kk` loops or `schedule(dynamic)`; what happens
  to the optimal `tile_size` when more parallelism is available?
- Compile with `-march=native` (or different optimisation levels) and rerun.
  Are the best parameters stable across binaries?
- Compare the tuned kernel with a tuned BLAS call (`cblas_sgemm`). How far is
  the hand-written tiled loop from a highly optimised library?

## Troubleshooting

| Symptom | Likely cause / fix |
| --- | --- |
| `OSError: cannot open shared object file` | `matmul.so` is missing, or you launched Python from another directory. Run `make` and run from this directory (the path is `./matmul.so`). |
| `undefined reference to omp_*` or `omp.h: No such file` | OpenMP flags are missing. `make` passes `-fopenmp` at both compile and link time; check your GCC installation. |
| `make` says `'matmul.so' is up to date` after changing compiler flags | Make tracks source timestamps, not flag changes. Run `make clean && make`. |
| `DuplicateStudyError` | The SQLite database already has this study. See "Re-running and resuming". |
| Every trial takes very long | Expected: `N = 4096`, two repetitions per trial. Lower `n_trials`, `N`, or `NUM_EXECUCOES` while exploring. |
| "Too-good" result around `1.0` seconds | The C code returns `1` when `malloc` fails, which looks like an excellent score. Check available RAM and the stderr message. |
| Results seem unstable | Wall-clock timing is noisy. Close other workloads, measure more repetitions, and treat differences below the noise level as ties. |
