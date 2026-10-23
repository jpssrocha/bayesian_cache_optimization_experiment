"""Bayesian optimisation (Optuna) of a tiled matrix-multiply kernel written in C.

The Python side is deliberately thin: it loads a pre-compiled shared library
with cffi, declares the C signature, and lets Optuna choose the kernel's two
"hyperparameters" (tile size and number of OpenMP threads). Each trial returns
the kernel's wall-clock time, so the study searches for the configuration that
makes the same numerical work finish fastest.

Pipeline::

    matmul.c --(gcc -O3 -fopenmp -shared -fPIC)--> matmul.so
    matmul.so --(cffi dlopen)--> Python --(Optuna trial)--> best parameters

Run it from this directory (the library path is relative) after compiling::

    gcc -O3 -fopenmp -fPIC -shared -o matmul.so matmul.c
    uv run main.py

See README.md for the full walkthrough.
"""

from cffi import FFI
import optuna


# ---------------------------------------------------------------------------
# 1. Bind to the compiled C kernel (ABI mode)
# ---------------------------------------------------------------------------
# With a plain FFI() handle, cffi does not compile a wrapper. cdef() only tells
# Python how to interpret the symbol's arguments and return value, and dlopen()
# resolves that symbol from the shared object at run time.
SHARED_LIB_MATMUL: str = "./matmul.so"  # relative to the process's working directory

matmul_lib = FFI()  # cffi handle used for declarations and library loading
matmul_lib.cdef("double matmul_tiled(int tile_size, int num_threads);")
matmul = matmul_lib.dlopen(SHARED_LIB_MATMUL)

# One throw-away call before the search. It forces the dynamic linker and the
# OpenMP runtime to initialise and touches the freshly allocated pages.
# Leaving that first-call overhead inside a trial would bias the measurement.
matmul.matmul_tiled(128, 8)


# ---------------------------------------------------------------------------
# 2. Tell Optuna what "good" means
# ---------------------------------------------------------------------------
def objective(trial: optuna.Trial) -> float:
    """Evaluate one (tile_size, num_threads) configuration.

    Optuna calls this function once per trial. ``suggest_int`` both proposes a
    value based on everything observed so far (TPE sampler by default) and
    records it in the trial, so the sampler can correlate parameters with the
    score. The returned number is the quantity being *minimised*: elapsed
    seconds, where lower is better.

    Parameters
    ----------
    trial:
        Handle to the current trial. Every searched parameter must be drawn
        through it; otherwise Optuna cannot learn from the result.

    Returns
    -------
    float
        Kernel wall-clock time in seconds. In C, allocation and matrix
        initialisation happen *outside* the timed region.
    """
    # Search space (bounds are inclusive). tile_size is the cache-blocking
    # factor: each 4096x4096 matrix is processed in ts x ts blocks.
    # num_threads is handed to omp_set_num_threads() inside the kernel.
    tile_size = trial.suggest_int("tile_size", 2, 600)
    num_threads = trial.suggest_int("num_threads", 1, 12)

    # matmul_tiled() returns the time of its last repetition (it runs twice),
    # in seconds. Optuna minimises by default, so no sign flip is needed.
    return matmul.matmul_tiled(tile_size, num_threads)


# ---------------------------------------------------------------------------
# 3. Run and persist the search
# ---------------------------------------------------------------------------
def main() -> None:
    """Create the study and run 200 sequential trials.

    The SQLite backend stores every trial (parameters, value, state, timing)
    in ``db.sqlite3``, which is what the Optuna dashboard reads. The study is
    identified by its name; with ``load_if_exists`` left at its default of
    ``False``, re-running against an existing database raises
    ``DuplicateStudyError`` (see README for how to resume or reset).
    """
    study = optuna.create_study(
        study_name="search_tilling_threads",  # typo kept for database continuity
        storage="sqlite:///db.sqlite3",
    )

    # 200 sequential trials; each one runs two full 4096^3 matmuls in C, so
    # expect this to take a while. Lower the trial count (or N in matmul.c)
    # while experimenting.
    study.optimize(objective, n_trials=200)

    # Only the winning configuration is printed here. study.best_value holds
    # the corresponding time, and study.trials holds the full history.
    print(study.best_params)


if __name__ == "__main__":
    main()
