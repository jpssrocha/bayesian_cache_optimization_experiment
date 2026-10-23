#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>


#define N 4096
#define NUM_EXECUCOES 2


double tempo_em_segundos(struct timespec inicio, struct timespec fim) {
    return (fim.tv_sec - inicio.tv_sec) +
           (fim.tv_nsec - inicio.tv_nsec) * 1e-9;
}

double matmul_tiled(int tile_size, int num_threads) {

    // Setting up hyperparameters
    const int ts = tile_size;
    omp_set_num_threads(num_threads);

    /* Matrizes no heap */
    float *A;
    float *B;
    float *C;

    srand(0);

    /* Alocação dinâmica */
    size_t size = (size_t)N * N * sizeof(float);

    A = (float *) malloc(size);
    B = (float *) malloc(size);
    C = (float *) malloc(size);

    if (!A || !B || !C) {
        fprintf(stderr, "Erro ao alocar memória\n");
        return 1;
    }

    /* Inicialização das matrizes A e B */
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i*N + j] = (float) rand() / RAND_MAX;
            B[i*N + j] = (float) rand() / RAND_MAX;
        }
    }

    double tempo;

    for (int execucao = 0; execucao < NUM_EXECUCOES; execucao++) {

        /* Zera a matriz resultado */
        #pragma omp parallel for
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                C[i*N + j] = 0.0f;

        struct timespec inicio, fim;
        clock_gettime(CLOCK_MONOTONIC, &inicio);

        /* Kernel principal com TILING */
        #pragma omp parallel for schedule(static)
        for (int ii = 0; ii < N; ii += ts) {
            for (int kk = 0; kk < N; kk += ts) {
                for (int jj = 0; jj < N; jj += ts) {

                    int i_max = (ii + ts < N) ? ii + ts : N;
                    int k_max = (kk + ts < N) ? kk + ts : N;
                    int j_max = (jj + ts < N) ? jj + ts : N;

                    for (int i = ii; i < i_max; i++) {
                        for (int k = kk; k < k_max; k++) {
                            float aik = A[i*N + k];
                            for (int j = jj; j < j_max; j++) {
                                C[i*N + j] += aik * B[k*N + j];
                            }
                        }
                    }
                }
            }
        }

        clock_gettime(CLOCK_MONOTONIC, &fim);

        tempo = tempo_em_segundos(inicio, fim);
        
    }

    /* Liberação de memória */
    free(A);
    free(B);
    free(C);

    return tempo;
}
