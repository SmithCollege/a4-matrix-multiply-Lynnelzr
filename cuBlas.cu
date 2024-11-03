#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <time.h>

void initialize_matrices(float *A, float *B, int N) {
    for (int i = 0; i < N * N; ++i) {
        A[i] = (float)rand() / RAND_MAX;
        B[i] = (float)rand() / RAND_MAX;
    }
}

void matrix_multiply(cublasHandle_t handle, float *A, float *B, float *C, int N) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Perform matrix multiplication using cuBLAS
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, B, N, A, N, &beta, C, N);
}

double measure_time(int N) {
    size_t size = N * N * sizeof(float);

    // Allocate memory on host
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    // Allocate memory on device
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // Initialize matrices on host and copy them to device
    initialize_matrices(h_A, h_B, N);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Measure time for matrix multiplication
    clock_t start = clock();
    matrix_multiply(handle, d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    clock_t end = clock();

    // Calculate elapsed time in seconds
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;

    // Clean up
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return time_spent;
}

int main() {
    const int max_size = 4096;  // Adjust the maximum matrix size as needed
    srand(time(0));  // Seed for random matrix initialization

    printf("Matrix Size\tTime (seconds)\n");

    for (int N = 512; N <= max_size; N *= 2) {
        double elapsed_time = measure_time(N);
        printf("%dx%d\t\t%.6f\n", N, N, elapsed_time);
    }

    return 0;
}

