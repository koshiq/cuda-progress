#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

__global__ void vectorAdd(int *a, int *b, int *c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}

void matrix_init(int *a, int n) {
    for (int i = 0; i < n; i++) {
        a[i] = rand() % 100;
    }
}

void error_check(int *a, int *b, int *c, int n) {
    for (int i = 0; i < n; i++) {
        assert(c[i] == a[i] + b[i]);
    }
}

int main() {
    int n = 1 << 16;
    int *h_a, *h_b, *h_c; //host pointers
    int *d_a, *d_b, *d_c; //device pointers

    size_t bytes = sizeof(int) * n; 

    h_a = (int*)malloc(bytes);
    h_b = (int*)malloc(bytes);
    h_c = (int*)malloc(bytes);

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    matrix_init(h_a, n);
    matrix_init(h_b, n);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    int NUM_THREADS = 256;
    int NUM_BLOCKS = (n + NUM_THREADS - 1) / NUM_THREADS;

    vectorAdd <<< NUM_BLOCKS, NUM_THREADS >>> (d_a, d_b, d_c, n);
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    error_check(h_a, h_b, h_c, n);

    cudaFree(d_a);
    cudaFree(d_b); 
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    printf("VECTOR ADDITION DONE.\n");

    return 0;
}
