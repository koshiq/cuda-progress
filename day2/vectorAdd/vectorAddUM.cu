#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

__global__ void vectorAddUM(int *a, int *b, int *c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}

void init_vector(int *a, int *b, int n) {
    for (int i = 0; i < n; i++) {
        a[i] = rand() % 100;
    }
}

void check_answer(int *a, int *b, int *c, int n) {
    for (int i = 0; i < n; i++) {
        assert(c[i] == a[i] + b[i]);
    }
}

int main() {
    int id;
    cudaGetDevice(&id);
    int n = 1 << 16;

    size_t bytes = sizeof(int) * n;
    int *a, *b, *c;

    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);

    init_vector(a, b, n);

    int BLOCK_SIZE = 256;
    int GRID_SIZE = (int)ceil(n / BLOCK_SIZE);

    vectorAddUM <<< GRID_SIZE, BLOCK_SIZE >>> (a, b, c, n);

    cudaDeviceSynchronize();


    check_answer(a, b, c, n);
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    printf("VECTOR ADDITION DONE.\n");

    return 0;
}
