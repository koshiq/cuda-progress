#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>

#define SHMMEM_SIZE 16 * 16 * 4

__global__ void tiledMatMUL(int *a, int *b, int *c, int n, int tile_size){
    __shared__ int A[SHMMEM_SIZE];
    __shared__ int B[SHMMEM_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int row = by * tile_size + ty;
    int col = bx * tile_size + tx;

    int temp_val = 0;

    for (int i = 0; i < n / tile_size; i++) {
        A[ty * tile_size + tx] = a[row * n + i * tile_size + tx];
        B[(ty * tile_size + tx) * n + col] = b[(i * tile_size + ty) * n + col];

        __syncthreads();

        for (int k = 0; k < tile_size; k++) {
            temp_val += A[ty * tile_size + k] * B[k * tile_size + tx];
        }

        __syncthreads();
    }

    c[row * n + col] = temp_val;
}






void init_matrices(int *a, int *b, int *c, int n) {
    for (int i = 0; i < n * n; i++) {
        a[i] = rand() % 100;
        b[i] = rand() % 100;
        c[i] = 0;
    }
}


int main() {
    int n = 1 << 10;
    size_t bytes = sizeof(int) * n * n;

    int *h_a, *h_b, *h_c;
    h_a = (int*)malloc(bytes);
    h_b = (int*)malloc(bytes);
    h_c = (int*)malloc(bytes);

    int *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, bytes);
    cudaMalloc((void**)&d_b, bytes);
    cudaMalloc((void**)&d_c, bytes);

    init_matrices(h_a, h_b, h_c, n);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    
    int BLOCK_SIZE = 16;
    int GRID_SIZE = (int)ceil(n / BLOCK_SIZE);
    
    dim3 grid(GRID_SIZE, GRID_SIZE);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

    tiledMatMUL<<<grid, threads>>>(d_a, d_b, d_c, n , BLOCK_SIZE);
    cudaDeviceSynchronize();

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    printf("Matrix multiplication completed successfully.\n");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
