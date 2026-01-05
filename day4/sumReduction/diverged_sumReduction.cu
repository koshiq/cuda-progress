#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <device_launch_parameters.h>
#include <assert.h>

#define SIZE 256
#define SHMEM_SIZE 256 * 4

__global__ void sum_reduction(int *v, int *v_r) {
    __shared__ int partial_sum[SHMEM_SIZE];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    partial_sum[threadIdx.x] = v[tid];
    __syncthreads();

    for (int s = 1; s < blockDim.x; s *= 2) {
        int index = 2 * s * threadIdx.x; //sequential threads
        if (index < blockDim.x) {
            partial_sum[index] += partial_sum[index + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        v_r[blockIdx.x] = partial_sum[0];
    }
}

void init_vector(int *v, int n) {
    for (int i = 0; i < n; i++) {
        v[i] = rand() % 10;
    }
}

int main() {
    int n = 1 << 16;
    size_t bytes = n * sizeof(int);

    int *h_v, *h_v_r;
    int *d_v, *d_v_r;

    h_v = (int *)malloc(bytes);
    h_v_r = (int *)malloc(bytes);

    cudaMalloc(&d_v, bytes);
    cudaMalloc(&d_v_r, bytes);

    init_vector(h_v, n);

    cudaMemcpy(d_v, h_v, bytes, cudaMemcpyHostToDevice);

    int TB_SIZE = SIZE;

    int GRID_SIZE = (int)ceil((float)n/TB_SIZE);

    sum_reduction <<<GRID_SIZE, TB_SIZE>>> (d_v, d_v_r);
    sum_reduction <<<1, TB_SIZE>>> (d_v_r, d_v_r);

    cudaMemcpy(h_v_r, d_v_r, bytes, cudaMemcpyDeviceToHost);

    printf("Sum: %d\n", h_v_r[0]);
    cudaFree(d_v);
    cudaFree(d_v_r);
    free(h_v);
    free(h_v_r);

    return 0;

}