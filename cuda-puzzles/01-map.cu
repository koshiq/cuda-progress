#include "math_functions_patch.h"
#include <stdio.h>
#include <cuda_runtime.h>
#pragma pop_macro("noexcept")

const int N = 4;

__global__ void map_spec(const float* a, float* output) {
    int i = threadIdx.x;
    output[i] = a[i] + 10.0f;
}

int main() {
    float h_a[N];
    float h_output[N];

    for (int i = 0; i < N; i++)
        h_a[i] = float(i);

    float *d_a = nullptr;
    float *d_output = nullptr;
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_output, N * sizeof(float));

    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);

    map_spec<<<1, N>>>(d_a, d_output);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        printf("h_output[%d] = %f\n", i, h_output[i]);
    }

    cudaFree(d_a);
    cudaFree(d_output);
    
    return 0;
}