#include <stdio.h>

__global__ void helloGPU() {
    printf("Hello World from GPU! Thread ID: %d\n", threadIdx.x);
}

int main() {
    printf("Hello World from CPU!\n");

    helloGPU<<<1, 1>>>();
    cudaDeviceSynchronize();
}