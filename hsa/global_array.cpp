#include "hip/hip_runtime.h"

__device__ int global_array[32];

__global__ void global_array_put(int* input) {
    int i = threadIdx.x;
    global_array[i] = input[i];
}

__global__ void global_array_get(int* output) {
    int i = threadIdx.x;
    output[i] = global_array[i];
}

__global__ void global_array_increase() {
    int i = threadIdx.x;
    global_array[i]++;
}

__global__ void global_array_insert(int* input, int* output) {
    int i = threadIdx.x;

    output[i] = global_array[i];
//    output[i] = 5;
    global_array[i] = input[i];
}
