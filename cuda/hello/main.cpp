//
// Created by laomd on 2018/12/6.
//

#include <cuda_runtime.h>
#include <cstdio>
using namespace std;

__device__ int getGlobalIdx_2D_2D()
{
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    return threadId;
}

__global__ void hello() {
    int global_tid = getGlobalIdx_2D_2D();
    printf("Hello world from thread %d\n", global_tid);
}
int main() {
    dim3 grid_dim(2, 4), block_dim(8, 16);

    freopen("hello.txt","w",stdout);
    hello<<<grid_dim, block_dim>>>();
    cudaDeviceReset();
}