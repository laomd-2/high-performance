//
// Created by laomd on 2018/12/6.
//

#ifndef CUDA_INDEX_H
#define CUDA_INDEX_H

#include <cuda_runtime.h>
#include <thrust/detail/vector_base.h>
#include <ostream>
using namespace std;

#define __JETBRAINS_IDE__

#ifdef __JETBRAINS_IDE__
dim3 gridDim, blockDim;
dim3 threadIdx, blockIdx;
#define WARP_SIZE 32
#define __popc(a) (a)
#define __ballot(a) (a)
#define __any(a) (a)
#endif

#define TIMER_START \
{cudaEvent_t start, stop;\
cudaEventCreate(&start);\
cudaEventCreate(&stop);\
cudaEventRecord(start, nullptr);

#define TIMER_END(elapsed) \
cudaEventRecord(stop, nullptr);\
cudaEventSynchronize (stop);\
cudaEventElapsedTime(&elapsed, start, stop);\
cudaEventDestroy(start);\
cudaEventDestroy(stop);}


// 1D grid of 1D blocks
__device__ int getGlobalIdx_1D_1D() {
    return blockIdx.x *blockDim.x + threadIdx.x;
}

// 1D grid of 2D blocks
__device__ int getGlobalIdx_1D_2D() {
    return blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
}

// 1D grid of 3D blocks
__device__ int getGlobalIdx_1D_3D() {
    return blockIdx.x * blockDim.x * blockDim.y * blockDim.z
        + threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;
}

// 2D grid of 1D blocks
__device__ int getGlobalIdx_2D_1D() {
    int blockId   = blockIdx.y * gridDim.x + blockIdx.x;
    int threadId = blockId * blockDim.x + threadIdx.x;
    return threadId;
}

// 2D grid of 2D blocks  
__device__ int getGlobalIdx_2D_2D() {
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    return threadId;
}

// 2D grid of 3D blocks
__device__ int getGlobalIdx_2D_3D() {
    int blockId = blockIdx.x
        + blockIdx.y * gridDim.x;
    int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
        + (threadIdx.z * (blockDim.x * blockDim.y))
        + (threadIdx.y * blockDim.x)
        + threadIdx.x;
    return threadId;
}

// 3D grid of 1D blocks
__device__ int getGlobalIdx_3D_1D() {
    int blockId = blockIdx.x
        + blockIdx.y * gridDim.x
        + gridDim.x * gridDim.y * blockIdx.z;
    int threadId = blockId * blockDim.x + threadIdx.x;
    return threadId;
}

// 3D grid of 2D blocks
__device__ int getGlobalIdx_3D_2D() {
    int blockId = blockIdx.x
        + blockIdx.y * gridDim.x
        + gridDim.x * gridDim.y * blockIdx.z;
    int threadId = blockId * (blockDim.x * blockDim.y)
        + (threadIdx.y * blockDim.x)
        + threadIdx.x;
    return threadId;
}

// 3D grid of 3D blocks
__device__ int getGlobalIdx_3D_3D() {
    int blockId = blockIdx.x
        + blockIdx.y * gridDim.x
        + gridDim.x * gridDim.y * blockIdx.z;
    int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
        + (threadIdx.z * (blockDim.x * blockDim.y))
        + (threadIdx.y * blockDim.x)
        + threadIdx.x;
    return threadId;
}

template <typename T, typename Alloc = std::allocator<T>>
ostream& operator<< (ostream& out, const thrust::detail::vector_base<T, Alloc>& a) {
    for (const auto& i: a)
        out << i << ' ';
    return out;
}

#endif //CUDA_INDEX_H
