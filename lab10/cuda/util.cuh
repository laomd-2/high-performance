//
// Created by laomd on 2018/12/6.
//

#ifndef CUDA_INDEX_H
#define CUDA_INDEX_H

#include <cuda_runtime.h>
using namespace std;

#ifdef IDE
dim3 gridDim, blockDim;
dim3 threadIdx, blockIdx;
#define warpSize 32
#define __popc(a) (a)
#define __ballot(a) (a)
#define __any(a) (a)
#define __syncthreads() ()
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

#endif //CUDA_INDEX_H
