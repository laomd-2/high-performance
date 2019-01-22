//
// Created by laomd on 2019/1/21.
//

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "csrspmat.h"
//#include "../util.cuh"
using namespace std;

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

#define FULL_MASK 0xffffffff
#define nWarps 1

#ifndef N
#define N 32
#endif

__global__ void kernel_multiply(const float* a_data, const int* r1, const int* c1,
                                const float* b_data, const int* r2, const int* c2) {
    __shared__ float sValA[nWarps];
    __shared__ int sColA[nWarps];
    __shared__ float result_row[nWarps * N];

    int blockId = blockIdx.x;
    int warpId = threadIdx.x / WARP_SIZE;
    int laneId = threadIdx.x % WARP_SIZE;
    int globalWarpId = blockId * nWarps + warpId;

    int aColIt = r1[globalWarpId] + laneId;
    int aColEnd = r1[globalWarpId + 1];
    int colA = aColIt < aColEnd ? c1[aColIt] : -1;
    float valA = aColIt < aColEnd ? a_data[aColIt] : 0.0f;

    int bColIt, bColEnd;
    int colB;
    float valB;

    for (int i = laneId; i < N; i += WARP_SIZE)
        result_row[warpId * N + i] = 0.0f;

    for(int k = 0, end = __popc(__ballot_sync(FULL_MASK, aColIt < aColEnd)); k < end; ++k)
    {
        if( laneId == k ) {
            sColA[warpId] = colA;
            sValA[warpId] = valA;
        }
        __syncthreads();

        bColIt = r2[sColA[warpId]] + laneId; // sColA is volatile and warp’s threads
        bColEnd = r2[sColA[warpId] + 1]; // are implicitly synchronized

        for(; __any_sync(FULL_MASK, bColIt < bColEnd ); bColIt += 32) {
            colB = bColIt < bColEnd ? c2[bColIt] : -1;
            valB = bColIt < bColEnd ? b_data[bColIt] : 0.0f;
            if (colB > -1)
                result_row[colB] += sValA[warpId] * valB;       // colB必不同，避免bank conflict
        }
    }

    if (globalWarpId == 0) {
        for (int i = laneId; i < N; i += WARP_SIZE)
            printf("(%d %d %lf) ", globalWarpId, i, result_row[warpId * N + i]);
    }
}

int main() {
    cudaDeviceReset();

    string filebase = "../data/csr_sparse";
    string file1 = filebase + to_string(N) + ".mtx";
    string file2 = filebase + to_string(N) + "-2.mtx";

    HostCsrSpMat mata, matb;
    ifstream fin;
    fin.open(file1);
    fin >> mata;
    fin.close();

    fin.open(file2);
    fin >> matb;
    fin.close();


    DeviceCsrSpMat a(mata), b(matb);

    kernel_multiply<<<1, nWarps * WARP_SIZE>>>(a.data, a.row_indices, a.col_indices,
            b.data, b.row_indices, b.col_indices);
    cudaDeviceReset();
}