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
#define nWarps 2

__global__ void kernel_multiply(const float* a_data, const int* r1, const int* c1,
                                const float* b_data, const int* r2, const int* c2) {
    __shared__ float sValA[nWarps];
    __shared__ int sColA[nWarps];

    int blockId = blockIdx.x;
    int warpId = threadIdx.x / WARP_SIZE;
    int laneId = threadIdx.x % WARP_SIZE;

    int aColIt = r1[warpId] + laneId;
    int aColEnd = r1[warpId + 1];
    int colA = aColIt < aColEnd ? c1[aColIt] : -1;
    float valA = aColIt < aColEnd ? a_data[aColIt] : 0.0f;

    int bColIt, bColEnd;
    int colB;
    float valB;
    __shared__ float result_row[nWarps][WARP_SIZE][WARP_SIZE];

    for (int j = 0; j < WARP_SIZE; ++j) {
        result_row[warpId][j][laneId] = 0;
    }

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
            colB = bColIt < bColEnd ? c2[bColIt] : 0;
            valB = bColIt < bColEnd ? b_data[bColIt] : 0.0f;
        }
        result_row[warpId][colB][laneId] += sValA[warpId] * valB;       // 避免bank conflict
    }
    for (int i = 1; i < WARP_SIZE; ++i) {    // 避免bank conflict
        result_row[warpId][laneId][laneId] += result_row[warpId][laneId][((i + laneId) & (WARP_SIZE - 1))];
//        printf("(%d %d %lf) ", warpId, i, result_row[i]);
//        if (laneId == 0)
//            printf("\n");
    }
    printf("%d %d %lf\n", warpId, laneId, result_row[warpId][laneId][laneId]);
}

int main(int argc, const char* argv[]) {
    cudaDeviceReset();

    string filebase = "../data/csr_sparse";
    string file1 = filebase + argv[1] + ".mtx";
    string file2 = filebase + argv[1] + "-2.mtx";

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