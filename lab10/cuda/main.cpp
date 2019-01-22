//
// Created by laomd on 2019/1/21.
//

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include "csrspmat.h"
//#include "../util.cuh"
using namespace std;
#define FULL_MASK 0xffffffff

#ifndef N
#define N 32
#endif

#define nWarps 16

//#define nWarps 2

__global__ void kernel_multiply(const float* a_data, const int* r1, const int* c1,
                                const float* b_data, const int* r2, const int* c2,
                                float* result) {
    __shared__ float sValA[nWarps];
    __shared__ int sColA[nWarps];

    int blockId = blockIdx.x;
    int warpId = threadIdx.x / warpSize;
    int laneId = threadIdx.x % warpSize;
    int globalWarpId = (blockId * nWarps) + warpId;

    for (int i = laneId; i < N; i += warpSize)
        result[globalWarpId * N + i] = 0.0f;

    int aColEnd = r1[globalWarpId + 1];
    int n_per_t = N / warpSize;
    for (int i = 0; i < n_per_t; ++i) {
        int aColIt = r1[globalWarpId] + laneId * n_per_t + i;

        int colA = aColIt < aColEnd ? c1[aColIt] : -1;
        float valA = aColIt < aColEnd ? a_data[aColIt] : 0.0f;

        int bColIt, bColEnd;
        int colB;
        float valB;

        for(int k = 0, end = __popc(__ballot(aColIt < aColEnd)); k < end; ++k)
        {
            if( laneId == k ) {
                sColA[warpId] = colA;
                sValA[warpId] = valA;
            }
            __syncthreads();

            bColIt = r2[sColA[warpId]] + laneId; // sColA is volatile and warp’s threads
            bColEnd = r2[sColA[warpId] + 1]; // are implicitly synchronized

            colB = -1;
            valB = 0.0f;
            for(; __any(bColIt < bColEnd ); bColIt += warpSize) {
                colB = bColIt < bColEnd ? c2[bColIt] : -1;
                valB = bColIt < bColEnd ? b_data[bColIt] : 0.0f;
                if (colB > -1) {
                    result[globalWarpId * N + colB] += sValA[warpId] * valB;       // colB必不同，避免bank conflict
                    if (globalWarpId == 0 && colB == 0)
                        printf("(%d %lf %d %lf)\n", sColA[warpId], sValA[warpId], colB, valB);
                }
            }
            __syncthreads();
        }
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

//    cout << mata;

    DeviceCsrSpMat a(mata), b(matb);

    float* result;
    cudaMalloc(&result, N * N * sizeof(float));

    kernel_multiply<<<N / nWarps, nWarps * 32>>>(a.data, a.row_indices, a.col_indices,
            b.data, b.row_indices, b.col_indices, result);
    float *host_res = new float[N * N];
    cudaMemcpy(host_res, result, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    int row = 0;
    for (int i = 0; i < N; ++i)
        printf("(%d %d %lf) ", row, i, host_res[row* N + i]);
    printf("\n");
    cudaDeviceReset();
}