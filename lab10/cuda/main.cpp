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
#include "device_csrspmat.h"
#include "util.cuh"
using namespace std;
#define FULL_MASK 0xffffffff

#ifndef N
#define N 32768
#endif

#define nWarps 16

//#define nWarps 2

__global__ void kernel_multiply(const float* a_data, const int* r1, const int* c1,
                                const float* b_data, const int* r2, const int* c2,
                                float* result) {
    __shared__ float sValA[nWarps][32];
    __shared__ int sColA[nWarps][32];

    int blockId = blockIdx.x;
    int warpId = threadIdx.x / warpSize;
    int laneId = threadIdx.x % warpSize;
    int globalWarpId = (blockId * nWarps) + warpId;

    for (int i = laneId; i < N; i += warpSize)
        result[globalWarpId * N + i] = 0.0f;

    int aColEnd = r1[globalWarpId + 1];
    int n_per_t = N / warpSize;
    for (int i = 0; i < n_per_t; ++i) {
        int aColIt = r1[globalWarpId] + laneId + i * warpSize;

        if (aColIt < aColEnd) {
            sColA[warpId][laneId] = c1[aColIt];
            sValA[warpId][laneId] = a_data[aColIt];
        }

        int bColIt, bColEnd;
        int colB, colA;
        float valA, valB;

        for(int k = 0, end = __popc(__ballot(aColIt < aColEnd)); k < end; ++k)
        {
            bColIt = r2[colA] + laneId; // sColA is volatile and warp’s threads
            bColEnd = r2[colA + 1]; // are implicitly synchronized

            int kk = (k + laneId) % end;
            colA = sColA[warpId][kk];
            valA = sValA[warpId][kk];

            for(; __any(bColIt < bColEnd ); bColIt += warpSize) {
                colB = bColIt < bColEnd ? c2[bColIt] : -1;
                valB = bColIt < bColEnd ? b_data[bColIt] : 0.0f;
                if (colB > -1)
                    result[globalWarpId * N + colB] += valA * valB;
            }
        }
    }
}

int main() {
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

    float elapsed;
    TIMER_START;
    DeviceCsrSpMat a(mata), b(matb);

    float* result;
    cudaMalloc(&result, N * N * sizeof(float));

    kernel_multiply<<<N / nWarps, nWarps * 32>>>(a.data, a.row_indices, a.col_indices,
            b.data, b.row_indices, b.col_indices, result);
    auto *host_res = new float[N * N];
    cudaMemcpy(host_res, result, N * N * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < N; ++i) {
            float x = host_res[i];
            if (x != 0.0f) {
                printf("(%d %lf) ", i, x);
            }
        }
    TIMER_END(elapsed);
    cout << N << ',' << elapsed / 1000 << endl;
}