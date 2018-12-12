//
// Created by laomd on 2018/12/11.
//

#ifndef CUDA_NAIVE_H
#define CUDA_NAIVE_H

#include "../util.cuh"

__global__ void mxvNaive(size_t m, size_t n, size_t column_pitch,
                         const float* d_matrix, const float* d_vec,
                         float* d_result) {
    size_t id = getGlobalIdx_2D_2D();

    if (id >= m) return;
    float sum = 0.0f;
    float *row = (float*)(((char*)d_matrix + id * column_pitch));
    for (int i = 0; i < n; ++i) {
        sum += row[i] * d_vec[i];
    }
    d_result[id] = sum;
}

__global__ void mxvTranspose(size_t m, size_t n, size_t column_pitch,
                             const float* d_matrix, const float* d_vec,
                             float* d_result) {
    size_t id = getGlobalIdx_2D_2D();

    if (id >= m) return;
    float sum = 0.0f;

    for (int i = 0; i < n; ++i) {
        float *row = (float*)(((char*)d_matrix + i * column_pitch));
        sum += row[id] * d_vec[i];
    }
    d_result[id] = sum;
}

#endif //CUDA_NAIVE_H
