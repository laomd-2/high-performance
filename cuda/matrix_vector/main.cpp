//
// Created by laomd on 2018/12/11.
//
#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include "matrix_gen.h"
#include "optimization.cuh"
using namespace std;

float cudamv(const float* matrix, const float* vec, size_t m, size_t n, float *result,
            unsigned g_size, unsigned b_size) {
    size_t column_pitch;
    float *d_matrix, *d_vec, *d_result;
    cudaMallocPitch(&d_matrix, &column_pitch, n * sizeof(float), m);
    cudaMalloc(&d_vec, n * sizeof(float));
    cudaMalloc(&d_result, m * sizeof(float));
    cudaMemcpy2D(d_matrix, column_pitch, matrix, n * sizeof(float), n * sizeof(float), m, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec, vec, n * sizeof(float), cudaMemcpyHostToDevice);

    dim3 grid_dim(g_size, g_size), block_dim(b_size, b_size);
    float elapsed;

    TIMER_START;
    mxvTranspose<<<grid_dim, block_dim>>>(m, n, column_pitch, d_matrix, d_vec, d_result);
    TIMER_END(elapsed);

    cudaMemcpy(result, d_result, m * sizeof(float), cudaMemcpyDeviceToHost);
    return elapsed;
}

float naive_matrix(int i, int j) {
    return (float)(i - 0.1 * j + 1);
}

float f_vector(int i, int j) {
    return (float)log(sqrt(i * i - i + 2));
}

int main(int argc, const char* argv[]) {
    if (argc < 4)
        return -1;
    size_t n = stoull(argv[1]);
    unsigned g_size = stoul(argv[2]), b_size = stoul(argv[3]);

    float *matrix, *vec, *result;
    matrix = new float[n * n];
    vec = new float[n];
    result = new float[n];
    get_transpose(matrix, n, n, naive_matrix);
    get_matrix(vec, n, 1, f_vector);

    float elapsed = cudamv(matrix, vec, n, n, result, g_size, b_size);

    for (int k = 0; k < n; ++k) {
        cout << result[k] << ' ';
    }
    cout << endl << elapsed << endl;
    return 0;
}