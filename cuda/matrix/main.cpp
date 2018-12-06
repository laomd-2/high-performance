//
// Created by laomd on 2018/12/6.
//
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cmath>
#include <ctime>
#include "../util.h"

typedef thrust::device_vector<double> DVector1D;

__global__ void matrix_mul_kernel(double* P, unsigned n) {
    size_t thread_n = gridDim.x * blockDim.x;
    size_t matrix_block_n = n / thread_n;

    size_t tid = getGlobalIdx_2D_2D();
    size_t i = tid / thread_n, j = tid % thread_n;

    size_t i_base = i * matrix_block_n;
    size_t j_base = j * matrix_block_n;
    for (i = 0; i < matrix_block_n; ++i, ++i_base) {
        size_t real_j = j_base;
        for (j = 0; j < matrix_block_n; ++j, ++real_j) {
            double sum = 0;
            for (int k = 0; k < n; ++k) {
                double a_ik = i_base - 0.1 * k + 1, b_kj = 0.2 * real_j - 0.1 * k;
                sum += a_ik * b_kj;
            }
            *(P + i_base * n + real_j) = sum;
        }
    }
}

int main() {
    unsigned n = 5000;
    dim3 grid_dim(25, 25), block_dim(10, 10);
    DVector1D result(n * n, 0);

    float elapsed=0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, nullptr);

    matrix_mul_kernel<<<grid_dim, block_dim>>>(result.data().get(), n);

    cudaEventRecord(stop, nullptr);
    cudaEventSynchronize (stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cout << elapsed / 1000.0 << 's' << endl;
//    cout << result << endl;
    return 0;
}