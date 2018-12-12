//
// Created by laomd on 2018/12/6.
//
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cmath>
#include <ctime>
#include "../util.cuh"

typedef thrust::device_vector<double> DVector1D;

__device__ void cal_one_ele(size_t i, size_t j, size_t n, double* P) {
    double sum = 0;
    for (int k = 0; k < n; ++k) {
        double a_ik = i - 0.1 * k + 1, b_kj = 0.2 * j - 0.1 * k;
        sum += a_ik * b_kj;
    }
    *(P + i * n + j) = sum;
}

__global__ void matrix_mul_kernel(double* P, unsigned n) {
    size_t n_per_block = n / gridDim.x;
    size_t n_per_thread = n_per_block / blockDim.x;

    size_t block_i = blockIdx.y * n_per_block,
           block_j = blockIdx.x * n_per_block;
    size_t thread_i = block_i + threadIdx.y * n_per_thread,
           thread_j = block_j + threadIdx.x * n_per_thread;
    for (int i = 0; i < n_per_thread; ++i) {
        size_t ii = thread_i + i;
        for (int j = 0; j < n_per_thread; ++j) {
            cal_one_ele(ii, thread_j + j, n, P);
        }
    }
}

int main(int argc, const char* argv[]) {
    if (argc < 4)
        return 1;
    int n = stoi(argv[1]);
    int g_n = stoi(argv[2]);
    int tile_n = stoi(argv[3]);
    dim3 grid_dim(g_n, g_n), block_dim(tile_n, tile_n);
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
    if (argc > 4)
        cout << result << endl;
    return 0;
}