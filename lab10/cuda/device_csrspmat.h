//
// Created by laomd on 2019/1/21.
//

#ifndef LAB10_DEVICE_CSRSPMAT_HPP
#define LAB10_DEVICE_CSRSPMAT_HPP

#include <cuda_runtime.h>
#include "../common/csrspmat.h"
using namespace std;

struct DeviceCsrSpMat {
    float* data;
    int* col_indices;
    int* row_indices;
    int n;

    DeviceCsrSpMat() = default;
    DeviceCsrSpMat(const HostCsrSpMat& host_mat): n(host_mat.n) {
        int nnz = host_mat.data.size(), n = host_mat.row_indices.size();
        cudaMalloc(&data, sizeof(float) * nnz);
        cudaMalloc(&col_indices, sizeof(int) * nnz);
        cudaMalloc(&row_indices, sizeof(int) * (n + 1));
        cudaMemset(row_indices, 0, sizeof(int));

        cudaMemcpy(data, host_mat.data.data(), sizeof(float) * nnz, cudaMemcpyHostToDevice);
        cudaMemcpy(col_indices, host_mat.col_indices.data(), sizeof(int) * nnz, cudaMemcpyHostToDevice);
        cudaMemcpy(row_indices + 1, host_mat.row_indices.data(), sizeof(int) * n, cudaMemcpyHostToDevice);
    }
};

#endif //LAB10_DEVICE_CSRSPMAT_HPP
