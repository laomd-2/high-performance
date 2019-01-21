//
// Created by laomd on 2019/1/21.
//

#ifndef LAB10_CSRSPMAT_HPP
#define LAB10_CSRSPMAT_HPP

#include <istream>
#include <ostream>
#include <unordered_map>
#include <vector>
#include <map>
#include <iterator>
using namespace std;

struct HostCsrSpMat {
    vector<float> data;
    vector<int> col_indices;
    vector<int> row_indices;
    int n;

    HostCsrSpMat() = default;

    template <typename Visit>
    void for_each(Visit visit) const {
        int i = 0;
        for (int row = 0; row < row_indices.size(); ++row) {
            int end = row_indices[row];
            while (i < end && i < data.size()) {
                visit(row, col_indices[i], data[i]);
                i++;
            }
        }
    }

    friend istream&operator>>(istream&, HostCsrSpMat&);
};

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

istream &operator>>(istream &in, HostCsrSpMat &mat) {
    int nnz;
    in >> mat.n >> nnz;
    mat.row_indices.resize(mat.n + 1);
    mat.data.resize(nnz);
    mat.col_indices.resize(nnz);
    for (float& x: mat.data)
        in >> x;
    for (int& x: mat.col_indices)
        in >> x;
    for (int& x: mat.row_indices)
        in >> x;
    return in;
}

ostream &operator<<(ostream &out, const HostCsrSpMat &mat) {
    mat.for_each([&](int row, int col, float value) {
        out << "(" << row << "," << col << "," << value << ")" << endl;
    });
    return out;
}

#endif //LAB10_CSRSPMAT_HPP
