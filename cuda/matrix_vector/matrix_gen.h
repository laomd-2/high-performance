//
// Created by laomd on 2018/12/11.
//

#ifndef CUDA_MATRIX_GEN_H
#define CUDA_MATRIX_GEN_H

#include <cstddef>

void get_matrix(float *matrix, size_t m, size_t n, float(*f)(int,int)) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            matrix[i * n + j] = f(i, j);
        }
    }
}

void get_transpose(float *matrix, size_t m, size_t n, float(*f)(int,int)) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            matrix[i * m + j] = f(j, i);
        }
    }
}
#endif //CUDA_MATRIX_GEN_H
