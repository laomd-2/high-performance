//
// Created by laomd on 2018/10/1.
//

#ifndef MATRIX_MATRIX_H
#define MATRIX_MATRIX_H

#include <msg.hpp>
#include <vector_manip.hpp>

void sparse_mat_vec_mul(const vector<MatrixElem>& local_A,
        const vector<double>& x,
        vector<double>& local_y) {
    for (auto ele : local_A) {
        int row = ele.i;
        int col = ele.j;
        local_y[row - 1] += ele.value * x[col - 1];
    }
}
#endif //MATRIX_MATRIX_H
