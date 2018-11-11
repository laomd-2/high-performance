//
// Created by laomd on 2018/10/1.
//

#ifndef MATRIX_MATRIX_H
#define MATRIX_MATRIX_H

#include <vector>
using namespace std;

vector<double> dense_mat_vect_mult(const vector<vector<double>>& local_A,
                                   const vector<double>& local_x) {
    int m = local_A.size(), n = local_x.size();
    vector<double> local_y(m, 0);

    int local_i, j;
    for (local_i = 0; local_i < m; local_i++) {
        for (j = 0; j < n; j++) {
            local_y[local_i] += local_A[local_i][j] * local_x[j];
        }
    }
    return local_y;
}
#endif //MATRIX_MATRIX_H
