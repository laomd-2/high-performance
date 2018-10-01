//
// Created by laomd on 2018/10/1.
//

#ifndef MATRIX_MATRIX_H
#define MATRIX_MATRIX_H

void mat_vect_mult(
        const double local_A[],
        const double x[],
        double local_y[],
        int local_m,
        int n
) {
    int local_i, j;

    for (local_i = 0; local_i < local_m; local_i++) {
        local_y[local_i] = 0;
        for (j = 0; j < n; j++) {
            local_y[local_i] += local_A[local_i*n+j] * x[j];
        }
    }
}

#endif //MATRIX_MATRIX_H
