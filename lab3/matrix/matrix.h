//
// Created by laomd on 2018/10/1.
//

#ifndef MATRIX_MATRIX_H
#define MATRIX_MATRIX_H

#include "mympi.h"

DoubleArray mat_vect_mult(
        DoubleArray local_A,
        DoubleArray local_x,
        int n
) {
    DoubleArray local_y = init_array(local_A.size / n);
    DoubleArray x = init_array(n);

    Allgatherv(local_x, x, MPI_COMM_WORLD);

    int local_i, j;
    for (local_i = 0; local_i < local_A.size / n; local_i++) {
        local_y.A[local_i] = 0;
        for (j = 0; j < n; j++) {
            local_y.A[local_i] += local_A.A[local_i*n+j] * x.A[j];
        }
    }
    return local_y;
}

#endif //MATRIX_MATRIX_H
