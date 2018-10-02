//
// Created by laomd on 2018/10/1.
//

#ifndef MATRIX_MATRIX_H
#define MATRIX_MATRIX_H

#include "type.h"

void mat_vect_mult(
        const double local_A[],
        const double local_x[],
        double local_y[],
        int local_m,
        int local_n,
        int n
) {
    int local_i, j;

    DoubleArray x = init_array(n);
    Comm_Info info = init_info(MPI_COMM_WORLD);

    int *v = get_v(n, info.comm_size);
    int *disp = get_disp(v, info.comm_size);

    MPI_Allgatherv(local_x, local_n, MPI_DOUBLE, x.A, v, disp, MPI_DOUBLE, MPI_COMM_WORLD);

    for (local_i = 0; local_i < local_m; local_i++) {
        local_y[local_i] = 0;
        for (j = 0; j < n; j++) {
            local_y[local_i] += local_A[local_i*n+j] * x.A[j];
        }
    }
}

#endif //MATRIX_MATRIX_H
