//
// Created by laomd on 2018/10/1.
//

#ifndef MATRIX_MATRIX_H
#define MATRIX_MATRIX_H

#include <mympi.h>

DoubleArray dense_mat_vect_mult(DoubleArray local_A, DoubleArray local_x, int n) {
    DoubleArray local_y = malloc_array(local_A.size / n);
    DoubleArray x = malloc_array(n);

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

void sparse_mat_vec_mul(MatrixElem local_A[], int size, DoubleArray x, DoubleArray local_y) {
//    printf("m=%d ", local_y.size);
    for (int i = 0; i < size; ++i) {
        int row = local_A[i].i;
        int col = local_A[i].j;
//        printf("%d %d, ", row, col);
        local_y.A[row - 1] += local_A[i].value * x.A[col - 1];
    }
//    printf("\n");
}
#endif //MATRIX_MATRIX_H
