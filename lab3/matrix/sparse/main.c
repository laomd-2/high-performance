//
// Created by laomd on 2018/10/2.
//
#include <stdio.h>
#include "mympi.h"

int main() {
    Init(0, 0);

    Comm_Info info = get_info(MPI_COMM_WORLD);

    MatrixElem elem[4] = {0};
    MatrixElem recv;
    if (info.rank == 0) {
        elem[0].i = 1;
        elem[0].j = 1;
        elem[1].value = 100;
    }
    MPI_Scatter(elem, 1, MPI_MATRIX_ELEM, &recv, 1, MPI_MATRIX_ELEM, 0, MPI_COMM_WORLD);
    printf("rank=%d: %d %d %lf", info.rank, recv.i, recv.j, recv.value);

//    FILE *matrix = 0, *vector = 0
//    int matrix_size[4];
//    if (info.rank == 0) {
//        matrix = fopen("matrix.mtx", "r");
//        vector = fopen("vector.mtx", "r");
//        fscanf(vector, "%d %d %d", matrix_size + 1, matrix_size, matrix_size + 3);
//        fscanf(matrix, "%d %d %d", matrix_size, matrix_size + 1, matrix_size + 2);
//    }
//    MPI_Bcast(matrix_size, 4, MPI_INT, 0, MPI_COMM_WORLD);

    Finalize();
    return 0;
}