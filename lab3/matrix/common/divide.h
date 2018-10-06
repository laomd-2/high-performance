//
// Created by laomd on 2018/10/2.
//

#ifndef MATRIX_DISTRIBUTE_H
#define MATRIX_DISTRIBUTE_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mympi.h>

DoubleArray divide_on_row(FILE *file, int cnt, int m, int n, MPI_Comm comm) {
    Comm_Info info = get_info(comm);

    int *v = get_v(m, info.comm_size);
    DoubleArray res = malloc_array(v[info.rank] * n);

    DoubleArray global_one_col;
    if (info.rank == 0)
        global_one_col = malloc_array(m);

    DoubleArray tmp = malloc_array(v[info.rank]);

    for (int i = 0; i < n; ++i) {
        int row, col;
        double entry;
        if (info.rank == 0) {
            while (cnt--) {
                fscanf(file, "%d %d %lf", &row, &col, &entry);
                if (col == i + 1) {
                    global_one_col.A[row - 1] = entry;
                } else
                    break;
            }
        }
        MPI_Request request;
        Iscatterv(global_one_col, tmp, comm, 0, &request);
        MPI_Wait(&request, MPI_STATUS_IGNORE);
        for (int j = 0; j < tmp.size; ++j)
            res.A[j * n + i] = tmp.A[j];

        if (info.rank == 0) {
            clear(global_one_col);
            global_one_col.A[row - 1] = entry;
        }
    }

    free_array(&tmp);
    if (info.rank == 0)
        free_array(&global_one_col);
    free(v);
    return res;
}

MatrixElem* divide_on_elem(FILE* file, int num_elems, MPI_Comm comm, int *local_balance) {
    Comm_Info info = get_info(comm);
    MatrixElem *array = NULL;
    if (info.rank == 0) {
        array = malloc(sizeof(MatrixElem) * num_elems);
        for (int i = 0; i < num_elems; i++) {
            fscanf(file, "%d %d %lf", &(array[i].i), &(array[i].j), &(array[i].value));
        }
    }
    int *balance = get_v(num_elems, info.comm_size);
    *local_balance = balance[info.rank];
    MatrixElem *local_A = NULL;

    if (balance[info.rank]) {
        local_A = malloc(sizeof(MatrixElem) * balance[info.rank]);
        MPI_Scatterv(array, balance, get_disp(balance, info.comm_size), MPI_MATRIX_ELEM,
                     local_A, balance[info.rank], MPI_MATRIX_ELEM, 0, MPI_COMM_WORLD);
    }
    free(balance);
    if (info.rank == 0)
        free(array);
    return local_A;
}

#endif //MATRIX_DISTRIBUTE_H
