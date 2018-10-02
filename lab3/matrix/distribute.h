//
// Created by laomd on 2018/10/2.
//

#ifndef MATRIX_DISTRIBUTE_H
#define MATRIX_DISTRIBUTE_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include "send_recv.h"

int* get_v(int m, int comm_size) {
    int *v = malloc(comm_size * sizeof(int));
    int local_m = m / comm_size;
    int remain = m % comm_size;
    for (int i = 0; i < comm_size; ++i) {
        v[i] = local_m + (int)(i < remain);
    }
    return v;
}

int* get_disp(const int* v, int comm_size) {
    int *disp = malloc(comm_size * sizeof(int));
    disp[0] = 0;
    for (int i = 1; i < comm_size; ++i) {
        disp[i] = disp[i - 1] + v[i - 1];
    }
    return disp;
}

DoubleArray distribute_by_row(FILE *file, int cnt, int m, int n, MPI_Comm comm) {
    Comm_Info info = init_info(comm);

    int *v = get_v(m, info.comm_size);
    int *disp = get_disp(v, info.comm_size);
    DoubleArray res = init_array(v[info.rank] * n);

    DoubleArray global_one_col;
    if (info.rank == 0) {
        global_one_col = init_array(m);
    }
    DoubleArray tmp = init_array(v[info.rank]);

    for (int i = 0; i < n; ++i) {
        int row, col;
        double entry;
        if (info.rank == 0) {
            while (cnt--) {
                fscanf(file, "%d %d %lf", &row, &col, &entry);
                if (col == i + 1) {
                    global_one_col.A[row - 1] = entry;
                } else {
//                    print_array(global_one_col);
                    break;
                }
            }
        }
        MPI_Request request;
        MPI_Iscatterv(global_one_col.A, v, disp, MPI_DOUBLE, tmp.A, tmp.size, MPI_DOUBLE, 0, comm, &request);
        MPI_Wait(&request, MPI_STATUS_IGNORE);
        for (int j = 0; j < tmp.size; ++j) {
            res.A[j * n + i] = tmp.A[j];
        }
//        print_array(tmp);
        if (info.rank == 0) {
            clear(global_one_col);
            global_one_col.A[row - 1] = entry;
        }
    }
    return res;
}

#endif //MATRIX_DISTRIBUTE_H
