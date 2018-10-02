//
// Created by laomd on 2018/10/2.
//

#ifndef MATRIX_DISTRIBUTE_H
#define MATRIX_DISTRIBUTE_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mympi.h"

DoubleArray distribute_by_row(FILE *file, int cnt, int m, int n, MPI_Comm comm) {
    Comm_Info info = init_info(comm);

    int *v = get_v(m, info.comm_size);
    DoubleArray res = init_array(v[info.rank] * n);

    DoubleArray global_one_col;
    if (info.rank == 0)
        global_one_col = init_array(m);

    DoubleArray tmp = init_array(v[info.rank]);

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

#endif //MATRIX_DISTRIBUTE_H
