//
// Created by laomd on 2018/10/1.
//

#ifndef MATRIX_MATRIX_IO_H
#define MATRIX_MATRIX_IO_H

#include <string.h>

static int pos_x = 0;
static int pos_y = 0;
static double entry = 0;

void read_matrix(FILE* file, double *arr, int m, int n, int end) {
    memset(arr, 0, m * n * sizeof(double));
    if (pos_x) {
        int i = (pos_y - 1) % m, j = (pos_x - 1) % n;
        arr[i * n + j] = entry;
    }

    while (fscanf(file, "%d %d %lf", &pos_x, &pos_y, &entry) != EOF
           && pos_y <= end) {
        int i = (pos_y - 1) % m, j = (pos_x - 1) % n;
//        printf("%d %d\n", pos_x, pos_y);
        arr[i * n + j] = entry;
    }
}

void output(FILE* out, const double *arr, int n, int rank) {
    int i;
    int base = n * rank + 1;
    for (i = 0; i < n; ++i) {
        if (arr[i]) {
            fprintf(out, "%d\t1\t%E\n", base + i, arr[i]);
        }
    }
}

#endif //MATRIX_MATRIX_IO_H
