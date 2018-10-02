//
// Created by laomd on 2018/10/1.
//

#ifndef MATRIX_MATRIX_IO_H
#define MATRIX_MATRIX_IO_H

#include <string.h>

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
