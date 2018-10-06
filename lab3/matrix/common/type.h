//
// Created by laomd on 2018/10/2.
//

#ifndef MATRIX_TYPE_H
#define MATRIX_TYPE_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    double *A;
    int size;
}DoubleArray;

typedef struct {
    int rank;
    int comm_size;
}Comm_Info;

typedef struct {
    int i;
    int j;
    double value;
}MatrixElem;

Comm_Info get_info(MPI_Comm comm) {
    Comm_Info info;
    MPI_Comm_rank(comm, &info.rank);
    MPI_Comm_size(comm, &info.comm_size);
    return info;
}

DoubleArray malloc_array(int size) {
    if (size < 0)
        size = 0;
    DoubleArray array;
    array.size = size;
    array.A = malloc(sizeof(double) * size);
    if (array.A != NULL)
        memset(array.A, 0, sizeof(double) * size);
    return array;
}

void free_array(DoubleArray *array) {
    free(array->A);
    array->A = NULL;
    array->size = 0;
}

void copy(DoubleArray *dest, DoubleArray src) {
    (*dest).size = src.size;
    memcpy((*dest).A, src.A, src.size * sizeof(double));
}

void clear(DoubleArray array) {
    memset(array.A, 0, array.size * sizeof(double));
}

void debug(DoubleArray array) {
    for (int i = 0; i < array.size; ++i) {
        if (array.A[i])
            printf("%g ", array.A[i]);
    }
    printf("\n");
}
void print_array(FILE* out, DoubleArray array) {
    int cnt = 0;
    for (int i = 0; i < array.size; ++i) {
        if (array.A[i])
            cnt++;
    }
    fprintf(out, "%d\t%d\t%d\n", array.size, 1, cnt);
    for (int i = 0; i < array.size; ++i) {
        if (array.A[i])
            fprintf(out, "%d\t%d\t%E\n", i + 1, 1, array.A[i]);
    }
}

#endif //MATRIX_TYPE_H
