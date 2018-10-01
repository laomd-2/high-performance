#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>
#include "matrix_io.h"
#include "matrix.h"
#include "send_recv.h"

void distribute(FILE *file, int m, int n, int comm_size, double **local_A, int *local_m) {
    int tmp_m;
    int remain = m % comm_size;

    double *tmp_A = NULL;

    int i, end = 1;
    for (i = 0; i < comm_size; i++) {
        tmp_m = m / comm_size;
        tmp_m += i < remain;
        end += tmp_m;
        send_or_copy(&tmp_m, local_m, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        if (i == 0) {
            *local_A = malloc(tmp_m * n * sizeof(double));
        }
        if (tmp_A == NULL) {
            tmp_A = malloc(tmp_m * n * sizeof(double));
        }
        read_matrix(file, tmp_A, tmp_m, n, end);
        send_or_copy(tmp_A, *local_A, tmp_m * n, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
    }
}

int main() {
    MPI_Init(NULL, NULL);
    int rank, comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    FILE *matrix = NULL;
    FILE *vector = NULL;

    int n, m, num_A, num_x;
    double *local_A = NULL;
    double *x = NULL;

    if (rank == 0) {
        vector = fopen("vector.mtx", "r");
        matrix = fopen("matrix.mtx", "r");
        if (matrix && vector) {
            fscanf(vector, "%d %d %d", &n, &m, &num_x);
            fscanf(matrix, "%d %d %d", &m, &n, &num_A);
        }
    }
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    x = malloc(n * sizeof(double));
    if (rank == 0) {
        if (vector) {
            read_matrix(vector, x, 1, n, n);
        }
    }
    MPI_Bcast(x, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int local_m, local_n;
    if (rank == 0) {
        if (matrix && vector) {
            distribute(matrix, m, n, comm_size, &local_A, &local_m);
        }
    } else {
        recv_divide(&local_m, &local_A, n);
    }

    double *local_y = malloc(local_m * sizeof(double));
    MPI_Barrier(MPI_COMM_WORLD);
    mat_vect_mult(local_A, x, local_y, local_m, n);

    if (rank == 0) {
        FILE *out = fopen("result.mtx", "w");
        fprintf(out, "%d\t1\t60222\n", m);
        int i;

        output(out, local_y, local_m, 0);

        MPI_Status status;
        int cnt;
        for (i = 1; i < comm_size; ++i) {
            MPI_Recv(local_y, local_m, MPI_DOUBLE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
            MPI_Get_count(&status, MPI_DOUBLE, &cnt);
            output(out, local_y, cnt, status.MPI_SOURCE);
        }
    } else {
        MPI_Send(local_y, local_m, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return 0;
}