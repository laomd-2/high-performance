#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>

void mat_vect_mult(
        const double local_A[],
        const double x[],
        double local_y[],
        int local_m,
        int n
        ) {
    int local_i, j;

    for (local_i = 0; local_i < local_m; local_i++) {
        local_y[local_i] = 0;
        for (j = 0; j < n; j++) {
            local_y[local_i] += local_A[local_i*n+j] * x[j];
        }
    }
}

int pos_x = 0, pos_y = 0;
double entry = 0;
void read_matrix(FILE* file, double *arr, int m, int n, int end, int* cnt) {
    memset(arr, m * n * sizeof(double), 0);
    if (pos_x) {
        int i = (pos_y - 1) % m, j = (pos_x - 1) % n;
        arr[i * n + j] = entry;
    }

    while ((*cnt)--) {
        if(fscanf(file, "%d %d %lf", &pos_x, &pos_y, &entry) == EOF)
            break;
        if (pos_y > end)
            break;
        int i = (pos_y - 1) % m, j = (pos_x - 1) % n;
        arr[i * n + j] = entry;
    }
}

void divide(FILE* file, int m, int n, int cnt, int comm_size, double **local_A, int *local_m) {
    int tmp_m;
    int remain = m % comm_size;
    int i;
    MPI_Request request;

    double *tmp_A = NULL;

    int end = 1;

    for (i = 0; i < comm_size; i++) {
        tmp_m = m / comm_size;
        tmp_m += i < remain;
        end += tmp_m;
        if (i == 0) {
            *local_m = tmp_m;
            *local_A = malloc(tmp_m * n * sizeof(double));
        } else {
            MPI_Isend(&tmp_m, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &request);
        }
        if (tmp_A == NULL) {
            tmp_A = malloc(tmp_m * n * sizeof(double));
        }
        read_matrix(file, tmp_A, tmp_m, n, end, &cnt);
        if (i == 0) {
            memcpy(*local_A, tmp_A, tmp_m * n * sizeof(double));
        } else {
            MPI_Isend(tmp_A, tmp_m * n, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &request);
        }
    }
}

void recv_divide(int *local_m, double **local_A, int n) {
    MPI_Recv(local_m, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    *local_A = malloc(*local_m * n * sizeof(double));
    MPI_Recv(*local_A, *local_m * n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

int num_nonzero = 0;
void output(FILE* out, double *arr, int n, int rank) {
    int i;
    int base = n * rank + 1;
    for (i = 0; i < n; ++i) {
        if (arr[i]) {
            fprintf(out, "%d\t1\t%lf\n", base + i, arr[i]);
            num_nonzero++;
        }
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
            read_matrix(vector, x, 1, n, n, &num_x);
        }
    }
    MPI_Bcast(x, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int local_m, local_n;
    if (rank == 0) {
        if (matrix && vector) {
            divide(matrix, m, n, num_A, comm_size, &local_A, &local_m);
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