#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>
#include "distribute.h"
#include "matrix.h"
#include "matrix_io.h"

int main() {
    MPI_Init(NULL, NULL);
    Comm_Info info = init_info(MPI_COMM_WORLD);

    FILE *matrix = NULL;
    FILE *vector = NULL;

    int matrix_size[2];
    int local_m, local_n;
    double *local_A = NULL;
    double *local_x = NULL;
    int cnt_A = 0, cnt_x = 0;

    if (info.rank == 0) {
        vector = fopen("vector.mtx", "r");
        matrix = fopen("matrix.mtx", "r");
        fscanf(vector, "%d %d %d", matrix_size + 1, matrix_size, &cnt_x);
        fscanf(matrix, "%d %d %d", matrix_size, matrix_size + 1, &cnt_A);
    }
    MPI_Bcast(matrix_size, 2, MPI_INT, 0, MPI_COMM_WORLD);

//    1、任务划分
    DoubleArray res = distribute_by_row(matrix, cnt_A, matrix_size[0], matrix_size[1], MPI_COMM_WORLD);
    local_A = res.A;
    local_m = res.size / matrix_size[1];
    res = distribute_by_row(vector, cnt_x, matrix_size[1], 1, MPI_COMM_WORLD);
    local_x = res.A;
    local_n = res.size;

//    2、局部矩阵与全局向量相乘
    double *local_y = malloc(local_m * sizeof(double));
    mat_vect_mult(local_A, local_x, local_y, local_m, local_n, matrix_size[1]);
//
//    3、任务聚合
    DoubleArray global_y;
    FILE* out;
    if (info.rank == 0) {
        global_y = init_array(matrix_size[0]);
        out = fopen("result.mtx", "w");
    }
    int *v = get_v(matrix_size[0], info.comm_size);
    int *disp = get_disp(v, info.comm_size);
    MPI_Gatherv(local_y, local_m, MPI_DOUBLE, global_y.A, v, disp, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//
//    4、输出
    if (info.rank == 0) {
        print_array(out, global_y);
    }
    MPI_Finalize();
    return 0;
}