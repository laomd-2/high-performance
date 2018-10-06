#include <stdio.h>
#include "divide.h"
#include "matrix_mul.h"

int main() {
    MPI_Init(NULL, NULL);
    Comm_Info info = get_info(MPI_COMM_WORLD);

    FILE *matrix = NULL;
    FILE *vector = NULL;

    int matrix_size[2];
    int cnt_A = 0, cnt_x = 0;

    if (info.rank == 0) {
        vector = fopen("vector.mtx", "r");
        matrix = fopen("matrix.mtx", "r");
        fscanf(vector, "%d %d %d", matrix_size + 1, matrix_size, &cnt_x);
        fscanf(matrix, "%d %d %d", matrix_size, matrix_size + 1, &cnt_A);
    }
    MPI_Bcast(matrix_size, 2, MPI_INT, 0, MPI_COMM_WORLD);

//    1、任务划分
    DoubleArray local_A = divide_on_row(matrix, cnt_A, matrix_size[0], matrix_size[1], MPI_COMM_WORLD);

    fclose(matrix);
    DoubleArray local_x = divide_on_row(vector, cnt_x, matrix_size[1], 1, MPI_COMM_WORLD);
    fclose(vector);

//    2、局部矩阵与全局向量相乘
    DoubleArray local_y = dense_mat_vect_mult(local_A, local_x, matrix_size[1]);
    free_array(&local_A);
    free_array(&local_x);

//    3、任务聚合
    DoubleArray global_y;
    if (info.rank == 0)
        global_y = malloc_array(matrix_size[0]);

    Gatherv(local_y, global_y, 0, MPI_COMM_WORLD);
    free_array(&local_y);

//    4、输出
    if (info.rank == 0) {
        FILE *out = fopen("result-dense.mtx", "w");
        print_array(out, global_y);
        free_array(&global_y);
        fclose(out);
    }
    MPI_Finalize();
    return 0;
}