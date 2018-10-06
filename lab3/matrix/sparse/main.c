//
// Created by laomd on 2018/10/2.
//
#include <stdio.h>
#include <divide.h>
#include <matrix_mul.h>

int main() {
    Init(0, 0);
    Comm_Info info = get_info(MPI_COMM_WORLD);

    FILE *matrix = 0, *vector = 0;
    int matrix_size[4];
    if (info.rank == 0) {
        matrix = fopen("matrix.mtx", "r");
        vector = fopen("vector.mtx", "r");
        fscanf(vector, "%d %d %d", matrix_size + 1, matrix_size, matrix_size + 3);
        fscanf(matrix, "%d %d %d", matrix_size, matrix_size + 1, matrix_size + 2);
    }
    MPI_Bcast(matrix_size, 4, MPI_INT, 0, MPI_COMM_WORLD);

    //    0、读入向量
    DoubleArray x = malloc_array(matrix_size[3]);
    if (info.rank == 0) {
        int tmp;
        for (int i = 0; i < matrix_size[3]; i++) {
            fscanf(vector, "%d %d %lf", &tmp, &tmp, x.A + i);
        }
        fclose(vector);
    }
    MPI_Request request;
    Ibcast(x, MPI_COMM_WORLD, 0, &request);

    //    1、任务划分
    int local_size;
    MatrixElem *local_A = divide_on_elem(matrix, matrix_size[2], MPI_COMM_WORLD, &local_size);
    fclose(matrix);
    printf("for process %d:\n", info.rank);

    //    2、局部矩阵与全局向量相乘
    DoubleArray local_y = malloc_array(matrix_size[0]);
    MPI_Wait(&request, MPI_STATUS_IGNORE);

    sparse_mat_vec_mul(local_A, local_size, x, local_y);
    free(local_A);

    //    3、任务聚合
    DoubleArray global_y = Reduce(local_y, MPI_SUM, 0, MPI_COMM_WORLD);
    free_array(&local_y);

    //    4、输出
    if (info.rank == 0) {
        FILE *out = fopen("result-sparse.mtx", "w");
        print_array(out, global_y);
        free_array(&global_y);
        fclose(out);
    }
    Finalize();
    return 0;
}