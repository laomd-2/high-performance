//
// Created by laomd on 2018/10/2.
//
#include <iostream>
#include <fstream>
#include <divide.h>
#include <matrix_mul.h>
#include <vector_io.hpp>
using namespace std;

int main() {
    Init(0, 0);
    Comm_Info info(MPI_COMM_WORLD);

    ifstream matrix, vector_;
    int matrix_size[4];
    if (info.rank == 0) {
        matrix.open("matrix.mtx");
        vector_.open("vector.mtx");
        if (matrix.is_open() && vector_.is_open()) {
            vector_ >> matrix_size[1] >> matrix_size[0] >> matrix_size[3];
            matrix >> matrix_size[0] >> matrix_size[1] >> matrix_size[2];
        } else {
            cout << "File not found." << endl;
            Finalize();
            return 0;
        }
    }
    MPI_Bcast(matrix_size, 4, MPI_INT, 0, MPI_COMM_WORLD);

    //    0、读入向量
    vector<double> x(matrix_size[3]);
    if (info.rank == 0) {
        int tmp;
        for (int i = 0; i < matrix_size[3]; i++) {
            vector_ >> tmp >> tmp >> x.data()[i];
        }
        vector_.close();
    }
    MPI_Request request;
    Ibcast(x, MPI_DOUBLE, 0, MPI_COMM_WORLD, &request);

    //    1、任务划分
    vector<MatrixElem> local_A = divide_on_elem(matrix, matrix_size[2], MPI_COMM_WORLD);

    matrix.close();

    //    2、局部矩阵与全局向量相乘
    vector<double> local_y(matrix_size[0]);
    MPI_Wait(&request, MPI_STATUS_IGNORE);

    sparse_mat_vec_mul(local_A, x, local_y);
//    printf("for progress %d\n", info.rank);
//    debug(local_y);

    //    3、任务聚合
    vector<double> global_y = Reduce(local_y, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    //    4、输出
    if (info.rank == 0) {
        ofstream out("result-sparse.mtx");
        out << global_y;
        out.close();
    }
    Finalize();
    return 0;
}