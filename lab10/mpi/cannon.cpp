//
// Created by laomd on 2019/1/19.
//
#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <ctime>
#include "mympi.h"
#include "mpi4cxx.hpp"
#include "cannon.h"
using namespace std;


int main(int argc, char* argv[]) {
    Init(&argc, &argv);

    SpMat matrix_a(0, 0), matrix_b(0, 0);
    Communicator COMM_WORLD(MPI_COMM_WORLD);
    Communicator cart_comm = COMM_WORLD.cart_create();
    auto rc_comm = cart_comm.cart_sub();
    Communicator row_comm = rc_comm[0], col_comm = rc_comm[1];

    if (COMM_WORLD.rank() == 0) {
        string filebase = "../data/csr_sparse";
        string file1 = filebase + argv[1] + ".mtx";
        string file2 = filebase + argv[1] + "-2.mtx";
        read_csr_matrix(file1, matrix_a);
        read_csr_matrix(file2, matrix_b);
    }

    matrix_scatter(matrix_a, row_comm, col_comm);
    matrix_scatter(matrix_b, row_comm, col_comm);

    COMM_WORLD.barrier();
    double start = MPI_Wtime();
    matrix_shift(matrix_a, col_comm.rank(), row_comm);
    matrix_shift(matrix_b, row_comm.rank(), col_comm);
    SpMat c = move(cannon_multiply(matrix_a, matrix_b, row_comm, col_comm));
    COMM_WORLD.barrier();
    double end = MPI_Wtime();
    if (COMM_WORLD.rank() == 0)
        cout << "mpi with " << COMM_WORLD.size() << " processors: " << (end - start) << "s" << endl;
    Finalize();
    return 0;
}