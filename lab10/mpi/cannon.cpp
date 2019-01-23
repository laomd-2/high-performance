//
// Created by laomd on 2019/1/19.
//
#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <ctime>
#include "mpi4cxx.hpp"
#include "csr_io.h"
using namespace std;

void matrix_shift(HostCsrSpMat &matrix, int offset, Communicator comm) {
    if (offset != 0) {
        int rank = comm.rank(), size = comm.size();
        int send = (rank - offset + size) % size, recv = (rank + offset) % size;
        matrix.data = move(comm.send_recv(move(matrix.data), MPI_FLOAT,
                                          send, 1, recv, MPI_STATUS_IGNORE));
        matrix.col_indices = move(comm.send_recv(move(matrix.col_indices), MPI_INT,
                                                 send, 3, recv, MPI_STATUS_IGNORE));
        matrix.row_indices = move(comm.send_recv(move(matrix.row_indices), MPI_INT,
                                                 send, 5, recv, MPI_STATUS_IGNORE));
    }
}

HostCsrSpMat cannon_multiply(HostCsrSpMat& a, HostCsrSpMat& b, Communicator row_comm, Communicator col_comm) {
    HostCsrSpMat c(move(a * b));
    int size = row_comm.size();
    while (--size) {
        matrix_shift(a, 1, row_comm);
        matrix_shift(b, 1, col_comm);
        c = c + a * b;
    }
    return c;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    Communicator COMM_WORLD(MPI_COMM_WORLD);
    Communicator cart_comm = COMM_WORLD.cart_create();
    auto comms = cart_comm.cart_sub();

    string filebase = "../data/csr_sparse";
    string file1 = filebase + argv[1] + ".mtx";
    string file2 = filebase + argv[1] + "-2.mtx";
    HostCsrSpMat a(move(read_sub_matrix(file1, comms[0], comms[1]))),
            b(move(read_sub_matrix(file2, comms[0], comms[1])));

    COMM_WORLD.barrier();
    double start = MPI_Wtime();
    matrix_shift(a, comms[1].rank(), comms[0]);
    matrix_shift(b, comms[0].rank(), comms[1]);
    HostCsrSpMat res = move(cannon_multiply(a, b, comms[0], comms[1]));
    COMM_WORLD.barrier();
    double end = MPI_Wtime();

//    int i = comms[0].rank(), j = comms[1].rank();
//    res.for_each([&](int r, int c, float v) {
//        printf("(%d %d %lf) ", i * res.n + r, j * res.n + c, v);
//    });
//    printf("\n");
    if (COMM_WORLD.rank() == 0)
        cout << argv[1] << ", " << COMM_WORLD.size() << ", " << (end - start) << "s" << endl;
    MPI_Finalize();
    return 0;
}