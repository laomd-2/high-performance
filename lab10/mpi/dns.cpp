//
// Created by laomd on 2019/1/23.
//

#include <iostream>
#include <fstream>
#include "csr_io.h"

// MPI树形归约
void mat_reduce_add(HostCsrSpMat& mat, Communicator comm) {
    int id = comm.rank();

    HostCsrSpMat other;
    other.n = mat.n;

    int size = comm.size();
    while (size > 1 && id < size) {
        size /= 2;
        if (id < size) {
//            printf("%d recv from %d\n", id, id + size);
            comm.recv(other.data, MPI_FLOAT, id + size, 1, MPI_STATUS_IGNORE);
            comm.recv(other.col_indices, MPI_INT, id + size, 2, MPI_STATUS_IGNORE);
            comm.recv(other.row_indices, MPI_INT, id + size, 3, MPI_STATUS_IGNORE);
            mat = mat + other;
        } else {
//            printf("%d send to %d\n", id, id - size);
            comm.send(mat.data, MPI_FLOAT, id - size, 1);
            comm.send(mat.col_indices, MPI_INT, id - size, 2);
            comm.send(mat.row_indices, MPI_INT, id - size, 3);
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    string filebase = "../data/csr_sparse";
    string file1 = filebase + argv[1] + ".mtx";
    string file2 = filebase + argv[1] + "-2.mtx";;

    Communicator COMM(MPI_COMM_WORLD);
    Communicator cart_comm = COMM.cart_create(3);
    auto comms = cart_comm.cart_sub(3);

    COMM.barrier();
    double start = MPI_Wtime();

    HostCsrSpMat a(move(read_sub_matrix(file1, comms[0], comms[2]))),
                 b(move(read_sub_matrix(file2, comms[2], comms[1])));

    HostCsrSpMat res(move(a * b));
    mat_reduce_add(res, comms[2]);

    COMM.barrier();
    double end = MPI_Wtime();
    if (COMM.rank() == 0)
        cout << argv[1] << ", " << COMM.size() << ", " << (end - start) << "s" << endl;
//    if (comms[2].rank() == 0) {
//        int i = comms[0].rank(), j = comms[1].rank();
//        res.for_each([&](int r, int c, float v) {
//           printf("(%d %d %lf) ", i * res.n + r, j * res.n + c, v);
//        });
//        printf("\n");
//    }
    MPI_Finalize();
}