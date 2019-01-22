//
// Created by laomd on 2019/1/20.
//

#ifndef LAB10_CANNON_H
#define LAB10_CANNON_H

#include <algorithm>
#include <map>
#include "mpi4cxx.hpp"
#include "mympi.h"
#include "triple_mat.h"
using namespace std;

void matrix_scatter(SpMat &matrix, Communicator row_comm, Communicator col_comm) {
    if (row_comm.rank() == 0) {
        vector<int> sendcounts;
        if (col_comm.rank() == 0) {
            int p = col_comm.size();
            int rows_per_comm = matrix.n / p;

            sort(matrix.values.begin(), matrix.values.end(),
                 [](const MatrixElem& a, const MatrixElem& b) {
                     return a.row < b.row;
             });
            sendcounts.resize(p, 0);
            for (const MatrixElem& elem: matrix.values)
                sendcounts[elem.row / rows_per_comm]++;
        }
        matrix.values = move(col_comm.scatterv(move(matrix.values), sendcounts, MPI_Matrix_Elem, 0));
    }

    row_comm.barrier();

    vector<int> sendcounts;
    if (row_comm.rank() == 0) {
        col_comm.bcast_c(&matrix.n, 1, MPI_INT, 0);

        int p = row_comm.size();
        int cols_per_p = matrix.n / p;

        sort(matrix.values.begin(), matrix.values.end(),
                [](const MatrixElem& a, const MatrixElem& b) {
                    return a.col < b.col;
        });
        sendcounts.resize(p, 0);
        for (const MatrixElem& elem: matrix.values)
            sendcounts[elem.col / cols_per_p]++;
    }
    matrix.values = move(row_comm.scatterv(move(matrix.values), sendcounts, MPI_Matrix_Elem, 0));
}

void matrix_shift(SpMat &matrix, int offset, Communicator comm) {
    int rank = comm.rank(), size = comm.size();
    matrix.values = move(comm.send_recv(move(matrix.values), MPI_Matrix_Elem,
            (rank - offset + size) % size, 1, (rank + offset) % size, MPI_STATUS_IGNORE));
}

SpMat cannon_multiply(SpMat& a, SpMat& b, Communicator row_comm, Communicator col_comm) {
    SpMat c(0, a.n);
    int size = row_comm.size();
    multiply_add(a, b, c);
    while (--size) {
        matrix_shift(a, 1, row_comm);
        matrix_shift(b, 1, col_comm);
        multiply_add(a, b, c);
    }
    return c;
}

#endif //LAB10_CANNON_H
