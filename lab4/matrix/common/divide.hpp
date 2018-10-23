//
// Created by laomd on 2018/10/2.
//

#ifndef MATRIX_DISTRIBUTE_H
#define MATRIX_DISTRIBUTE_H

#include <istream>
#include <vector_io.hpp>
#include <type.hpp>
#include <msg.hpp>
#include <mympi.hpp>
using namespace std;

vector<MatrixElem> divide_onebyone(istream &fin, int num_elems, MPI_Comm comm) {
    Comm_Info info(comm);
    vector<int> balance = get_v(num_elems, info.comm_size);

    vector<MatrixElem> local_A(balance[info.rank]);

    if (info.rank == 0) {
        vector<MatrixElem> buffer;
        buffer.reserve(balance[0]);

        for (int i = 0; i < info.comm_size; i++) {
            buffer.resize(balance[i]);
            for (int j = 0; j < balance[i]; j++) {
                fin >> buffer[j];
            }
            if (balance[i]) {
                if (i == 0) {
                    local_A = buffer;
                } else {
                    Send(buffer, MPI_MATRIX_ELEM, i, i, comm);
                }
            }
        }
    } else {
        Recv(local_A, MPI_MATRIX_ELEM, 0, info.rank, comm, MPI_STATUS_IGNORE);
    }
    return local_A;
}

vector<MatrixElem> divide_scatter(istream &fin, int num_elems, MPI_Comm comm) {
    Comm_Info info(comm);
    vector<MatrixElem> array;
    if (info.rank == 0) {
        array = vector<MatrixElem>(num_elems);
        for (int i = 0; i < num_elems; i++)
            fin >> array[i];
    }
    vector<int> balance = get_v(num_elems, info.comm_size);
    vector<MatrixElem> local_A;

    if (balance[info.rank]) {
        local_A = vector<MatrixElem>(balance[info.rank]);
        Scatterv(array, local_A, MPI_MATRIX_ELEM, 0, MPI_COMM_WORLD);
    }
    return local_A;
}

#endif //MATRIX_DISTRIBUTE_H
