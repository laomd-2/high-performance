//
// Created by laomd on 2018/10/2.
//

#ifndef MATRIX_DISTRIBUTE_H
#define MATRIX_DISTRIBUTE_H

#include <istream>
#include <vector>
#include <type.hpp>
#include <msg.hpp>
#include <mympi.hpp>
using namespace std;

vector<MatrixElem> divide_on_elem(istream& fin, int num_elems, MPI_Comm comm) {
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
