//
// Created by laomd on 2019/1/23.
//

#ifndef LAB10_CSR_IO_H
#define LAB10_CSR_IO_H

#include "../common/csrspmat.h"
#include "mpi4cxx.hpp"

HostCsrSpMat read_sub_matrix(const string &file, Communicator comm1, Communicator comm2) {
    int i = comm1.rank(), j = comm2.rank();

    HostCsrSpMat mat;
    ifstream fin(file);
    fin >> mat;

    int n_per = mat.n / comm1.size();
    int row_first = i * n_per, col_first = j * n_per;
    return mat.take(row_first, col_first, n_per);
}

#endif //LAB10_CSR_IO_H
