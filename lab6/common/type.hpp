//
// Created by laomd on 2018/10/2.
//

#ifndef MATRIX_TYPE_H
#define MATRIX_TYPE_H

#include <mpi.h>
#include <istream>
#include <ostream>
using namespace std;

//#ifndef uint_fast64_t
//#define uint_fast64_t unsigned long long
//#endif

struct Comm_Info {
    int rank;
    int comm_size;

    explicit Comm_Info(MPI_Comm comm) {
        MPI_Comm_size(comm, &comm_size);
        MPI_Comm_rank(comm, &rank);
    }
};

#endif //MATRIX_TYPE_H
