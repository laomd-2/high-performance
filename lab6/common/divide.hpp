//
// Created by laomd on 2018/10/2.
//

#ifndef MATRIX_DISTRIBUTE_H
#define MATRIX_DISTRIBUTE_H

#include <istream>
#include <tuple>
#include <cmath>
#include "msg.hpp"
#include "vector_manip.hpp"
#include "io.hpp"
using namespace std;

vector<uint_fast64_t> divide_by_element(istream& in, uint_fast64_t n, MPI_Comm comm) {
    Comm_Info info(comm);
    vector<uint_fast64_t> balances = get_v(n, info.comm_size);

    vector<uint_fast64_t> local;
    if (info.rank == 0) {
        vector<MPI_Request> requests(info.comm_size - 1);
        vector<uint_fast64_t> buffer;
        buffer.reserve(balances[0]);
        for (int i = 0; i < info.comm_size; ++i) {
            uint_fast64_t balance = balances[i];
            buffer.resize(balance);
            for (uint_fast64_t &x: buffer)
                read(in, x);
            if (i == 0)
                local = buffer;
            else
                Isend(buffer, MPI_UINT64_T, i, i, comm, &(requests[i - 1]));
        }
        MPI_Waitall(requests.size(), requests.data(), MPI_STATUS_IGNORE);
    } else {
        local.resize(balances[info.rank]);
        Recv(local, MPI_UINT64_T, 0, info.rank, comm, MPI_STATUS_IGNORE);
    }
    return local;
}

#endif //MATRIX_DISTRIBUTE_H
