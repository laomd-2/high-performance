//
// Created by laomd on 2018/10/2.
//

#ifndef MATRIX_MSG_H
#define MATRIX_MSG_H

#include <mpi.h>
#include <vector>
#include "type.hpp"
#include "vector_manip.hpp"
using namespace std;

template <typename T>
void Bcast(vector<T>& buffer,  MPI_Datatype datatype, int root, MPI_Comm comm) {
    MPI_Bcast(buffer.data(), buffer.size(), datatype, root, comm);
}

template <typename T>
vector<int> Gatherv(const vector<T>& local, vector<T>& global, MPI_Datatype datatype,
             int root, MPI_Comm comm) {
    Comm_Info info(comm);
    vector<int> v(info.comm_size);
    uint_fast64_t size = local.size();
    MPI_Allgather(&size, 1, MPI_INT, v.data(), 1, MPI_INT, comm);
    vector<int> disp = get_disp(v, info.comm_size);
    if (info.rank == root)
        global.resize((accumulate(v.begin(), v.end(), 0)));
    MPI_Gatherv(local.data(), local.size(), datatype,
                global.data(), v.data(), disp.data(), datatype, root, comm);
    return v;
}

template <typename T>
void Isend(const vector<T>& sendbuf, MPI_Datatype datatype,
        int tag, int dest, MPI_Comm comm, MPI_Request *request) {
    MPI_Isend(sendbuf.data(), sendbuf.size(), datatype,
            dest, tag, comm, request);
}

template <typename T>
void Recv(vector<T>& recvbuf, MPI_Datatype datatype,
          int src, int tag, MPI_Comm comm, MPI_Status *status) {
    MPI_Recv(recvbuf.data(), recvbuf.size(), datatype, src, tag, comm, status);
}
#endif //MATRIX_MSG_H
