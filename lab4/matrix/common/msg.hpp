//
// Created by laomd on 2018/10/2.
//

#ifndef MATRIX_MSG_H
#define MATRIX_MSG_H

#include <mpi.h>
#include <vector>
#include <type.hpp>
#include <vector_manip.hpp>
using namespace std;

template <typename T>
void Iscatterv(const vector<T>& global, vector<T>& local, MPI_Datatype datatype,
            int root, MPI_Comm comm, MPI_Request* request) {
    Comm_Info info(comm);
    vector<int> v = get_v(global.size(), info.comm_size);
    vector<int> disp = get_disp(v, info.comm_size);
    MPI_Iscatterv(global.data(), v.data(), disp.data(), datatype,
            local.data(), local.size(), datatype,
            root, comm, request);
}

template <typename T>
void Scatterv(const vector<T>& global, vector<T>& local, MPI_Datatype datatype,
            int root, MPI_Comm comm) {
    Comm_Info info(comm);
    vector<int> v = get_v(global.size(), info.comm_size);
    vector<int> disp = get_disp(v, info.comm_size);
    MPI_Scatterv(global.data(), v.data(), disp.data(), datatype,
                  local.data(), local.size(), datatype,
                  root, comm);
}

template <typename T>
void Ibcast(vector<T>& buffer,  MPI_Datatype datatype, int root, MPI_Comm comm, MPI_Request *request) {
    MPI_Ibcast(buffer.data(), buffer.size(), datatype, root, comm, request);
}

template <typename T>
void Gatherv(const vector<T>& local, vector<T> global, MPI_Datatype datatype,
        int root, MPI_Comm comm) {
    Comm_Info info(comm);
    vector<int> v = get_v(global.size(), info.comm_size);
    vector<int> disp = get_disp(v, info.comm_size);
    MPI_Gatherv(local.data(), local.size(), datatype,
            global.data(), v.data(), disp.data(), datatype, root, comm);
}

template <typename T>
vector<T> Reduce(const vector<T>& local, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm) {
    Comm_Info info(comm);
    vector<T> global;
    if (info.rank == root)
        global = vector<T>(local.size());
    MPI_Reduce(local.data(), global.data(), local.size(), datatype, op, root, comm);
    return global;
}

template <typename T>
void Allgatherv(const vector<T>& local, vector<T> global, MPI_Datatype datatype, MPI_Comm comm) {
    Comm_Info info(comm);
    vector<int> v = get_v(global.size(), info.comm_size);
    vector<int> disp = get_disp(v, info.comm_size);
    MPI_Allgatherv(local.data(), local.size(), datatype,
            global.data(), v.data(), disp.data(), datatype, comm);
}

template <typename T>
void Isend(const vector<T>& sendbuf, MPI_Datatype datatype,
        int tag, int dest, MPI_Comm comm, MPI_Request *request) {
    MPI_Isend(sendbuf.data(), sendbuf.size(), datatype,
            dest, tag, comm, request);
}

template <typename T>
void Send(const vector<T>& sendbuf, MPI_Datatype datatype,
           int tag, int dest, MPI_Comm comm) {
    MPI_Send(sendbuf.data(), sendbuf.size(), datatype,
              dest, tag, comm);
}

template <typename T>
void Recv(vector<T>& recvbuf, MPI_Datatype datatype,
          int src, int tag, MPI_Comm comm, MPI_Status *status) {
    MPI_Recv(recvbuf.data(), recvbuf.size(), datatype, src, tag, comm, status);
}
#endif //MATRIX_MSG_H
