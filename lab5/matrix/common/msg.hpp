//
// Created by laomd on 2018/10/2.
//

#ifndef MATRIX_MSG_H
#define MATRIX_MSG_H

#include <mympi.hpp>
#include <vector>
#include <type.hpp>
#include <vector_manip.hpp>
using namespace std;

template <typename T>
void Bcast(vector<T>& buffer,  MPI_Datatype datatype, int root, MPI_Comm comm) {
    MPI_Bcast(buffer.data(), buffer.size(), datatype, root, comm);
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
vector<T> Gatherv(const vector<T>& local, MPI_Datatype datatype, int n, int root, MPI_Comm comm) {
    Comm_Info info(comm);
    vector<T> global;
    if (info.rank == root)
        global = vector<T>(n);
    vector<int> v = get_v(n, info.comm_size);
    vector<int> disp = get_disp(v, info.comm_size);
    MPI_Gatherv(local.data(), local.size(), datatype,
            global.data(), v.data(), disp.data(), datatype,
            root, comm);
    return global;
}

pair<MPI_Comm, MPI_Comm > Cart_2d_sub(MPI_Comm comm) {
    MPI_Comm comm_cart, comm_row, comm_col;
    Comm_Info info(comm);

    int dims[2] = {0, 0};
    MPI_Dims_create(info.comm_size, 2, dims);
    int periods[2] = {0, 0};
    MPI_Cart_create(comm, 2, dims, periods, 0, &comm_cart);

    int remain_dims[2] = {0, 1};
    MPI_Cart_sub(comm_cart, remain_dims, &comm_row);
    remain_dims[0] = 1;
    remain_dims[1] = 0;
    MPI_Cart_sub(comm_cart, remain_dims, &comm_col);

    return make_pair(comm_row, comm_col);
}
#endif //MATRIX_MSG_H
