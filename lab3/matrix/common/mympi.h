//
// Created by laomd on 2018/10/2.
//

#ifndef MATRIX_MYMPI_H
#define MATRIX_MYMPI_H

#include <stdlib.h>
#include <mpi.h>
#include <type.h>

MPI_Datatype MPI_MATRIX_ELEM;
void Init(int* argc, char*** argv) {
    MPI_Init(argc, argv);

    int array_of_blocklengths[3] = {1, 1, 1};

    MPI_Aint array_of_displacements[3] = {0, sizeof(int), sizeof(int) * 2};
    MPI_Datatype array_of_datatypes[3] = {MPI_INT, MPI_INT, MPI_DOUBLE};

    MPI_Type_create_struct(3, array_of_blocklengths, array_of_displacements, array_of_datatypes, &MPI_MATRIX_ELEM);
    MPI_Type_commit(&MPI_MATRIX_ELEM);
}

void Finalize() {
    MPI_Type_free(&MPI_MATRIX_ELEM);
    MPI_Finalize();
}

int* get_v(int m, int comm_size) {
    int *v = malloc(comm_size * sizeof(int));
    int local_m = m / comm_size;
    int remain = m % comm_size;
    for (int i = 0; i < comm_size; ++i) {
        v[i] = local_m + (int)(i < remain);
    }
    return v;
}

int* get_disp(const int* v, int comm_size) {
    int *disp = malloc(comm_size * sizeof(int));
    disp[0] = 0;
    for (int i = 1; i < comm_size; ++i) {
        disp[i] = disp[i - 1] + v[i - 1];
    }
    return disp;
}

void Iscatterv(DoubleArray global, DoubleArray local,
        MPI_Comm comm, int root, MPI_Request* request) {
    Comm_Info info = get_info(comm);
    int *v = get_v(global.size, info.comm_size);
    int *disp = get_disp(v, info.comm_size);
    MPI_Iscatterv(global.A, v, disp, MPI_DOUBLE, local.A, local.size, MPI_DOUBLE, root, comm, request);
    free(v);
    free(disp);
}

void Ibcast(DoubleArray buffer, MPI_Comm comm, int root, MPI_Request *request) {
    MPI_Ibcast(buffer.A, buffer.size, MPI_DOUBLE, root, comm, request);
}

void Gatherv(DoubleArray local, DoubleArray global,
        int root, MPI_Comm comm) {
    Comm_Info info = get_info(comm);
    int *v = get_v(global.size, info.comm_size);
    int *disp = get_disp(v, info.comm_size);
    MPI_Gatherv(local.A, local.size, MPI_DOUBLE, global.A, v, disp, MPI_DOUBLE, root, comm);
    free(v);
    free(disp);
}

DoubleArray Reduce(DoubleArray local, MPI_Op op, int root, MPI_Comm comm) {
    Comm_Info info = get_info(comm);
    DoubleArray global;
    if (info.rank == root)
        global = malloc_array(local.size);
    MPI_Reduce(local.A, global.A, local.size, MPI_DOUBLE, op, root, comm);
    return global;
}

void Allgatherv(DoubleArray local, DoubleArray global, MPI_Comm comm) {
    Comm_Info info = get_info(comm);
    int *v = get_v(global.size, info.comm_size);
    int *disp = get_disp(v, info.comm_size);
    MPI_Allgatherv(local.A, local.size, MPI_DOUBLE, global.A, v, disp, MPI_DOUBLE, comm);
    free(v);
    free(disp);
}
#endif //MATRIX_MYMPI_H
