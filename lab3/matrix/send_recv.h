//
// Created by laomd on 2018/10/1.
//

#ifndef MATRIX_SEND_RECV_H
#define MATRIX_SEND_RECV_H

#include "type.h"

DoubleArray recv_array(int n) {
    DoubleArray array;
    MPI_Recv(&array.size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    array.A = malloc(array.size * n * sizeof(double));
    MPI_Recv(array.A, array.size * n, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    return array;
}

void send_or_copy(
        DoubleArray src_buffer,
        DoubleArray *dest_buffer,
        int dest,
        int tag,
        MPI_Comm comm) {
    if (dest == 0) {
        copy(dest_buffer, src_buffer);
    } else {
        MPI_Request request;
        MPI_Isend(src_buffer.A, src_buffer.size, MPI_DOUBLE, dest, tag, comm, &request);
    }
}


#endif //MATRIX_SEND_RECV_H
