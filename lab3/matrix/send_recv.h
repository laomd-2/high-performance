//
// Created by laomd on 2018/10/1.
//

#ifndef MATRIX_SEND_RECV_H
#define MATRIX_SEND_RECV_H

void send_or_copy(
        const void* src_buffer,
        void* dest_buffer,
        int count,
        MPI_Datatype datatype,
        int dest,
        int tag,
        MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    if (dest == rank) {
        size_t size = sizeof(int);
        if (datatype == MPI_DOUBLE)
            size = sizeof(double);
        memcpy(dest_buffer, src_buffer, count * size);
    } else {
        MPI_Request request;
        MPI_Isend(src_buffer, count, datatype, dest, tag, comm, &request);
    }
}


void recv_divide(int *local_m, double **local_A, int n) {
    MPI_Recv(local_m, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    *local_A = malloc(*local_m * n * sizeof(double));
    MPI_Recv(*local_A, *local_m * n, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}


#endif //MATRIX_SEND_RECV_H
