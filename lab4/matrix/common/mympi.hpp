//
// Created by laomd on 2018/10/23.
//

#ifndef MATRIX_MYMPI_H
#define MATRIX_MYMPI_H

#include <mpi.h>

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

#endif //MATRIX_MYMPI_H
