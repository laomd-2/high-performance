//
// Created by laomd on 2018/10/23.
//

#ifndef MATRIX_MYMPI_H
#define MATRIX_MYMPI_H

#include <mpi.h>

MPI_Datatype MPI_Matrix_Elem;
void Init(int* argc, char*** argv) {
    MPI_Init(argc, argv);

    int array_of_blocklengths[] = {1, 1, 1};

    MPI_Aint array_of_displacements[] = {0, sizeof(int), 2*sizeof(int)};
    MPI_Datatype array_of_datatypes[] = {MPI_INT, MPI_INT, MPI_FLOAT};

    MPI_Type_create_struct(3, array_of_blocklengths, array_of_displacements, array_of_datatypes, &MPI_Matrix_Elem);
    MPI_Type_commit(&MPI_Matrix_Elem);
}

void Finalize() {
    MPI_Type_free(&MPI_Matrix_Elem);
    MPI_Finalize();
}

#endif //MATRIX_MYMPI_H
