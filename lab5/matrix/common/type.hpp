//
// Created by laomd on 2018/10/2.
//

#ifndef MATRIX_TYPE_H
#define MATRIX_TYPE_H

#include <mpi.h>
#include <istream>
#include <ostream>
using namespace std;

struct Comm_Info {
    int rank;
    int comm_size;

    explicit Comm_Info(MPI_Comm comm) {
        MPI_Comm_size(comm, &comm_size);
        MPI_Comm_rank(comm, &rank);
    }
};

struct MatrixElem {
    int i;
    int j;
    double value;
public:
    explicit MatrixElem(int x=-1, int y=-1, double v=0): i(x), j(y), value(v) { }

    friend istream& operator>>(istream& fin, MatrixElem&);

    friend ostream& operator<<(ostream& out, const MatrixElem&);

    bool operator==(const MatrixElem& other) const {
        return i == other.i && j == other.j && value == other.value;
    }

    bool operator!=(const MatrixElem& other) const {
        return !(*this == other);
    }
};

istream& operator>>(istream& fin, MatrixElem& a) {
    return fin >> a.i >> a.j >> a.value;
}

ostream& operator<<(ostream& out, const MatrixElem& a) {
    return out << "(" << a.i << ',' << a.j << ',' << a.value << ')';
}

#endif //MATRIX_TYPE_H
