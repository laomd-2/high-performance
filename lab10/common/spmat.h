//
// Created by laomd on 2019/1/20.
//

#ifndef LAB10_SPMAT_H
#define LAB10_SPMAT_H

#include <fstream>
#include <map>
#include <string>
#include <vector>
using namespace std;

struct MatrixElem {
    int row;
    int col;
    float value;
};

struct SpMat {
    vector<MatrixElem> values;
    int n;
    SpMat(size_t nn, size_t nnz): values(nnz), n(nn) { }
};

void read_csr_matrix(string filename, SpMat& matrix_a) {
    size_t nnz;
    ifstream fin(filename);
    if (fin.is_open()) {
        fin >> matrix_a.n >> nnz;
        matrix_a.values.resize(nnz, {0, 0, 0});
        for (MatrixElem& x: matrix_a.values)
            fin >> x.value;
        for (MatrixElem& x: matrix_a.values)
            fin >> x.col;
        int x, i = 0;
        for (int row = 0; row < matrix_a.n + 1; ++row) {
            fin >> x;
            while (i < x && i < nnz) {
                matrix_a.values[i].row = row;
                i++;
            }
        }
    }
}

void multiply_add(const SpMat &a, const SpMat &b, SpMat &c) {
    map<pair<int, int>, float> c_tmp;
    for (const MatrixElem& x: c.values)
        c_tmp[make_pair(x.row, x.col)] = x.value;
    c.values.clear();
    for (const MatrixElem& e1: a.values) {
        for (const MatrixElem& e2: b.values) {
            if (e1.col == e2.row) {
                c_tmp[make_pair(e1.row, e2.col)] += e1.value * e2.value;
            }
        }
    }

    for (const auto & item: c_tmp)
        if (item.second != 0.0f)
            c.values.push_back({item.first.first, item.first.second, item.second});
}

#endif //LAB10_SPMAT_H
