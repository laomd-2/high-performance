//
// Created by laomd on 2019/1/20.
//

#include <ctime>
#include <fstream>
#include <iostream>
#include "../common/csrspmat.h"
using namespace std;

int main(int argc, char* argv[]) {
    string filebase = "../data/csr_sparse";
    string file1 = filebase + argv[1] + ".mtx";
    string file2 = filebase + argv[1] + "-2.mtx";

    HostCsrSpMat mata, matb;
    ifstream fin;
    fin.open(file1);
    fin >> mata;
    fin.close();

    fin.open(file2);
    fin >> matb;
    fin.close();

    clock_t start = clock();

    int first, last, c;
    float v;
    map<pair<int, int>, float> c_tmp;
    mata.for_each([&](int row, int col, float val) {
        first = col > 0 ? matb.row_indices[col - 1] : 0;
        last = matb.row_indices[col];
        for (int i = first; i < last; ++i) {
            c = matb.col_indices[i];
            v = matb.data[i];
            c_tmp[make_pair(row, c)] += val * v;
        }
    });
    HostCsrSpMat result(c_tmp, mata.n);

    cout << "serial: " << (double)(clock() - start) / CLOCKS_PER_SEC << "s" << endl;
    return 0;
}