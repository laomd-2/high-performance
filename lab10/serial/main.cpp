//
// Created by laomd on 2019/1/20.
//

#include <ctime>
#include <iostream>
#include <spmat.h>
using namespace std;

int main(int argc, char* argv[]) {
    SpMat matrix_a(0, 0), matrix_b(0, 0), c(0, 0);
    string filebase = "../data/csr_sparse";
    string file1 = filebase + argv[1] + ".mtx";
    string file2 = filebase + argv[1] + "-2.mtx";
    read_csr_matrix(file1, matrix_a);
    read_csr_matrix(file2, matrix_b);
    clock_t start = clock();
    multiply_add(matrix_a, matrix_b, c);
    cout << "serial: " << (double)(clock() - start) / CLOCKS_PER_SEC << "s" << endl;
    return 0;
}