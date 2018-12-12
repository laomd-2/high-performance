//
// Created by laomd on 2018/12/6.
//
#include <iostream>
#include <vector>
#include <ctime>
using namespace std;

typedef vector<double> DVector1D;

void cal_one_ele(size_t i, size_t j, size_t n, double* P) {
    double sum = 0;
    for (int k = 0; k < n; ++k) {
        double a_ik = i - 0.1 * k + 1, b_kj = 0.2 * j - 0.1 * k;
        sum += a_ik * b_kj;
    }
    *(P + i * n + j) = sum;
}

int n = 5000;
DVector1D result(n * n, 0);

int main() {
    clock_t start = clock();

    int i, j;
    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            cal_one_ele(i, j, n, result.data());
        }
    }

    clock_t end = clock();
    cout << (double)(end - start) / CLOCKS_PER_SEC << endl;
}