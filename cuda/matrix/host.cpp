//
// Created by laomd on 2018/12/6.
//
#include <iostream>
using namespace std;

int main() {
    int i, j;
    int n = 16;
    for (i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            double sum = 0;
            for (int k = 0; k < n; ++k) {
                double a_ik = i - 0.1 * k + 1, b_kj = 0.2 * j - 0.1 * k;
                sum += a_ik * b_kj;
            }
            cout << sum << ' ';
        }
        cout << endl;
    }
}