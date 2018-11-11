//
// Created by laomd on 2018/11/11.
//
#include <iostream>
#include <fstream>
#include <ctime>
#include <cmath>
#include <cstdlib>
#include <io.hpp>
using namespace std;

int main(int argc, char* argv[]) {
//    ofstream fout(argv[1], ios::binary);
    __int64 n = pow(2, atoi(argv[2])) + 0.5;
//    write(fout, n);
//    srand(time(NULL));
    __int64 x;
//    for (int i = 0; i < n; ++i) {
//        x = rand() % 100;
//        write(fout, x);
//    }
//    fout.close();

    ifstream fin(argv[1], ios::binary);
    read(fin, x);
    while (n--) {
        read(fin, x);
        cout << x << ' ';
    }
    cout << endl;
}

// 3 43 20 47 57 2 83 90 52 19 36 44 42 8 34 48 87 34 83 70 49 18 77 5 14 85 78 19 69 44 26 10