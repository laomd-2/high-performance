//
// Created by laomd on 2018/10/25.
//
#include <cstring>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <io.hpp>
using namespace std;

void test() {
    ifstream fin("../data/matrix2.mtx");

    char a[8], b[8];
    char c[30];

    ifstream::pos_type last = std::ios::beg;
    while (read(fin, a)) {
        read(fin, b);
        read(fin, c);
//        cout << a << ' ' << b << ' ' << c << endl;
        auto cur = fin.tellg();
//
        auto off = cur - last;
        cout << off << endl;
        last = cur;
    }
}

void transform() {
    ifstream fin("../test/matrix.mtx");
    ofstream fout("../test/matrix2.mtx");
    fout.sync_with_stdio(false);

    char a[8], b[8];
    char c[30];
    memset(a, 0, sizeof(a));
    memset(b, 0, sizeof(b));
    memset(c, 0, sizeof(c));
    while (fin >> a >> b >> c) {
//        cout << a << ' ' << b << ' ' << c << endl;
        fout.write(a, 8).write(b, 8).write(c, 30);
        memset(a, 0, sizeof(a));
        memset(b, 0, sizeof(b));
        memset(c, 0, sizeof(c));
    }
}

int main() {
    transform();
//    test();
    return 0;
}