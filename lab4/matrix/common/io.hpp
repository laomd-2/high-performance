//
// Created by laomd on 2018/10/26.
//

#ifndef MATRIX_IO_HPP
#define MATRIX_IO_HPP

#include <istream>
#include <ostream>
using namespace std;

template <typename T>
ostream& write(ostream& out, T a) {
    return out.write(reinterpret_cast<char *>(&a), sizeof(a));
}

template <typename T>
istream& read(istream& in, T& a) {
    return in.read(reinterpret_cast<char *>(&a), sizeof(a));
}

#endif //MATRIX_IO_HPP
