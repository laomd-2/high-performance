//
// Created by laomd on 2018/10/23.
//

#ifndef MATRIX_MSG_HPP
#define MATRIX_MSG_HPP

#include <iostream>
#include <algorithm>
#include <vector>
using namespace std;

template <typename T>
void debug(const vector<T>& array) {
    for (auto i: array)
        if (i != T()) cout << i << ' ';
    cout << endl;
}

template <typename T>
ostream& operator<<(ostream& out, const vector<T>& array) {
    out.setf(ios::scientific);
    int cnt = count_if(array.begin(), array.end(), [](T i) { return i != T();});
    out << array.size() << ' ' << 1 << ' ' << cnt << endl;
    for (int i = 0; i < array.size(); i++)
        if (array[i] != T()) out << i + 1 << '\t' << 1 << '\t' << array[i] << endl;
    return out;
}

#endif //MATRIX_MSG_HPP
