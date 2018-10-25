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


vector<int> get_v(int m, int comm_size) {
    vector<int> v(comm_size);
    int local_m = m / comm_size;
    int remain = m % comm_size;
    for (int i = 0; i < comm_size; ++i) {
        v[i] = local_m + (int)(i < remain);
    }
    return v;
}

vector<int> get_disp(const vector<int>& v, int comm_size) {
    vector<int> disp(comm_size);
    disp[0] = 0;
    for (int i = 1; i < comm_size; ++i) {
        disp[i] = disp[i - 1] + v[i - 1];
    }
    return disp;
}

template <typename T>
vector<vector<T>> to_2d(const vector<T>& arr, int n) {
    vector<vector<T>> res;

    vector<int> v = get_v(arr.size(), n);
    auto it = arr.begin();
    for (int i = 0; i < n; i++) {
        int b = v[i];
        res.push_back(vector<T>(it, it + b));
        it += b;
    }
    return res;
}

#endif //MATRIX_MSG_HPP
