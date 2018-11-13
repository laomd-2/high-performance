//
// Created by laomd on 2018/10/23.
//

#ifndef MATRIX_MSG_HPP
#define MATRIX_MSG_HPP

#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>
using namespace std;

template <typename T>
ostream& operator<<(ostream& out, const vector<T>& array) {
    for (auto i: array)
        cout << i << ' ';
    return out;
}

template <typename T>
ostream& operator<<(ostream& out, const vector<vector<T>>& array) {
    for (auto& x: array)
        out << x << endl;
    return out;
}

vector<uint_fast64_t> get_v(uint_fast64_t m, int comm_size) {
    vector<uint_fast64_t> v(comm_size);
    uint_fast64_t local_m = m / comm_size;
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
vector<T> copy_every_n(const vector<T> &a, uint_fast64_t n, uint_fast64_t offset = 0) {
    vector<T> result;
    uint_fast64_t tmp = 0;
    copy_if(a.begin() + offset, a.end(), std::back_inserter(result), [&](T num) {
        if (tmp == 0) {
            tmp = n - 1;
            return true;
        }
        tmp--;
        return false;

    });
    return result;
}

template <typename T>
vector<vector<T>> divide(const vector<T>& a, const vector<T>& d) {
    vector<vector<T>> res;
    auto it = a.begin();
    for (T x: d) {
        vector<T> tmp;
        while (it != a.end() && *it <= x) {
            tmp.push_back(*it);
            ++it;
        }
        res.push_back(tmp);
    }
    res.push_back(vector<T>(it, a.end()));
    return res;
}
#endif //MATRIX_MSG_HPP
