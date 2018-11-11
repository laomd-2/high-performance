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
vector<T> prefix_sum(const vector<T>& arr) {
    vector<int> prefix_sums(arr.size() + 1, 0);
    for (int i = 0; i < arr.size(); ++i) {
        prefix_sums[i + 1] = prefix_sums[i] + arr[i];
    }
    return prefix_sums;
}

template <typename T>
vector<T> samples(const vector<T>& a, int n, int offset = 0) {
    vector<T> result;
    int tmp = 0;
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
