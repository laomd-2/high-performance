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

struct Prefix {
    int begin_index;
    vector<int> range;
    int prefix_sum;
};

vector<Prefix> get_prefix(const vector<int>& balance, int num_pros) {
    vector<int> v = get_v(balance.size(), num_pros);
    auto it = balance.begin();
    int last = 0, last_rank = 0;
    vector<Prefix> params(num_pros);
    for (int i = 0; i < num_pros; ++i) {
        params[i].prefix_sum = last;
        params[i].range = vector<int>(it, it + v[i]);
        params[i].begin_index = last_rank;
        last = accumulate(it, it + v[i], last);
        it += v[i];
        last_rank += v[i];
    }
    return params;
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
