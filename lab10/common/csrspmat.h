//
// Created by laomd on 2019/1/21.
//

#ifndef LAB10_CSRSPMAT_HPP
#define LAB10_CSRSPMAT_HPP

#include <istream>
#include <ostream>
#include <unordered_map>
#include <vector>
#include <map>
#include <iterator>
using namespace std;

struct HostCsrSpMat {
private:
    HostCsrSpMat(const map<pair<int, int>, float>& map_view, int nn): n(nn), row_indices(nn + 1, 0) {
        int nnz = map_view.size();
        data.reserve(nnz);
        col_indices.reserve(nnz);

        for (const auto& item: map_view) {
            if (item.second != 0.0f) {
                col_indices.push_back(item.first.second);
                data.push_back(item.second);
                row_indices[item.first.first]++;
            }
        }

        row_indices[n] = data.size() + 1;
        for (int i = 1; i < n; ++i)
            row_indices[i] += row_indices[i - 1];
    }

public:
    vector<float> data;
    vector<int> col_indices;
    vector<int> row_indices;
    int n;

    HostCsrSpMat() = default;

    HostCsrSpMat take(int row_first, int col_first, int n_per) {
        HostCsrSpMat sub;
        sub.n = n_per;
        int row_last = row_first + n_per, col_last = col_first + n_per;

        int first = row_first > 0 ? row_indices[row_first - 1] : 0;
        for (int row = row_first; row < row_last; ++row) {
            sub.row_indices.push_back(0);
            int last = row_indices[row];
            int col;
            float x;
            while (first < last) {
                col = col_indices[first];
                if (col >= col_first && col < col_last) {
                    ++sub.row_indices.back();
                    sub.col_indices.push_back(col - col_first);
                    sub.data.push_back(data[first]);
                }
                ++first;
            }
            first = last;
        }
        for (int l = 1; l < n_per; ++l)
            sub.row_indices[l] += sub.row_indices[l - 1];
        sub.row_indices.push_back(sub.data.size() + 1);
        return sub;
    }

    template <typename Visit>
    void for_each(Visit visit) const {
        int i = 0;
        for (int row = 0; row < row_indices.size(); ++row) {
            int end = row_indices[row];
            while (i < end && i < data.size()) {
                visit(row, col_indices[i], data[i]);
                i++;
            }
        }
    }

    HostCsrSpMat operator*(const HostCsrSpMat& b) const {
        map<pair<int, int>, float> map_view;
        for_each([&](int r, int c, float v) {
            int r_first = c > 0 ? b.row_indices[c - 1]: 0;
            int r_last = b.row_indices[c];
            while (r_first < r_last) {
                c = b.col_indices[r_first];
                map_view[make_pair(r, c)] += v * b.data[r_first];
                r_first++;
            }
        });
        return HostCsrSpMat(map_view, n);
    }

    HostCsrSpMat operator+(const HostCsrSpMat& b) const {
        map<pair<int, int>, float> map_view;
        for_each([&](int r, int c, float v) {
            map_view[make_pair(r, c)] = v;
        });
        b.for_each([&](int r, int c, float v) {
            map_view[make_pair(r, c)] += v;
        });
        return HostCsrSpMat(map_view, n);
    }

    friend istream&operator>>(istream&, HostCsrSpMat&);
};

istream &operator>>(istream &in, HostCsrSpMat &mat) {
    int nnz;
    in >> mat.n >> nnz;
    mat.row_indices.resize(mat.n + 1);
    mat.data.resize(nnz);
    mat.col_indices.resize(nnz);
    for (float& x: mat.data)
        in >> x;
    for (int& x: mat.col_indices)
        in >> x;
    for (int& x: mat.row_indices)
        in >> x;
    return in;
}

ostream &operator<<(ostream &out, const HostCsrSpMat &mat) {
    mat.for_each([&](int row, int col, float value) {
        out << "(" << row << "," << col << "," << value << ")" << endl;
    });
    return out;
}

#endif //LAB10_CSRSPMAT_HPP
