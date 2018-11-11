//
// Created by laomd on 2018/10/2.
//

#ifndef MATRIX_DISTRIBUTE_H
#define MATRIX_DISTRIBUTE_H

#include <istream>
#include <tuple>
#include <cmath>
#include <vector_manip.hpp>
#include <msg.hpp>
using namespace std;

vector<vector<double>> get_sub_matrix(int m, int n,
                                      MPI_Comm comm_row,
                                      MPI_Comm comm_col) {
    Comm_Info row_info(comm_row), col_info(comm_col);
    vector<int> row_balance = get_v(m, col_info.comm_size);
    vector<int> col_balance = get_v(n, row_info.comm_size);
    vector<int> begin_rows = prefix_sum(row_balance);
    vector<int> begin_cols = prefix_sum(col_balance);

    vector<vector<double>> local_A(row_balance[col_info.rank],
            vector<double >(col_balance[row_info.rank]));

    for (int i = 0; i < local_A.size(); ++i) {
        for (int j = 0; j < local_A[i].size(); ++j) {
            int real_i = begin_rows[col_info.rank] + i;
            int real_j = begin_cols[row_info.rank] + j;
            local_A[i][j] = real_i - 0.1 * real_j + 1;
        }
    }
    return local_A;
}

vector<double> get_sub_vector(int n, MPI_Comm comm_row, MPI_Comm comm_col) {
    Comm_Info row_info(comm_row), col_info(comm_col);
    vector<int> row_balance = get_v(n, col_info.comm_size);
    vector<int> begin_rows = prefix_sum(row_balance);

    vector<double> local_x(row_balance[col_info.rank]);
    if (row_info.rank == col_info.rank) {
        for (int i = 0; i < local_x.size(); ++i) {
            int real_i = begin_rows[col_info.rank] + i;
            local_x[i] = 0.1 * real_i;
        }
    }
    Bcast(local_x, MPI_DOUBLE, row_info.rank, comm_col);
    return local_x;
}

#endif //MATRIX_DISTRIBUTE_H
