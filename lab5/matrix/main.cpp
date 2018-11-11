#include <iostream>
#include <fstream>
#include <tuple>
#include <divide.hpp>
#include <matrix_mul.hpp>
using namespace std;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    Comm_Info info(MPI_COMM_WORLD);

    MPI_Comm comm_row, comm_col;
    tie(comm_row, comm_col) = Cart_2d_sub(MPI_COMM_WORLD);
    Comm_Info row_info(comm_row);

    int m = atoi(argv[1]), n = atoi(argv[2]);

    vector<vector<double>> local_A = get_sub_matrix(m, n, comm_row, comm_col);
    vector<double> local_x = get_sub_vector(n, comm_row, comm_col);

    vector<double> local_y = dense_mat_vect_mult(local_A, local_x);

    auto row_y = Reduce(local_y, MPI_DOUBLE, MPI_SUM, 0, comm_row);
    if (row_info.rank == 0) {
        auto global_y = Gatherv(row_y, MPI_DOUBLE, m, 0, comm_col);
//        if (info.rank == 0) {
//            string file_name = "../result/" +
//                    to_string(m) + "x" + to_string(n) +
//                    "-" + to_string(info.comm_size) + ".txt";
//            ofstream res(file_name);
//            res << global_y;
//            res.close();
//        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (info.rank == 0)
        cout << m << ',' << n << ',' << info.comm_size << ',' << MPI_Wtime() - start << endl;
    MPI_Finalize();
    return 0;
}