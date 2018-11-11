#include <iostream>
#include <fstream>
#include <algorithm>
#include <divide.hpp>

using namespace std;

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    Comm_Info info(MPI_COMM_WORLD);

    ifstream fin(argv[1], ios::binary);
    __int64 n;
    read(fin, n);
    n = pow(2, atoi(argv[2])) + 0.5;

    vector<__int64> local = divide_by_element(fin, n, MPI_COMM_WORLD);
    fin.close();
    sort(local.begin(), local.end());
    int num_samples = info.comm_size * info.comm_size;
    vector<__int64> sample = samples(local, n / num_samples);

    vector<__int64> global_sample;
    Gatherv(sample, global_sample, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    vector<__int64> pivots(info.comm_size - 1);
    if (info.rank == 0) {
        sort(global_sample.begin(), global_sample.end());
        pivots = samples(global_sample, info.comm_size, info.comm_size);
    }
    Bcast(pivots, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    vector<vector<__int64>> local_blocks = divide(local, pivots);
    vector<__int64> local2;
    vector<int> v;
    for (int i = 0; i < info.comm_size; ++i) {
        auto&& tmp = Gatherv(local_blocks[i], local2, MPI_LONG_LONG, i, MPI_COMM_WORLD);
        if (info.rank == i)
            v = tmp;
    }

    auto it = local2.begin();
    for (int j = 0; j < info.comm_size - 1; ++j) {
        it += v[j];
        inplace_merge(local2.begin(), it, it + v[j + 1]);
    }

    MPI_Finalize();
    return 0;
}