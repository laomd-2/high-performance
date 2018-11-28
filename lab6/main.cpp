#pragma warning(disable: 4244)

#include <iostream>
#include <fstream>
#include <algorithm>
#include <divide.hpp>

using namespace std;

namespace std {
    template <>
    struct less<uint_fast64_t> {
        bool operator()(uint_fast64_t a, uint_fast64_t b) {
            return a < b;
        }
    };
}
int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    Comm_Info info(MPI_COMM_WORLD);

    ifstream fin(argv[1], ios::binary);
    uint_fast64_t n;
    read(fin, n);
    int k = atoi(argv[2]);
    n = pow(2, k) + 0.5;

    vector<uint_fast64_t> local = divide_read_directly(fin, n, MPI_COMM_WORLD);
    fin.close();
    stable_sort(local.begin(), local.end());

    int num_samples = info.comm_size * info.comm_size;
    vector<uint_fast64_t> sample = copy_every_n(local, n / num_samples);

    vector<uint_fast64_t> global_sample;
    Gatherv(sample, global_sample, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    vector<uint_fast64_t> pivots(info.comm_size - 1);
    if (info.rank == 0) {
        sort(global_sample.begin(), global_sample.end());
        pivots = copy_every_n(global_sample, info.comm_size, info.comm_size);
    }
    Bcast(pivots, MPI_UINT64_T, 0, MPI_COMM_WORLD);

    vector<vector<uint_fast64_t>> local_blocks = divide(local, pivots);
    vector<uint_fast64_t> local2;
    vector<int> v;
    for (int i = 0; i < info.comm_size; ++i) {
        auto&& tmp = Gatherv(local_blocks[i], local2, MPI_UINT64_T, i, MPI_COMM_WORLD);
        if (info.rank == i)
            v = tmp;
    }

    auto it = local2.begin();
    for (int j = 0; j < info.comm_size - 1; ++j) {
        it += v[j];
        inplace_merge(local2.begin(), it, it + v[j + 1]);
    }

    if (info.rank == 0)
        cout << k << ',' << info.comm_size << ',' << MPI_Wtime() - start << endl;
    MPI_Finalize();
    return 0;
}