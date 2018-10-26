//
// Created by laomd on 2018/10/2.
//

#ifndef MATRIX_DISTRIBUTE_H
#define MATRIX_DISTRIBUTE_H

#include <istream>
#include <numeric>
#include <vector_manip.hpp>
#include <type.hpp>
#include <msg.hpp>
#include <mympi.hpp>
#include <io.hpp>
#include <threaded_queue.hpp>
using namespace std;

vector<MatrixElem> divide_onebyone(istream &fin, int num_elems, MPI_Comm comm) {
    Comm_Info info(comm);
    vector<int> balance = get_v(num_elems, info.comm_size);

    vector<MatrixElem> local_A(balance[info.rank]);

    if (info.rank == 0) {
        vector<MatrixElem> buffer;
        buffer.reserve(balance[0]);

        for (int i = 0; i < info.comm_size; i++) {
            buffer.resize(balance[i]);
            for (int j = 0; j < balance[i]; j++) {
                fin >> buffer[j];
            }
            if (balance[i]) {
                if (i == 0) {
                    local_A = buffer;
                } else {
                    Send(buffer, MPI_MATRIX_ELEM, i, i, comm);
                }
            }
        }
    } else {
        Recv(local_A, MPI_MATRIX_ELEM, 0, info.rank, comm, MPI_STATUS_IGNORE);
    }
    return local_A;
}

namespace {
    struct ProParam {
        string filename;
        int offset;
        const int* fields;
        vector<int> v;
        int begin_rank;
        ThreadedQueue<pair<int, vector<MatrixElem>>>* q;
    };

    struct ConsParam {
        vector<MatrixElem>& local_A;
        MPI_Comm comm;
        int work;
        ThreadedQueue<pair<int, vector<MatrixElem>>>& q;
    };
}

void *producer(void *para) {
    auto param = (ProParam*)para;
    ifstream fin(param->filename);
    const int* fields = param->fields;

    int line_width = (fields[0] + fields[1] + fields[2]);
    fin.seekg((param->offset + 1) * line_width, ios::beg);
    vector<int>& v = param->v;
    ThreadedQueue<pair<int, vector<MatrixElem>>>& q = *(param->q);

    vector<MatrixElem> buffer;
    buffer.reserve(v[0]);

    for (int rank = 0; rank < v.size(); rank++) {
        int balance = v[rank];
        buffer.resize(balance);

        auto * a = new char[fields[0]]{0};
        auto * b = new char[fields[1]]{0};
        auto * c = new char[fields[2]]{0};
        for (int i = 0; i < balance; ++i) {
            fin.read(a, fields[0]);
            fin.read(b, fields[1]);
            fin.read(c, fields[2]);
            buffer[i].i = atoi(a);
            buffer[i].j = atoi(b);
            buffer[i].value = atof(c);
//            cout << buffer[i] << endl;
        }
        q.put(make_pair(rank + param->begin_rank, buffer));
        delete[] a;
        delete[] b;
        delete[] c;
    }
    return nullptr;
}

void *consumer(void *para) {
    auto param = (ConsParam*)para;
    vector<MatrixElem>& local_A = param->local_A;
    MPI_Comm comm = param->comm;
    ThreadedQueue<pair<int, vector<MatrixElem>>>& q = param->q;

    for (int i = 0; i < param->work; ++i) {
        auto x = q.get();
        if (x.first == 0) {
            local_A = x.second;
        } else {
            Send(x.second, MPI_MATRIX_ELEM, x.first, x.first, comm);
        }
    }
    return nullptr;
}

vector<MatrixElem> divide_pipeline(const char *filename, const int* field_offsets,
        int num_elems, MPI_Comm comm,
        const int num_pros=1, const int num_cons=1) {
    Comm_Info info(comm);
    vector<int> balance = get_v(num_elems, info.comm_size);

    vector<MatrixElem> local_A(balance[info.rank]);
    if (info.rank == 0) {
        ThreadedQueue<pair<int, vector<MatrixElem>>> q;
        pthread_t tids[num_pros + num_cons];

        vector<int> v = get_v(info.comm_size, num_pros);
        auto it = balance.begin();
        int last = 0, last_rank = 0;
        ProParam params[num_pros];
        for (int i = 0; i < num_pros; ++i) {
            params[i].filename = filename;
            params[i].offset = last;
            params[i].fields = field_offsets;
            params[i].q = &q;
            params[i].v = vector<int>(it, it + v[i]);
            params[i].begin_rank = last_rank;
            pthread_create(tids + i, nullptr, producer, params + i);
            last = accumulate(it, it + v[i], last);
            it += v[i];
            last_rank += v[i];
        }

        v = get_v(info.comm_size, num_cons);
        for (int j = 0; j < num_cons; ++j) {
            ConsParam p2{local_A, comm, v[j], q};
            pthread_create(tids + num_pros + j, nullptr, consumer, &p2);
        }

        for (pthread_t tid : tids) {
            pthread_join(tid, nullptr);
        }
    } else {
        Recv(local_A, MPI_MATRIX_ELEM, 0, info.rank, comm, MPI_STATUS_IGNORE);
    }
    return local_A;
}

vector<MatrixElem> divide_scatter(istream &fin, int num_elems, MPI_Comm comm) {
    Comm_Info info(comm);
    vector<MatrixElem> array;
    if (info.rank == 0) {
        array = vector<MatrixElem>(num_elems);
        for (int i = 0; i < num_elems; i++)
            fin >> array[i];
    }
    vector<int> balance = get_v(num_elems, info.comm_size);
    vector<MatrixElem> local_A;

    if (balance[info.rank]) {
        local_A = vector<MatrixElem>(balance[info.rank]);
        Scatterv(array, local_A, MPI_MATRIX_ELEM, 0, MPI_COMM_WORLD);
    }
    return local_A;
}

#endif //MATRIX_DISTRIBUTE_H
