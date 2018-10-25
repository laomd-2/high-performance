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
        istream& fin;
        vector<int>& v;;
        ThreadedQueue<pair<int, vector<MatrixElem>>>& q;
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
    istream& fin = param->fin;
    vector<int>& v = param->v;
    ThreadedQueue<pair<int, vector<MatrixElem>>>& q = param->q;

    vector<MatrixElem> buffer;
    buffer.reserve(v[0]);
    for (int rank = 0; rank < v.size(); rank++) {
        int balance = v[rank];
        buffer.resize(balance);
        for (int i = 0; i < balance; ++i) {
            fin >> buffer[i];
        }
        q.put(make_pair(rank, buffer));
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

vector<MatrixElem> divide_pipeline(istream& fin, int num_elems, MPI_Comm comm,
        const int num_cons=1) {
    Comm_Info info(comm);
    vector<int> balance = get_v(num_elems, info.comm_size);

    vector<MatrixElem> local_A(balance[info.rank]);
    if (info.rank == 0) {
        ThreadedQueue<pair<int, vector<MatrixElem>>> q;
        pthread_t tids[1 + num_cons];

        ProParam p1{fin, balance, q};
        pthread_create(tids, nullptr, producer, &(p1));

        vector<int> v = get_v(info.comm_size, num_cons);
        for (int j = 0; j < num_cons; ++j) {
            ConsParam p2{local_A, comm, v[j], q};
            pthread_create(tids + 1 + j, nullptr, consumer, &p2);
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
