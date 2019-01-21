//
// Created by laomd on 2019/1/20.
//

#ifndef LAB10_MPI4CXX_HPP
#define LAB10_MPI4CXX_HPP

#include <mpi.h>
#include <vector>
using namespace std;


class Communicator {
    const MPI_Comm _comm;
    int comm_size{}, _rank{};
public:
    explicit Communicator(MPI_Comm comm): _comm(comm) {
        MPI_Comm_rank(comm, &_rank);
        MPI_Comm_size(comm, &comm_size);
    }

    int size() const {
        return comm_size;
    }

    int rank() const {
        return _rank;
    }

    template <typename T>
    vector<T> send_recv(vector<T>&& buffer, MPI_Datatype datatype,
            int dest, int sendtag, int source, MPI_Status* status) const {
        int recvcount, sendcount = buffer.size();
        MPI_Sendrecv(&sendcount, 1, MPI_INT, dest, sendtag - 1,
                &recvcount, 1, MPI_INT, source, MPI_ANY_TAG,
                _comm, MPI_STATUS_IGNORE);
        vector<T> recvbuff(recvcount);
        MPI_Sendrecv(buffer.data(), sendcount, datatype, dest, sendtag,
                recvbuff.data(), recvcount, datatype, source, MPI_ANY_TAG,
                _comm, status);
        return recvbuff;
    }

    template <typename T>
    void bcast_c(T* buffer, int n, MPI_Datatype datatype, int root) const {
        MPI_Bcast(buffer, n, datatype, root, _comm);
    }

    template <typename T>
    vector<T> scatterv(vector<T>&& global, const vector<int>& sendcounts, MPI_Datatype datatype, int root) const {
        int n;
        MPI_Scatter(sendcounts.data(), 1, MPI_INT, &n, 1, datatype, root, _comm);
        vector<T> local(n);
        vector<int> disp;
        if (_rank == root) {
            disp.push_back(0);
            for (int x: sendcounts) {
                disp.push_back(disp.back() + x);
            }
            disp.pop_back();
        }
        MPI_Scatterv(global.data(), sendcounts.data(), disp.data(), datatype, local.data(), n, datatype, root, _comm);
        return local;
    }

    void barrier() const {
        MPI_Barrier(_comm);
    }

    Communicator cart_create() const {
        MPI_Comm comm_cart;

        int dims[2] = {0, 0};
        MPI_Dims_create(comm_size, 2, dims);
        int periods[2] = {0, 0};
        MPI_Cart_create(_comm, 2, dims, periods, 0, &comm_cart);
        return Communicator(comm_cart);
    }

    pair<Communicator, Communicator> cart_2d_sub() const {
        MPI_Comm comm_row, comm_col;

        int remain_dims[2] = {0, 1};
        MPI_Cart_sub(_comm, remain_dims, &comm_row);
        remain_dims[0] = 1;
        remain_dims[1] = 0;
        MPI_Cart_sub(_comm, remain_dims, &comm_col);

        return make_pair(Communicator(comm_row), Communicator(comm_col));
    }

    void cart_shift(int dimension, int offset) const {
        int source, dest;
        MPI_Cart_shift(_comm, dimension, offset, &source, &dest);
    }

    void cart_coords(int rank, int* coord) const {
        MPI_Cart_coords(_comm, rank, 2, coord);
    }
};

#endif //LAB10_MPI4CXX_HPP
