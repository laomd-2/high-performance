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
    void recv(vector<T>& recvbuff, MPI_Datatype datatype, int source, int recvtag, MPI_Status* status) const {
        int recvcount;
        MPI_Recv(&recvcount, 1, MPI_INT, source, recvtag,
                     _comm, MPI_STATUS_IGNORE);
        recvbuff.resize(recvcount);
        MPI_Recv(recvbuff.data(), recvcount, datatype, source, recvtag, _comm, status);
    }

    template <typename T>
    void send(const vector<T>& sendbuff, MPI_Datatype datatype, int dest, int sendtag) const {
        int sendcount = sendbuff.size();
        MPI_Send(&sendcount, 1, MPI_INT, dest, sendtag,
                 _comm);
        MPI_Send(sendbuff.data(), sendcount, datatype, dest, sendtag, _comm);
    }

    template <typename T>
    void bcast_c(T* buffer, int n, MPI_Datatype datatype, int root) const {
        MPI_Bcast(buffer, n, datatype, root, _comm);
    }

    template <typename T>
    vector<T> scatter(vector<T>&& global, int n_per, MPI_Datatype datatype, int root) const {
        bcast_c(&n_per, 1, MPI_INT, root);
        vector<T> local(n_per);
        MPI_Scatter(global.data(), n_per, datatype, local.data(), n_per, datatype, root, _comm);
        return local;
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

    Communicator cart_create(int dim=2) const {
        MPI_Comm comm_cart;

        vector<int> dims(dim, 0), periods(dim, 0);
        MPI_Dims_create(comm_size, dim, dims.data());
        MPI_Cart_create(_comm, dim, dims.data(), periods.data(), 0, &comm_cart);
        return Communicator(comm_cart);
    }

    vector<Communicator> cart_sub(int dim=2) const {
        vector<MPI_Comm> res(dim);

        vector<int> remain_dims(dim, 0);
        remain_dims[0] = 1;
        MPI_Cart_sub(_comm, remain_dims.data(), res.data());
        for (int i = 1; i < dim; ++i) {
            remain_dims[i] = 1;
            remain_dims[i - 1] = 0;
            MPI_Cart_sub(_comm, remain_dims.data(), res.data() + i);
        }
        return vector<Communicator>(res.begin(), res.end());
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
