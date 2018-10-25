//
// Created by laomd on 2018/10/25.
//

#ifndef MATRIX_PRO_CONS_HPP
#define MATRIX_PRO_CONS_HPP

#include <pthread.h>
#include <queue>

template <typename T>
class ThreadedQueue {
    pthread_mutex_t lock;
    pthread_cond_t empty, full;
    int _max_size;
    std::queue<T> _queue;
    int test_num;
public:
    explicit ThreadedQueue(int max_size=INT_MAX) :
            _max_size(max_size), test_num(0) {
        pthread_mutex_init(&lock, 0);
        pthread_cond_init(&empty, 0);
        pthread_cond_init(&full, 0);
    }

    ~ThreadedQueue() {
        pthread_mutex_destroy(&lock);
        pthread_cond_destroy(&empty);
        pthread_cond_destroy(&full);
    }

    T get() {
        pthread_mutex_lock(&lock);
        while (_queue.empty()) {
            pthread_cond_wait(&empty, &lock);
        }
        T x = _queue.front();
        _queue.pop();
        test_num--;
        pthread_cond_signal(&full);
        pthread_mutex_unlock(&lock);
        return x;
    }

    void put(T x) {
        pthread_mutex_lock(&lock);
        while (_queue.size() == _max_size) {
            pthread_cond_wait(&full, &lock);
        }
        _queue.push(x);
        test_num++;
        pthread_cond_signal(&empty);
        pthread_mutex_unlock(&lock);
    }

    int test() const {
        return test_num;
    }
};
#endif //MATRIX_PRO_CONS_HPP
