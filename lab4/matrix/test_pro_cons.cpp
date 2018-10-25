//
// Created by laomd on 2018/10/25.
//
#include <iostream>
#include <threaded_queue.hpp>
#include <vector_manip.hpp>
using namespace std;

void* pro(void* q) {
    auto * queue1 = (ThreadedQueue<int>*)q;
    for (int j = 0; j < 10; ++j) {
        queue1->put(j);
    }
    return 0;
}

void* cons(void* q) {
    auto * queue1 = (ThreadedQueue<int>*)q;
    for (int j = 0; j < 10; ++j) {
        queue1->get();
    }
    return 0;
}

int main() {
//    ThreadedQueue<int> queue1(100);
//    const int num_threads = 21;
//    pthread_t tids[num_threads];
//    for (int i = 0; i < num_threads / 3; ++i) {
//        pthread_create(tids + i, 0, cons, &queue1);
//    }
//    for (int i = num_threads / 3; i < num_threads; ++i) {
//        pthread_create(tids + i, 0, pro, &queue1);
//    }
//    for (pthread_t tid : tids) {
//        pthread_join(tid, 0);
//    }
//    cout << queue1.test() << endl;
    vector<int> v{1, 2,3,4,5,6,7};
    for (auto& i: to_2d(v, 1)) {
        debug(i);
    }
}