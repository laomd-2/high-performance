//
// Created by laomd on 2018/12/6.
//
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <iostream>
#include "../util.cuh"

using namespace std;

int main() {
    thrust::host_vector<int> h(4);
    for (int i = 0; i < h.size(); ++i) {
        h[i] = i;
    }

    thrust::device_vector<int> d(h);
    cout << d << endl;

    thrust::device_vector<int> d2(10, 1);
    thrust::fill(d2.begin(), d2.begin() + 5, 2);
    cout << d2 << endl;
    return 0;
}