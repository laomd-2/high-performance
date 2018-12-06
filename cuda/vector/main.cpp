//
// Created by laomd on 2018/12/6.
//
#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <iostream>

using namespace thrust;
using namespace std;

int main() {
    host_vector<int> h(4);
    for (int i = 0; i < h.size(); ++i) {
        h[i] = i;
    }

    device_vector<int> d(h);
    for (auto i: d)
        cout << i << ' ';
    cout << endl;
    return 0;
}