//
// Created by laomd on 2018/12/6.
//

#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>
#include "../util.h"
using namespace std;

__global__ void hello() {
    int global_tid = getGlobalIdx_2D_2D();
    printf("Hello world from thread %d\n", global_tid);
}
int main() {
    dim3 grid_dim(2, 4), block_dim(8, 16);

    freopen("hello.txt","w",stdout);
    hello<<<grid_dim, block_dim>>>();
}