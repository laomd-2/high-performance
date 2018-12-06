#include <stdlib.h>
#include <cuda_runtime.h>

int main() {
    int blk_size = 64;
    float* Md;
    size_t size = blk_size * blk_size * sizeof(float);

    auto *host_Md = new float[size];
    for (int i = 0; i < blk_size * blk_size; i++)
        host_Md[i] = i;

    cudaMalloc(&Md, size);
    cudaMemcpy(Md, host_Md, size, cudaMemcpyHostToDevice);

    free(host_Md);
    cudaFree(Md);
    
    return 0;
}