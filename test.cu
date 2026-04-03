#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void add_kernel(const int *a, const int *b, int *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // gives thread's index, parallelism from many threads running here.
    if  (i<n)  // safe check that memory does not read outside of bounds
    {
        c[i] = a[i] + b[i];
    }
}

int main(void)
{
    int n = 1000;
    int size = n * sizeof(int);

    int *h_a, *h_b, *h_c;
    int *d_a, *d_b, *d_c;

    h_a = (int*)malloc(size);  // better to use : cudaMallocHost(&h_a,size); (pin transfer)I should not overuse though, malloc is good.
    h_b = (int*)malloc(size);  // better to use : cudaMallocHost(&h_b,size);
    h_c = (int*)malloc(size);  // better to use : cudaMallocHost(&h_c,size);

    cudaMalloc(&d_a,size);
    cudaMalloc(&d_b,size);
    cudaMalloc(&d_c,size);

    for (int i=0;i<n;i++)
    {
        h_a[i] = i;
        h_b[i] = 2 * i;
    }

    cudaMemcpy(d_a,h_a,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b,size,cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int num_blocks = (threads_per_block + n - 1) / threads_per_block;

    add_kernel<<<num_blocks, threads_per_block>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_c,d_c,size,cudaMemcpyDeviceToHost);

    for (int i = 0; i < 5; i++) 
    {
        printf("c[%d] = %d\n", i,h_c[i]);
    }

    free(h_a);
    free(h_b);
    free(h_c);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return 0;
}