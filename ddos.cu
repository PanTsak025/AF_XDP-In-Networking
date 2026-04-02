#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda_runtime.h>


__global__ void classify_kernel(const int16_t *dst_port,const int8_t *protocol,const int16_t *pkt_length, int *flags, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) 
    {
        flags[i] = (dst_port[i] == 80 && protocol[i] == 6 && pkt_length[i] > 100); 
    }
}