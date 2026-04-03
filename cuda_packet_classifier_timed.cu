#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#define NUM_SLOTS 2
#define NUM_BATCHES 8
#define BATCH_SIZE 4096
#define THREADS_PER_BLOCK 256

typedef struct {
    uint16_t *dst_port;
    uint8_t  *protocol;
    uint16_t *pkt_len;
    uint8_t  *flags;
} HostBuffers;

typedef struct {
    uint16_t *dst_port;
    uint8_t  *protocol;
    uint16_t *pkt_len;
    uint8_t  *flags;
} DeviceBuffers;

typedef struct {
    float h2d_ms;
    float kernel_ms;
    float d2h_ms;
} BatchTiming;

__global__ void classify_kernel(const uint16_t *dst_port,
                                const uint8_t *protocol,
                                const uint16_t *pkt_len,
                                uint8_t *flags,
                                int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        flags[i] = 
        (
            protocol[i] == 6 &&
            (dst_port[i] == 80 || dst_port[i] == 443) &&
            (pkt_len[i] > 100 && pkt_len[i] < 1400)
        );
    }
}

static void allocate_host_buffers(HostBuffers *h, int n)
{
    size_t port_bytes = (size_t)n * sizeof(uint16_t);
    size_t proto_bytes = (size_t)n * sizeof(uint8_t);
    size_t len_bytes = (size_t)n * sizeof(uint16_t);
    size_t flag_bytes = (size_t)n * sizeof(uint8_t);

    CHECK_CUDA(cudaMallocHost((void **)&h->dst_port, port_bytes));
    CHECK_CUDA(cudaMallocHost((void **)&h->protocol, proto_bytes));
    CHECK_CUDA(cudaMallocHost((void **)&h->pkt_len, len_bytes));
    CHECK_CUDA(cudaMallocHost((void **)&h->flags, flag_bytes));
}

static void free_host_buffers(HostBuffers *h)
{
    CHECK_CUDA(cudaFreeHost(h->dst_port));
    CHECK_CUDA(cudaFreeHost(h->protocol));
    CHECK_CUDA(cudaFreeHost(h->pkt_len));
    CHECK_CUDA(cudaFreeHost(h->flags));
}

static void allocate_device_buffers(DeviceBuffers *d, int n)
{
    size_t port_bytes = (size_t)n * sizeof(uint16_t);
    size_t proto_bytes = (size_t)n * sizeof(uint8_t);
    size_t len_bytes = (size_t)n * sizeof(uint16_t);
    size_t flag_bytes = (size_t)n * sizeof(uint8_t);

    CHECK_CUDA(cudaMalloc((void **)&d->dst_port, port_bytes));
    CHECK_CUDA(cudaMalloc((void **)&d->protocol, proto_bytes));
    CHECK_CUDA(cudaMalloc((void **)&d->pkt_len, len_bytes));
    CHECK_CUDA(cudaMalloc((void **)&d->flags, flag_bytes));
}

static void free_device_buffers(DeviceBuffers *d)
{
    CHECK_CUDA(cudaFree(d->dst_port));
    CHECK_CUDA(cudaFree(d->protocol));
    CHECK_CUDA(cudaFree(d->pkt_len));
    CHECK_CUDA(cudaFree(d->flags));
}

static void fill_fake_packet_batch(HostBuffers *h, int n, int batch_id)
{
    for (int i = 0; i < n; i++) {
        h->dst_port[i] = (i % 7 == 0) ? 80 : (uint16_t)(1024 + (i % 4000));
        h->protocol[i] = (i % 5 == 0) ? 6 : 17;  /* 6 = TCP, 17 = UDP */
        h->pkt_len[i] = (uint16_t)(60 + ((i * 37 + batch_id * 11) % 1400));
        h->flags[i] = 0;
    }
}

static void enqueue_batch(HostBuffers *h,
                          DeviceBuffers *d,
                          int n,
                          cudaStream_t stream,
                          BatchTiming *timing)
{
    size_t port_bytes = (size_t)n * sizeof(uint16_t);
    size_t proto_bytes = (size_t)n * sizeof(uint8_t);
    size_t len_bytes = (size_t)n * sizeof(uint16_t);
    size_t flag_bytes = (size_t)n * sizeof(uint8_t);

    int num_blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    cudaEvent_t start_h2d, stop_h2d;
    cudaEvent_t start_kernel, stop_kernel;
    cudaEvent_t start_d2h, stop_d2h;

    CHECK_CUDA(cudaEventCreate(&start_h2d));
    CHECK_CUDA(cudaEventCreate(&stop_h2d));
    CHECK_CUDA(cudaEventCreate(&start_kernel));
    CHECK_CUDA(cudaEventCreate(&stop_kernel));
    CHECK_CUDA(cudaEventCreate(&start_d2h));
    CHECK_CUDA(cudaEventCreate(&stop_d2h));

    CHECK_CUDA(cudaEventRecord(start_h2d, stream));
    CHECK_CUDA(cudaMemcpyAsync(d->dst_port, h->dst_port, port_bytes,
                               cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d->protocol, h->protocol, proto_bytes,
                               cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d->pkt_len, h->pkt_len, len_bytes,
                               cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaEventRecord(stop_h2d, stream));

    CHECK_CUDA(cudaEventRecord(start_kernel, stream));
    classify_kernel<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(
        d->dst_port, d->protocol, d->pkt_len, d->flags, n);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(stop_kernel, stream));

    CHECK_CUDA(cudaEventRecord(start_d2h, stream));
    CHECK_CUDA(cudaMemcpyAsync(h->flags, d->flags, flag_bytes,
                               cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaEventRecord(stop_d2h, stream));

    CHECK_CUDA(cudaEventSynchronize(stop_d2h));

    CHECK_CUDA(cudaEventElapsedTime(&timing->h2d_ms, start_h2d, stop_h2d));
    CHECK_CUDA(cudaEventElapsedTime(&timing->kernel_ms, start_kernel, stop_kernel));
    CHECK_CUDA(cudaEventElapsedTime(&timing->d2h_ms, start_d2h, stop_d2h));

    CHECK_CUDA(cudaEventDestroy(start_h2d));
    CHECK_CUDA(cudaEventDestroy(stop_h2d));
    CHECK_CUDA(cudaEventDestroy(start_kernel));
    CHECK_CUDA(cudaEventDestroy(stop_kernel));
    CHECK_CUDA(cudaEventDestroy(start_d2h));
    CHECK_CUDA(cudaEventDestroy(stop_d2h));
}

static int count_suspicious(const HostBuffers *h, int n)
{
    int count = 0;
    for (int i = 0; i < n; i++) {
        if (h->flags[i]) {
            count++;
        }
    }
    return count;
}

int main(void)
{
    HostBuffers hbuf[NUM_SLOTS];
    DeviceBuffers dbuf[NUM_SLOTS];
    cudaStream_t streams[NUM_SLOTS];
    BatchTiming timings[NUM_BATCHES];

    float total_h2d = 0.0f;
    float total_kernel = 0.0f;
    float total_d2h = 0.0f;

    for (int s = 0; s < NUM_SLOTS; s++) {
        allocate_host_buffers(&hbuf[s], BATCH_SIZE);
        allocate_device_buffers(&dbuf[s], BATCH_SIZE);
        CHECK_CUDA(cudaStreamCreate(&streams[s]));
    }

    for (int k = 0; k < NUM_BATCHES; k++) {
        int slot = k % NUM_SLOTS;

        fill_fake_packet_batch(&hbuf[slot], BATCH_SIZE, k);
        enqueue_batch(&hbuf[slot], &dbuf[slot], BATCH_SIZE, streams[slot], &timings[k]);

        {
            int suspicious = count_suspicious(&hbuf[slot], BATCH_SIZE);
            printf("batch %d (slot %d): suspicious=%d | H2D=%.3f ms | kernel=%.3f ms | D2H=%.3f ms\n",
                   k, slot, suspicious, timings[k].h2d_ms, timings[k].kernel_ms, timings[k].d2h_ms);
        }

        total_h2d += timings[k].h2d_ms;
        total_kernel += timings[k].kernel_ms;
        total_d2h += timings[k].d2h_ms;
    }

    printf("\nAverages over %d batches:\n", NUM_BATCHES);
    printf("  avg H2D    = %.3f ms\n", total_h2d / NUM_BATCHES);
    printf("  avg kernel = %.3f ms\n", total_kernel / NUM_BATCHES);
    printf("  avg D2H    = %.3f ms\n", total_d2h / NUM_BATCHES);

    for (int s = 0; s < NUM_SLOTS; s++) {
        CHECK_CUDA(cudaStreamDestroy(streams[s]));
        free_device_buffers(&dbuf[s]);
        free_host_buffers(&hbuf[s]);
    }

    return 0;
}
