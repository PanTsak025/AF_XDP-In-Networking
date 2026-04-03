#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda_runtime.h>
/*
    Two types of errors : 
        Launch Errors caught by cudeGetLastError()
        Runtime Errors caught when synchronizing and cpying memory
*/
#define CHECK_CUDA(call)                                                         \
    do {                                                                         \
        cudaError_t err__ = (call);                                              \
        if (err__ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err__));              \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)

/*
 * A simple packet feature batch in Struct-of-Arrays (SoA) form.
 * One thread will classify one packet.
 */
typedef struct {
    uint16_t *dst_port;
    uint8_t  *protocol;
    uint16_t *pkt_len;
    uint8_t  *flags;
} HostBatch;

typedef struct {
    uint16_t *dst_port;
    uint8_t  *protocol;
    uint16_t *pkt_len;
    uint8_t  *flags;
} DeviceBatch;

__global__ void classify_kernel(const uint16_t *dst_port,
                                const uint8_t *protocol,
                                const uint16_t *pkt_len,
                                uint8_t *flags,
                                int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) 
    {

        flags[i] = (dst_port[i] == 443 || pkt_len[i] > 1200);
    }
}

static void allocate_host_batch(HostBatch *batch, int n)
{
    size_t port_bytes = (size_t)n * sizeof(uint16_t);
    size_t proto_bytes = (size_t)n * sizeof(uint8_t);
    size_t len_bytes = (size_t)n * sizeof(uint16_t);
    size_t flag_bytes = (size_t)n * sizeof(uint8_t);

    /*
     * Pinned host memory helps async transfer performance.
     * This is useful later for stream-based pipelining.
     */
    CHECK_CUDA(cudaMallocHost((void **)&batch->dst_port, port_bytes));
    CHECK_CUDA(cudaMallocHost((void **)&batch->protocol, proto_bytes));
    CHECK_CUDA(cudaMallocHost((void **)&batch->pkt_len, len_bytes));
    CHECK_CUDA(cudaMallocHost((void **)&batch->flags, flag_bytes));
}

static void free_host_batch(HostBatch *batch)
{
    CHECK_CUDA(cudaFreeHost(batch->dst_port));
    CHECK_CUDA(cudaFreeHost(batch->protocol));
    CHECK_CUDA(cudaFreeHost(batch->pkt_len));
    CHECK_CUDA(cudaFreeHost(batch->flags));

    batch->dst_port = NULL;
    batch->protocol = NULL;
    batch->pkt_len = NULL;
    batch->flags = NULL;
}

static void allocate_device_batch(DeviceBatch *batch, int n)
{
    size_t port_bytes = (size_t)n * sizeof(uint16_t);
    size_t proto_bytes = (size_t)n * sizeof(uint8_t);
    size_t len_bytes = (size_t)n * sizeof(uint16_t);
    size_t flag_bytes = (size_t)n * sizeof(uint8_t);

    CHECK_CUDA(cudaMalloc((void **)&batch->dst_port, port_bytes));
    CHECK_CUDA(cudaMalloc((void **)&batch->protocol, proto_bytes));
    CHECK_CUDA(cudaMalloc((void **)&batch->pkt_len, len_bytes));
    CHECK_CUDA(cudaMalloc((void **)&batch->flags, flag_bytes));
}

static void free_device_batch(DeviceBatch *batch)
{
    CHECK_CUDA(cudaFree(batch->dst_port));
    CHECK_CUDA(cudaFree(batch->protocol));
    CHECK_CUDA(cudaFree(batch->pkt_len));
    CHECK_CUDA(cudaFree(batch->flags));

    batch->dst_port = NULL;
    batch->protocol = NULL;
    batch->pkt_len = NULL;
    batch->flags = NULL;
}

static void fill_demo_packets(HostBatch *batch, int n, int batch_id)
{
    int i;

    for (i = 0; i < n; i++) {
        /* Create mostly benign traffic. */
        batch->dst_port[i] = (i % 3 == 0) ? 443 : 53;
        batch->protocol[i] = 17; /* UDP by default */
        batch->pkt_len[i] = 60 + (uint16_t)(i % 80);
        batch->flags[i] = 0;
    }

    /* Inject some packets that should match the rule. */
    for (i = 0; i < n; i += 257) {
        batch->dst_port[i] = 80;
        batch->protocol[i] = 6;   /* TCP */
        batch->pkt_len[i] = 200 + (uint16_t)(batch_id % 20);
    }
}

static void enqueue_batch(cudaStream_t stream,
                          const HostBatch *h,
                          const DeviceBatch *d,
                          int n,
                          int threads_per_block)
{
    int num_blocks;
    size_t port_bytes = (size_t)n * sizeof(uint16_t);
    size_t proto_bytes = (size_t)n * sizeof(uint8_t);
    size_t len_bytes = (size_t)n * sizeof(uint16_t);
    size_t flag_bytes = (size_t)n * sizeof(uint8_t);

    num_blocks = (n + threads_per_block - 1) / threads_per_block;

    CHECK_CUDA(cudaMemcpyAsync(d->dst_port, h->dst_port, port_bytes,
                               cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d->protocol, h->protocol, proto_bytes,
                               cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d->pkt_len, h->pkt_len, len_bytes,
                               cudaMemcpyHostToDevice, stream));

    classify_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        d->dst_port, d->protocol, d->pkt_len, d->flags, n);

    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaMemcpyAsync(h->flags, d->flags, flag_bytes,
                               cudaMemcpyDeviceToHost, stream));
}

static int count_suspicious(const HostBatch *batch, int n)
{
    int i;
    int total = 0;

    for (i = 0; i < n; i++) {
        total += batch->flags[i] ? 1 : 0;
    }

    return total;
}

static void print_first_matches(const HostBatch *batch, int n, int max_to_print)
{
    int i;
    int printed = 0;

    for (i = 0; i < n && printed < max_to_print; i++) {
        if (batch->flags[i]) {
            printf("  suspicious packet at index %d: dst_port=%u proto=%u len=%u\n",
                   i,
                   (unsigned)batch->dst_port[i],
                   (unsigned)batch->protocol[i],
                   (unsigned)batch->pkt_len[i]);
            printed++;
        }
    }

    if (printed == 0) {
        printf("  no suspicious packets found in this batch\n");
    }
}

int main(void)
{
    enum { NUM_SLOTS = 2 };
    const int batch_size = 4096;
    const int threads_per_block = 256;
    const int total_batches = 6;

    HostBatch h[NUM_SLOTS];
    DeviceBatch d[NUM_SLOTS];
    cudaStream_t streams[NUM_SLOTS];
    int k;

    for (k = 0; k < NUM_SLOTS; k++) {
        allocate_host_batch(&h[k], batch_size);
        allocate_device_batch(&d[k], batch_size);
        CHECK_CUDA(cudaStreamCreate(&streams[k]));
    }

    /*
     * Double-buffered pipeline:
     * even batches use slot 0, odd batches use slot 1.
     */
    for (k = 0; k < total_batches; k++) {
        int slot = k % NUM_SLOTS;

        /*
         * If this slot was used two iterations ago, wait for its previous work
         * to finish before reusing the same host/device buffers.
         */
        if (k >= NUM_SLOTS) {
            int completed_batch = k - NUM_SLOTS;
            int completed_slot = completed_batch % NUM_SLOTS;
            int suspicious;

            CHECK_CUDA(cudaStreamSynchronize(streams[completed_slot]));

            suspicious = count_suspicious(&h[completed_slot], batch_size);
            printf("batch %d completed on slot %d: suspicious=%d\n",
                   completed_batch, completed_slot, suspicious);
            print_first_matches(&h[completed_slot], batch_size, 3);
        }

        fill_demo_packets(&h[slot], batch_size, k);
        enqueue_batch(streams[slot], &h[slot], &d[slot], batch_size,
                      threads_per_block);
    }

    /* Drain the final in-flight batches. */
    for (k = total_batches - NUM_SLOTS; k < total_batches; k++) {
        if (k >= 0) {
            int slot = k % NUM_SLOTS;
            int suspicious;

            CHECK_CUDA(cudaStreamSynchronize(streams[slot]));
            suspicious = count_suspicious(&h[slot], batch_size);
            printf("batch %d completed on slot %d: suspicious=%d\n",
                   k, slot, suspicious);
            print_first_matches(&h[slot], batch_size, 3);
        }
    }

    for (k = 0; k < NUM_SLOTS; k++) {
        CHECK_CUDA(cudaStreamDestroy(streams[k]));
        free_device_batch(&d[k]);
        free_host_batch(&h[k]);
    }

    return 0;
}
