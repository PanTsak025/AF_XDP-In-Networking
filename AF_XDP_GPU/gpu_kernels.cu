#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

#include "gpu_kernels.h"

#define GPU_MAX_IP_ENTRIES   65536u
#define EMPTY_KEY            0xFFFFFFFFu
#define DEFAULT_THRESHOLD    2000u

static uint8_t             *d_packet_buffer   = NULL;
static uint32_t            *d_offsets         = NULL;
static uint32_t            *d_lengths         = NULL;
static uint64_t            *d_epoch_secs      = NULL;
static uint8_t             *d_malicious_flags = NULL;
static struct gpu_pkt_meta *d_meta            = NULL;

/* Persistent per-IP state */
static uint32_t *d_ip_keys          = NULL;
static uint64_t *d_total_counts     = NULL;
static uint64_t *d_window_counts    = NULL;
static uint64_t *d_last_window      = NULL;
static uint32_t *d_malicious_ip     = NULL;   /* changed from uint8_t* to uint32_t* */

static uint32_t g_max_batch_size    = 0;
static uint32_t g_max_total_bytes   = 0;
static uint32_t g_threshold         = DEFAULT_THRESHOLD;

static int check_cuda(cudaError_t err, const char *where)
{
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", where, cudaGetErrorString(err));
        return -1;
    }
    return 0;
}

__device__ static inline uint16_t load_be16(const uint8_t *p)
{
    return ((uint16_t)p[0] << 8) | (uint16_t)p[1];
}

/*
 * Match CPU ip->saddr in-memory layout on little-endian systems.
 */
__device__ static inline uint32_t load_u32_cpu_layout(const uint8_t *p)
{
    return ((uint32_t)p[0]) |
           ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

__device__ static inline uint32_t hash_ip(uint32_t ip)
{
    uint32_t x = ip;
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

__device__ static inline uint64_t atomic_exchange_u64(uint64_t *addr, uint64_t val)
{
    unsigned long long *ptr = (unsigned long long *)addr;
    unsigned long long old = *ptr;
    unsigned long long assumed;

    do {
        assumed = old;
        old = atomicCAS(ptr, assumed, (unsigned long long)val);
    } while (old != assumed);

    return (uint64_t)old;
}

__device__ static inline uint64_t atomic_add_u64(uint64_t *addr, uint64_t val)
{
    return (uint64_t)atomicAdd((unsigned long long *)addr, (unsigned long long)val);
}

/*
 * Find existing or insert new slot.
 * Same basic semantics as CPU linear probing.
 * If table is full for a new IP, return failure and classify as non-malicious.
 */
__device__ static inline uint32_t find_or_insert_slot(uint32_t *keys, uint32_t src_ip)
{
    uint32_t idx = hash_ip(src_ip) % GPU_MAX_IP_ENTRIES;

    for (uint32_t i = 0; i < GPU_MAX_IP_ENTRIES; i++) {
        uint32_t slot = (idx + i) % GPU_MAX_IP_ENTRIES;
        uint32_t old = atomicCAS(&keys[slot], EMPTY_KEY, src_ip);

        if (old == EMPTY_KEY || old == src_ip)
            return slot;
    }

    return 0xFFFFFFFFu;
}

/*
 * Non-blocking update path.
 * Safer under single-IP floods than spin locks.
 */
__device__ static inline uint8_t update_source_counter_gpu(uint32_t *keys,
                                                           uint64_t *total_counts,
                                                           uint64_t *window_counts,
                                                           uint64_t *last_window,
                                                           uint32_t *malicious_ip,
                                                           uint32_t src_ip,
                                                           uint64_t now_sec,
                                                           uint32_t threshold)
{
    uint32_t slot = find_or_insert_slot(keys, src_ip);
    if (slot == 0xFFFFFFFFu)
        return 0;

    uint64_t prev_window = atomic_exchange_u64(&last_window[slot], now_sec);

    if (prev_window != now_sec) {
        atomic_exchange_u64(&window_counts[slot], 0ULL);
        atomicExch(&malicious_ip[slot], 0U);
    }

    uint64_t new_window = atomic_add_u64(&window_counts[slot], 1ULL) + 1ULL;
    atomic_add_u64(&total_counts[slot], 1ULL);

    if (new_window > (uint64_t)threshold) {
        atomicExch(&malicious_ip[slot], 1U);
        return 1;
    }

    return malicious_ip[slot] ? 1 : 0;
}

__global__ void classify_packets_kernel(const uint8_t *packet_buffer,
                                        const uint32_t *offsets,
                                        const uint32_t *lengths,
                                        const uint64_t *epoch_secs,
                                        uint32_t pkt_count,
                                        uint8_t *malicious_flags,
                                        struct gpu_pkt_meta *meta,
                                        uint32_t *ip_keys,
                                        uint64_t *total_counts,
                                        uint64_t *window_counts,
                                        uint64_t *last_window,
                                        uint32_t *malicious_ip,
                                        uint32_t threshold)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pkt_count)
        return;

    malicious_flags[i]   = 0;
    meta[i].src_ip       = 0;
    meta[i].pkt_len      = (uint16_t)lengths[i];
    meta[i].l4_proto     = 0;
    meta[i].valid        = 0;
    meta[i].eth_counted  = 0;
    meta[i].ipv4_counted = 0;
    meta[i].malformed    = 0;
    meta[i]._pad         = 0;

    const uint8_t *pkt = packet_buffer + offsets[i];
    uint32_t len = lengths[i];
    const uint8_t *data = pkt;
    const uint8_t *data_end = pkt + len;

    if (len < 14) {
        meta[i].malformed = 1;
        return;
    }

    if (data + 14 > data_end) {
        meta[i].malformed = 1;
        return;
    }

    meta[i].eth_counted = 1;

    uint16_t eth_proto = load_be16(pkt + 12);
    if (eth_proto != 0x0800) {
        return;
    }

    if (len < 14 + 20) {
        meta[i].malformed = 1;
        return;
    }

    const uint8_t *ip = pkt + 14;
    uint8_t version = ip[0] >> 4;
    uint8_t ihl = ip[0] & 0x0F;
    uint32_t ip_hdr_len = (uint32_t)ihl * 4;

    if (version != 4) {
        meta[i].malformed = 1;
        return;
    }

    if (ip_hdr_len < 20) {
        meta[i].malformed = 1;
        return;
    }

    if (ip + ip_hdr_len > data_end) {
        meta[i].malformed = 1;
        return;
    }

    meta[i].ipv4_counted = 1;
    meta[i].src_ip = load_u32_cpu_layout(ip + 12);
    meta[i].pkt_len = (uint16_t)len;
    meta[i].l4_proto = ip[9];

    if (meta[i].l4_proto == 6) {
        const uint8_t *tcp = ip + ip_hdr_len;
        if (tcp + 20 > data_end) {
            meta[i].malformed = 1;
            return;
        }
        meta[i].valid = 1;
    } else if (meta[i].l4_proto == 17) {
        const uint8_t *udp = ip + ip_hdr_len;
        if (udp + 8 > data_end) {
            meta[i].malformed = 1;
            return;
        }
        meta[i].valid = 1;
    } else {
        meta[i].valid = 1;
    }

    if (meta[i].valid) {
        malicious_flags[i] = update_source_counter_gpu(ip_keys,
                                                       total_counts,
                                                       window_counts,
                                                       last_window,
                                                       malicious_ip,
                                                       meta[i].src_ip,
                                                       epoch_secs[i],
                                                       threshold);
    }
}

extern "C" int gpu_init(uint32_t max_batch_size, uint32_t max_total_bytes)
{
    g_max_batch_size  = max_batch_size;
    g_max_total_bytes = max_total_bytes;
    g_threshold       = DEFAULT_THRESHOLD;

    if (check_cuda(cudaMalloc((void **)&d_packet_buffer, max_total_bytes),
                   "cudaMalloc d_packet_buffer") != 0)
        return -1;

    if (check_cuda(cudaMalloc((void **)&d_offsets, max_batch_size * sizeof(uint32_t)),
                   "cudaMalloc d_offsets") != 0)
        return -1;

    if (check_cuda(cudaMalloc((void **)&d_lengths, max_batch_size * sizeof(uint32_t)),
                   "cudaMalloc d_lengths") != 0)
        return -1;

    if (check_cuda(cudaMalloc((void **)&d_epoch_secs, max_batch_size * sizeof(uint64_t)),
                   "cudaMalloc d_epoch_secs") != 0)
        return -1;

    if (check_cuda(cudaMalloc((void **)&d_malicious_flags, max_batch_size * sizeof(uint8_t)),
                   "cudaMalloc d_malicious_flags") != 0)
        return -1;

    if (check_cuda(cudaMalloc((void **)&d_meta, max_batch_size * sizeof(struct gpu_pkt_meta)),
                   "cudaMalloc d_meta") != 0)
        return -1;

    if (check_cuda(cudaMalloc((void **)&d_ip_keys, GPU_MAX_IP_ENTRIES * sizeof(uint32_t)),
                   "cudaMalloc d_ip_keys") != 0)
        return -1;

    if (check_cuda(cudaMalloc((void **)&d_total_counts, GPU_MAX_IP_ENTRIES * sizeof(uint64_t)),
                   "cudaMalloc d_total_counts") != 0)
        return -1;

    if (check_cuda(cudaMalloc((void **)&d_window_counts, GPU_MAX_IP_ENTRIES * sizeof(uint64_t)),
                   "cudaMalloc d_window_counts") != 0)
        return -1;

    if (check_cuda(cudaMalloc((void **)&d_last_window, GPU_MAX_IP_ENTRIES * sizeof(uint64_t)),
                   "cudaMalloc d_last_window") != 0)
        return -1;

    if (check_cuda(cudaMalloc((void **)&d_malicious_ip, GPU_MAX_IP_ENTRIES * sizeof(uint32_t)),
                   "cudaMalloc d_malicious_ip") != 0)
        return -1;

    return gpu_reset_state();
}

extern "C" int gpu_reset_state(void)
{
    if (!d_ip_keys || !d_total_counts || !d_window_counts ||
        !d_last_window || !d_malicious_ip)
        return -1;

    if (check_cuda(cudaMemset(d_ip_keys, 0xFF, GPU_MAX_IP_ENTRIES * sizeof(uint32_t)),
                   "cudaMemset d_ip_keys") != 0)
        return -1;

    if (check_cuda(cudaMemset(d_total_counts, 0, GPU_MAX_IP_ENTRIES * sizeof(uint64_t)),
                   "cudaMemset d_total_counts") != 0)
        return -1;

    if (check_cuda(cudaMemset(d_window_counts, 0, GPU_MAX_IP_ENTRIES * sizeof(uint64_t)),
                   "cudaMemset d_window_counts") != 0)
        return -1;

    if (check_cuda(cudaMemset(d_last_window, 0, GPU_MAX_IP_ENTRIES * sizeof(uint64_t)),
                   "cudaMemset d_last_window") != 0)
        return -1;

    if (check_cuda(cudaMemset(d_malicious_ip, 0, GPU_MAX_IP_ENTRIES * sizeof(uint32_t)),
                   "cudaMemset d_malicious_ip") != 0)
        return -1;

    return 0;
}

extern "C" int gpu_set_threshold(uint32_t threshold)
{
    g_threshold = threshold;
    return 0;
}

extern "C" int gpu_classify_batch(const uint8_t *packet_buffer,
                                  const uint32_t *offsets,
                                  const uint32_t *lengths,
                                  const uint64_t *epoch_secs,
                                  uint32_t pkt_count,
                                  uint32_t total_bytes,
                                  uint8_t *malicious_flags_out,
                                  struct gpu_pkt_meta *meta_out)
{
    if (!packet_buffer || !offsets || !lengths || !epoch_secs ||
        !malicious_flags_out || !meta_out)
        return -1;

    if (pkt_count == 0)
        return 0;

    if (pkt_count > g_max_batch_size || total_bytes > g_max_total_bytes)
        return -1;

    if (check_cuda(cudaMemcpy(d_packet_buffer, packet_buffer, total_bytes, cudaMemcpyHostToDevice),
                   "cudaMemcpy packet_buffer H2D") != 0)
        return -1;

    if (check_cuda(cudaMemcpy(d_offsets, offsets, pkt_count * sizeof(uint32_t), cudaMemcpyHostToDevice),
                   "cudaMemcpy offsets H2D") != 0)
        return -1;

    if (check_cuda(cudaMemcpy(d_lengths, lengths, pkt_count * sizeof(uint32_t), cudaMemcpyHostToDevice),
                   "cudaMemcpy lengths H2D") != 0)
        return -1;

    if (check_cuda(cudaMemcpy(d_epoch_secs, epoch_secs, pkt_count * sizeof(uint64_t), cudaMemcpyHostToDevice),
                   "cudaMemcpy epoch_secs H2D") != 0)
        return -1;

    uint32_t threads = 256;
    uint32_t blocks = (pkt_count + threads - 1) / threads;

    classify_packets_kernel<<<blocks, threads>>>(d_packet_buffer,
                                                 d_offsets,
                                                 d_lengths,
                                                 d_epoch_secs,
                                                 pkt_count,
                                                 d_malicious_flags,
                                                 d_meta,
                                                 d_ip_keys,
                                                 d_total_counts,
                                                 d_window_counts,
                                                 d_last_window,
                                                 d_malicious_ip,
                                                 g_threshold);

    if (check_cuda(cudaGetLastError(), "kernel launch") != 0)
        return -1;

    if (check_cuda(cudaDeviceSynchronize(), "kernel sync") != 0)
        return -1;

    if (check_cuda(cudaMemcpy(malicious_flags_out, d_malicious_flags,
                              pkt_count * sizeof(uint8_t),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy malicious_flags D2H") != 0)
        return -1;

    if (check_cuda(cudaMemcpy(meta_out, d_meta,
                              pkt_count * sizeof(struct gpu_pkt_meta),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy meta D2H") != 0)
        return -1;

    return 0;
}

extern "C" void gpu_cleanup(void)
{
    if (d_packet_buffer)   cudaFree(d_packet_buffer);
    if (d_offsets)         cudaFree(d_offsets);
    if (d_lengths)         cudaFree(d_lengths);
    if (d_epoch_secs)      cudaFree(d_epoch_secs);
    if (d_malicious_flags) cudaFree(d_malicious_flags);
    if (d_meta)            cudaFree(d_meta);

    if (d_ip_keys)         cudaFree(d_ip_keys);
    if (d_total_counts)    cudaFree(d_total_counts);
    if (d_window_counts)   cudaFree(d_window_counts);
    if (d_last_window)     cudaFree(d_last_window);
    if (d_malicious_ip)    cudaFree(d_malicious_ip);

    d_packet_buffer   = NULL;
    d_offsets         = NULL;
    d_lengths         = NULL;
    d_epoch_secs      = NULL;
    d_malicious_flags = NULL;
    d_meta            = NULL;

    d_ip_keys         = NULL;
    d_total_counts    = NULL;
    d_window_counts   = NULL;
    d_last_window     = NULL;
    d_malicious_ip    = NULL;

    g_max_batch_size  = 0;
    g_max_total_bytes = 0;
    g_threshold       = DEFAULT_THRESHOLD;
}