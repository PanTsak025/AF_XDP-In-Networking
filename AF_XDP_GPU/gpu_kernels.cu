#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
 // THIS CODE IS FOR UNDERSTANDING SIMPLE DDOS, NEEDS LOGIC REFINING
#include "gpu_kernels.h"

static uint8_t  *d_packet_buffer = NULL;
static uint32_t *d_offsets = NULL;
static uint32_t *d_lengths = NULL;

static uint8_t  *d_malicious_flags = NULL;
static uint32_t *d_src_ips = NULL;
static uint8_t  *d_valid_flags = NULL;
static uint8_t  *d_is_ipv4 = NULL;
static uint8_t  *d_l4_proto = NULL;
static uint8_t  *d_malformed_flags = NULL;

static uint32_t g_max_batch_size = 0;
static uint32_t g_max_total_bytes = 0;

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

__device__ static inline uint32_t load_be32(const uint8_t *p)
{
    return ((uint32_t)p[0] << 24) |
           ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] << 8)  |
           (uint32_t)p[3];
}

__global__ void classify_packets_kernel(const uint8_t *packet_buffer,
                                        const uint32_t *offsets,
                                        const uint32_t *lengths,
                                        uint32_t pkt_count,
                                        uint8_t *malicious_flags,
                                        uint32_t *src_ips,
                                        uint8_t *valid_flags,
                                        uint8_t *is_ipv4,
                                        uint8_t *l4_proto,
                                        uint8_t *malformed_flags)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pkt_count)
        return;

    malicious_flags[i] = 0;
    src_ips[i] = 0;
    valid_flags[i] = 0;
    is_ipv4[i] = 0;
    l4_proto[i] = 0;
    malformed_flags[i] = 0;

    const uint8_t *pkt = packet_buffer + offsets[i];
    uint32_t len = lengths[i];

    if (len < 14) {
        malformed_flags[i] = 1;
        return;
    }

    valid_flags[i] = 1;

    /* Ethernet type field at bytes 12-13 */
    uint16_t eth_proto = load_be16(pkt + 12);

    if (eth_proto != 0x0800) {
        /* Not IPv4; still a valid Ethernet packet */
        return;
    }

    is_ipv4[i] = 1;

    if (len < 14 + 20) {
        malformed_flags[i] = 1;
        return;
    }

    const uint8_t *ip = pkt + 14;
    uint8_t version = ip[0] >> 4;
    uint8_t ihl = ip[0] & 0x0F;
    uint32_t ip_hdr_len = (uint32_t)ihl * 4;

    if (version != 4 || ip_hdr_len < 20) {
        malformed_flags[i] = 1;
        return;
    }

    if (14 + ip_hdr_len > len) {
        malformed_flags[i] = 1;
        return;
    }

    l4_proto[i] = ip[9];
    src_ips[i] = load_be32(ip + 12);

    /* Dummy behavior for now:
       all parsed packets are classified as non-malicious */
    malicious_flags[i] = 0;
}

extern "C" int gpu_init(uint32_t max_batch_size, uint32_t max_total_bytes)
{
    g_max_batch_size = max_batch_size;
    g_max_total_bytes = max_total_bytes;

    if (check_cuda(cudaMalloc((void **)&d_packet_buffer, max_total_bytes), "cudaMalloc d_packet_buffer") != 0)
        return -1;
    if (check_cuda(cudaMalloc((void **)&d_offsets, max_batch_size * sizeof(uint32_t)), "cudaMalloc d_offsets") != 0)
        return -1;
    if (check_cuda(cudaMalloc((void **)&d_lengths, max_batch_size * sizeof(uint32_t)), "cudaMalloc d_lengths") != 0)
        return -1;

    if (check_cuda(cudaMalloc((void **)&d_malicious_flags, max_batch_size * sizeof(uint8_t)), "cudaMalloc d_malicious_flags") != 0)
        return -1;
    if (check_cuda(cudaMalloc((void **)&d_src_ips, max_batch_size * sizeof(uint32_t)), "cudaMalloc d_src_ips") != 0)
        return -1;
    if (check_cuda(cudaMalloc((void **)&d_valid_flags, max_batch_size * sizeof(uint8_t)), "cudaMalloc d_valid_flags") != 0)
        return -1;
    if (check_cuda(cudaMalloc((void **)&d_is_ipv4, max_batch_size * sizeof(uint8_t)), "cudaMalloc d_is_ipv4") != 0)
        return -1;
    if (check_cuda(cudaMalloc((void **)&d_l4_proto, max_batch_size * sizeof(uint8_t)), "cudaMalloc d_l4_proto") != 0)
        return -1;
    if (check_cuda(cudaMalloc((void **)&d_malformed_flags, max_batch_size * sizeof(uint8_t)), "cudaMalloc d_malformed_flags") != 0)
        return -1;

    return 0;
}

extern "C" int gpu_classify_batch(const uint8_t *packet_buffer,
                                  const uint32_t *offsets,
                                  const uint32_t *lengths,
                                  uint32_t pkt_count,
                                  uint32_t total_bytes,
                                  uint8_t *malicious_flags_out,
                                  uint32_t *src_ips_out,
                                  uint8_t *valid_flags_out,
                                  uint8_t *is_ipv4_out,
                                  uint8_t *l4_proto_out,
                                  uint8_t *malformed_flags_out)
{
    if (!packet_buffer || !offsets || !lengths)
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

    uint32_t threads = 256;
    uint32_t blocks = (pkt_count + threads - 1) / threads;

    classify_packets_kernel<<<blocks, threads>>>(d_packet_buffer,
                                                 d_offsets,
                                                 d_lengths,
                                                 pkt_count,
                                                 d_malicious_flags,
                                                 d_src_ips,
                                                 d_valid_flags,
                                                 d_is_ipv4,
                                                 d_l4_proto,
                                                 d_malformed_flags);

    if (check_cuda(cudaGetLastError(), "kernel launch") != 0)
        return -1;

    if (check_cuda(cudaDeviceSynchronize(), "kernel sync") != 0)
        return -1;

    if (check_cuda(cudaMemcpy(malicious_flags_out, d_malicious_flags, pkt_count * sizeof(uint8_t), cudaMemcpyDeviceToHost),
                   "cudaMemcpy malicious_flags D2H") != 0)
        return -1;

    if (check_cuda(cudaMemcpy(src_ips_out, d_src_ips, pkt_count * sizeof(uint32_t), cudaMemcpyDeviceToHost),
                   "cudaMemcpy src_ips D2H") != 0)
        return -1;

    if (check_cuda(cudaMemcpy(valid_flags_out, d_valid_flags, pkt_count * sizeof(uint8_t), cudaMemcpyDeviceToHost),
                   "cudaMemcpy valid_flags D2H") != 0)
        return -1;

    if (check_cuda(cudaMemcpy(is_ipv4_out, d_is_ipv4, pkt_count * sizeof(uint8_t), cudaMemcpyDeviceToHost),
                   "cudaMemcpy is_ipv4 D2H") != 0)
        return -1;

    if (check_cuda(cudaMemcpy(l4_proto_out, d_l4_proto, pkt_count * sizeof(uint8_t), cudaMemcpyDeviceToHost),
                   "cudaMemcpy l4_proto D2H") != 0)
        return -1;

    if (check_cuda(cudaMemcpy(malformed_flags_out, d_malformed_flags, pkt_count * sizeof(uint8_t), cudaMemcpyDeviceToHost),
                   "cudaMemcpy malformed_flags D2H") != 0)
        return -1;

    return 0;
}

extern "C" void gpu_cleanup(void)
{
    if (d_packet_buffer) cudaFree(d_packet_buffer);
    if (d_offsets) cudaFree(d_offsets);
    if (d_lengths) cudaFree(d_lengths);
    if (d_malicious_flags) cudaFree(d_malicious_flags);
    if (d_src_ips) cudaFree(d_src_ips);
    if (d_valid_flags) cudaFree(d_valid_flags);
    if (d_is_ipv4) cudaFree(d_is_ipv4);
    if (d_l4_proto) cudaFree(d_l4_proto);
    if (d_malformed_flags) cudaFree(d_malformed_flags);

    d_packet_buffer = NULL;
    d_offsets = NULL;
    d_lengths = NULL;
    d_malicious_flags = NULL;
    d_src_ips = NULL;
    d_valid_flags = NULL;
    d_is_ipv4 = NULL;
    d_l4_proto = NULL;
    d_malformed_flags = NULL;

    g_max_batch_size = 0;
    g_max_total_bytes = 0;
}