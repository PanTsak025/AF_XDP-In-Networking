#ifndef GPU_KERNELS_H
#define GPU_KERNELS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct gpu_pkt_meta {
    uint32_t src_ip;
    uint16_t pkt_len;
    uint8_t  l4_proto;
    uint8_t  valid;
    uint8_t  eth_counted;
    uint8_t  ipv4_counted;
    uint8_t  malformed;
    uint8_t  _pad;
};

int gpu_init(uint32_t max_batch_size, uint32_t max_total_bytes);
int gpu_reset_state(void);
int gpu_set_threshold(uint32_t threshold);

int gpu_classify_batch(const uint8_t *packet_buffer,
                       const uint32_t *offsets,
                       const uint32_t *lengths,
                       const uint64_t *epoch_secs,
                       uint32_t pkt_count,
                       uint32_t total_bytes,
                       uint8_t *malicious_flags_out,
                       struct gpu_pkt_meta *meta_out);

void gpu_cleanup(void);

#ifdef __cplusplus
}
#endif

#endif