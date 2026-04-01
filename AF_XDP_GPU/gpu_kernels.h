#ifndef GPU_KERNELS_H
#define GPU_KERNELS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int gpu_init(uint32_t max_batch_size, uint32_t max_total_bytes);

int gpu_classify_batch(const uint8_t *packet_buffer,
                       const uint32_t *offsets,
                       const uint32_t *lengths,
                       uint32_t pkt_count,
                       uint32_t total_bytes,
                       uint8_t *malicious_flags_out,
                       uint32_t *src_ips_out,
                       uint8_t *valid_flags_out,
                       uint8_t *is_ipv4_out,
                       uint8_t *l4_proto_out,
                       uint8_t *malformed_flags_out);

void gpu_cleanup(void);

#ifdef __cplusplus
}
#endif

#endif