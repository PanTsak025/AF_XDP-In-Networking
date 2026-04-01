#include <linux/bpf.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_endian.h>

struct {
    __uint(type, BPF_MAP_TYPE_XSKMAP);
    __uint(max_entries, 64);
    __type(key, __u32);
    __type(value, __u32);
} xsks_map SEC(".maps");

SEC("xdp")
int xdp_sock_prog(struct xdp_md *ctx)
{
    __u32 qid = ctx->rx_queue_index;

    if (bpf_map_lookup_elem(&xsks_map, &qid))
        return bpf_redirect_map(&xsks_map, qid, 0);

    return XDP_PASS;
}

char LICENSE[] SEC("license") = "GPL";