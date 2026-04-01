// Redirects packets from their RX queue[i] into an AF_XDP socket.
// The AF_XDP sockets are stored in an XSKMAP[i].

#include <linux/bpf.h>
#include <linux/if_ether.h>
#include <linux/ip.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_endian.h>

struct 
{
    __uint(type, BPF_MAP_TYPE_XSKMAP);   // type of map used by XDP to redirect packets into AF_XDP sockets
    __uint(max_entries, 64);   // number of queues
    __type(key, __u32);
    __type(value, __u32);
} xsks_map SEC(".maps");

SEC("xdp")
int xdp_sock_prog(struct xdp_md *ctx)
{
    void *data = (void *)(long)ctx->data;
    void *data_end = (void *)(long)ctx->data_end;

    struct ethhdr *eth = data;
    if ((void *)(eth + 1) > data_end)
        return XDP_PASS;

    if (bpf_ntohs(eth->h_proto) != ETH_P_IP)
        return XDP_PASS;

    struct iphdr *ip = (void *)(eth + 1);
    if ((void *)(ip + 1) > data_end)
        return XDP_PASS;

    if(ip->protocol == 1)
    {
        return XDP_PASS;
    }

    __u32 qid = ctx->rx_queue_index;   // get id of RX queue the packet arrived on

    return bpf_redirect_map(&xsks_map, qid, XDP_PASS); // redirect to xskmap[i], if there is no socket there, do XDP_PASS
}

char _license[] SEC("license") = "GPL";