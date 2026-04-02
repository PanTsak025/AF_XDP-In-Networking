#define _GNU_SOURCE

#include <errno.h>
#include <net/if.h>
#include <arpa/inet.h>
#include <poll.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>

#include <linux/if_ether.h>
#include <linux/ip.h>
#include <linux/udp.h>
#include <linux/tcp.h>
#include <linux/if_link.h>

#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include <xdp/xsk.h>

#include "gpu_kernels.h"

#define NUM_FRAMES              4096
#define FRAME_SIZE              XSK_UMEM__DEFAULT_FRAME_SIZE
#define UMEM_SIZE               (NUM_FRAMES * FRAME_SIZE)

#define RX_RING_SIZE            1024
#define FQ_RING_SIZE            2048
#define CQ_RING_SIZE            2048

#define RX_BURST_SIZE           64
#define CLASSIFY_BATCH_SIZE     1024

#define MAX_IP_ENTRIES          65536
#define DDOS_PPS_THRESHOLD      2000

#define MAX_BATCH_PKTS          CLASSIFY_BATCH_SIZE
#define MAX_PKT_SIZE            FRAME_SIZE
#define MAX_BATCH_BYTES         (MAX_BATCH_PKTS * MAX_PKT_SIZE)

static volatile sig_atomic_t running = 1;

struct xsk_state {
    void *umem_area;

    struct xsk_umem *umem;
    struct xsk_ring_prod fq;
    struct xsk_ring_cons cq;

    struct xsk_socket *xsk;
    struct xsk_ring_cons rx;

    int ifindex;
    int queue_id;
    int xsk_map_fd;
};

uint64_t total_classification_time_ns = 0;
uint64_t total_classified_pkts = 0;

struct stats {
    uint64_t total_pkts;
    uint64_t total_bytes;
    uint64_t eth_pkts;
    uint64_t ipv4_pkts;
    uint64_t tcp_pkts;
    uint64_t udp_pkts;
    uint64_t other_l4_pkts;
    uint64_t malformed_pkts;
    uint64_t malicious_pkts;
    uint64_t dropped_fill_reserve;
    uint64_t dropped_batch_oversize;
    uint64_t gpu_submit_fail;
};

struct ip_counter {
    uint32_t src_ip;
    uint64_t total_count;
    uint64_t window_count;
    time_t last_window;
    uint8_t used;
    uint8_t malicious;
};

struct gpu_batch_host {
    uint8_t  packet_buffer[MAX_BATCH_BYTES];
    uint32_t offsets[MAX_BATCH_PKTS];
    uint32_t lengths[MAX_BATCH_PKTS];
    uint32_t count;
    uint32_t total_bytes;
};

struct gpu_result_host {
    uint8_t  malicious_flags[MAX_BATCH_PKTS];
    uint32_t src_ips[MAX_BATCH_PKTS];
    uint8_t  valid_flags[MAX_BATCH_PKTS];
    uint8_t  is_ipv4[MAX_BATCH_PKTS];
    uint8_t  l4_proto[MAX_BATCH_PKTS];
    uint8_t  malformed_flags[MAX_BATCH_PKTS];
};

static struct stats g_stats = {0};
static struct ip_counter g_ip_table[MAX_IP_ENTRIES];

static void on_sigint(int signo)
{
    (void)signo;
    running = 0;
}

static inline uint64_t get_time_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

static inline uint8_t *xsk_umem_get_data(void *umem_area, uint64_t addr)
{
    return (uint8_t *)umem_area + addr;
}

static uint32_t hash_ip(uint32_t ip)
{
    uint32_t x = ip;
    x ^= x >> 16;
    x *= 0x7feb352d;
    x ^= x >> 15;
    x *= 0x846ca68b;
    x ^= x >> 16;
    return x;
}

static struct ip_counter *get_ip_counter(uint32_t src_ip)
{
    uint32_t idx = hash_ip(src_ip) % MAX_IP_ENTRIES;

    for (uint32_t i = 0; i < MAX_IP_ENTRIES; i++) {
        struct ip_counter *entry = &g_ip_table[(idx + i) % MAX_IP_ENTRIES];

        if (!entry->used) {
            entry->used = 1;
            entry->src_ip = src_ip;
            entry->total_count = 0;
            entry->window_count = 0;
            entry->last_window = 0;
            entry->malicious = 0;
            return entry;
        }

        if (entry->src_ip == src_ip)
            return entry;
    }

    return NULL;
}

static void update_source_counter(uint32_t src_ip)
{
    struct ip_counter *entry = get_ip_counter(src_ip);
    if (!entry)
        return;

    time_t now = time(NULL);

    if (entry->last_window != now) {
        entry->window_count = 0;
        entry->last_window = now;
        entry->malicious = 0;
    }

    entry->window_count++;
    entry->total_count++;

    if (entry->window_count > DDOS_PPS_THRESHOLD)
        entry->malicious = 1;
}

static void reset_gpu_batch(struct gpu_batch_host *batch)
{
    batch->count = 0;
    batch->total_bytes = 0;
}

static void update_stats_from_gpu_results(const struct gpu_batch_host *batch,
                                          const struct gpu_result_host *result)
{
    for (uint32_t i = 0; i < batch->count; i++) {
        g_stats.total_pkts++;
        g_stats.total_bytes += batch->lengths[i];

        if (result->malformed_flags[i]) {
            g_stats.malformed_pkts++;
            continue;
        }

        if (result->valid_flags[i])
            g_stats.eth_pkts++;

        if (result->is_ipv4[i])
            g_stats.ipv4_pkts++;

        if (result->l4_proto[i] == IPPROTO_TCP)
            g_stats.tcp_pkts++;
        else if (result->l4_proto[i] == IPPROTO_UDP)
            g_stats.udp_pkts++;
        else if (result->is_ipv4[i])
            g_stats.other_l4_pkts++;

        if (result->is_ipv4[i] && result->src_ips[i] != 0) {
            update_source_counter(result->src_ips[i]);
            
            // GPU's decision
            if (result->malicious_flags[i])
                g_stats.malicious_pkts++;
        }
    }
}

static int recycle_batch_addrs(struct xsk_state *xsks, uint64_t *addrs, uint32_t count)
{
    uint32_t idx_fq = 0;
    int reserved = xsk_ring_prod__reserve(&xsks->fq, count, &idx_fq);
    if (reserved != (int)count) {
        g_stats.dropped_fill_reserve++;
        return -1;
    }

    for (uint32_t i = 0; i < count; i++)
        *xsk_ring_prod__fill_addr(&xsks->fq, idx_fq + i) = addrs[i];

    xsk_ring_prod__submit(&xsks->fq, count);
    return 0;
}

static int flush_gpu_batch(struct xsk_state *xsks,
                           struct gpu_batch_host *batch,
                           struct gpu_result_host *result,
                           uint64_t *recycle_addrs)
{
    if (batch->count == 0)
        return 0;

    if (gpu_classify_batch(batch->packet_buffer,
                           batch->offsets,
                           batch->lengths,
                           batch->count,
                           batch->total_bytes,
                           result->malicious_flags,
                           result->src_ips,
                           result->valid_flags,
                           result->is_ipv4,
                           result->l4_proto,
                           result->malformed_flags) != 0) {
        g_stats.gpu_submit_fail++;
        return -1;
    }

    update_stats_from_gpu_results(batch, result);

    if (recycle_batch_addrs(xsks, recycle_addrs, batch->count) < 0)
        return -1;

    reset_gpu_batch(batch);
    return 0;
}

static int populate_fill_ring(struct xsk_state *xsks)
{
    uint32_t idx;
    uint32_t to_fill = FQ_RING_SIZE;

    int ret = xsk_ring_prod__reserve(&xsks->fq, to_fill, &idx);
    if (ret != (int)to_fill)
        return -1;

    for (uint32_t i = 0; i < to_fill; i++)
        *xsk_ring_prod__fill_addr(&xsks->fq, idx + i) = i * FRAME_SIZE;

    xsk_ring_prod__submit(&xsks->fq, to_fill);
    return 0;
}

static int setup_umem_and_socket(struct xsk_state *xsks, const char *ifname)
{
    int ret;

    struct xsk_umem_config umem_cfg = {
        .fill_size = FQ_RING_SIZE,
        .comp_size = CQ_RING_SIZE,
        .frame_size = FRAME_SIZE,
        .frame_headroom = 0,
        .flags = 0,
    };

    struct xsk_socket_config xsk_cfg = {
        .rx_size = RX_RING_SIZE,
        .tx_size = 0,
        .libbpf_flags = XSK_LIBBPF_FLAGS__INHIBIT_PROG_LOAD,
        .xdp_flags = 0,
        .bind_flags = 0,
    };

    xsks->ifindex = if_nametoindex(ifname);
    if (xsks->ifindex == 0) {
        perror("if_nametoindex");
        return -1;
    }

    ret = posix_memalign(&xsks->umem_area, getpagesize(), UMEM_SIZE);
    if (ret != 0) {
        fprintf(stderr, "posix_memalign failed: %s\n", strerror(ret));
        return -1;
    }
    memset(xsks->umem_area, 0, UMEM_SIZE);

    ret = xsk_umem__create(&xsks->umem,
                           xsks->umem_area,
                           UMEM_SIZE,
                           &xsks->fq,
                           &xsks->cq,
                           &umem_cfg);
    if (ret) {
        fprintf(stderr, "xsk_umem__create failed: %d\n", ret);
        return -1;
    }

    ret = xsk_socket__create(&xsks->xsk,
                             ifname,
                             xsks->queue_id,
                             xsks->umem,
                             &xsks->rx,
                             NULL,
                             &xsk_cfg);
    if (ret) {
        fprintf(stderr, "xsk_socket__create failed: %d\n", ret);
        return -1;
    }

    return populate_fill_ring(xsks);
}

static int load_and_attach_xdp(const char *obj_path,
                               const char *ifname,
                               struct bpf_object **obj_out,
                               int *xsk_map_fd_out)
{
    struct bpf_object *obj;
    struct bpf_program *prog;
    struct bpf_map *map;
    int ifindex, prog_fd, err;

    ifindex = if_nametoindex(ifname);
    if (ifindex == 0) {
        perror("if_nametoindex");
        return -1;
    }

    obj = bpf_object__open_file(obj_path, NULL);
    if (libbpf_get_error(obj))
        return -1;

    err = bpf_object__load(obj);
    if (err) {
        bpf_object__close(obj);
        return -1;
    }

    prog = bpf_object__find_program_by_name(obj, "xdp_sock_prog");
    if (!prog) {
        bpf_object__close(obj);
        return -1;
    }

    prog_fd = bpf_program__fd(prog);

    err = bpf_xdp_attach(ifindex, prog_fd, XDP_FLAGS_UPDATE_IF_NOEXIST, NULL);
    if (err) {
        bpf_object__close(obj);
        return -1;
    }

    map = bpf_object__find_map_by_name(obj, "xsks_map");
    if (!map) {
        bpf_xdp_detach(ifindex, XDP_FLAGS_UPDATE_IF_NOEXIST, NULL);
        bpf_object__close(obj);
        return -1;
    }

    *xsk_map_fd_out = bpf_map__fd(map);
    *obj_out = obj;
    return 0;
}

static int register_socket_in_xskmap(struct xsk_state *xsks)
{
    int sock_fd = xsk_socket__fd(xsks->xsk);
    uint32_t key = xsks->queue_id;

    if (bpf_map_update_elem(xsks->xsk_map_fd, &key, &sock_fd, 0) != 0) {
        perror("bpf_map_update_elem");
        return -1;
    }

    return 0;
}

static void print_periodic_stats(time_t start_ts)
{
    time_t now = time(NULL);
    double elapsed = difftime(now, start_ts);
    if (elapsed <= 0.0)
        elapsed = 1.0;

    double pps = (double)g_stats.total_pkts / elapsed;
    double bps = ((double)g_stats.total_bytes * 8.0) / elapsed;

    printf("\n=== AF_XDP GPU-batched stats ===\n");
    printf("elapsed_sec           : %.0f\n", elapsed);
    printf("total_pkts            : %llu\n", (unsigned long long)g_stats.total_pkts);
    printf("total_bytes           : %llu\n", (unsigned long long)g_stats.total_bytes);
    printf("pps                   : %.2f\n", pps);
    printf("bps                   : %.2f\n", bps);
    printf("eth_pkts              : %llu\n", (unsigned long long)g_stats.eth_pkts);
    printf("ipv4_pkts             : %llu\n", (unsigned long long)g_stats.ipv4_pkts);
    printf("tcp_pkts              : %llu\n", (unsigned long long)g_stats.tcp_pkts);
    printf("udp_pkts              : %llu\n", (unsigned long long)g_stats.udp_pkts);
    printf("other_l4_pkts         : %llu\n", (unsigned long long)g_stats.other_l4_pkts);
    printf("malformed_pkts        : %llu\n", (unsigned long long)g_stats.malformed_pkts);
    printf("malicious_pkts        : %llu\n", (unsigned long long)g_stats.malicious_pkts);
    printf("fill_reserve_fail     : %llu\n", (unsigned long long)g_stats.dropped_fill_reserve);
    printf("dropped_batch_oversize: %llu\n", (unsigned long long)g_stats.dropped_batch_oversize);
    printf("gpu_submit_fail       : %llu\n", (unsigned long long)g_stats.gpu_submit_fail);
    printf("rx_burst_size         : %d\n", RX_BURST_SIZE);
    printf("classify_batch        : %d\n", CLASSIFY_BATCH_SIZE);
    printf("threshold_pps         : %d\n", DDOS_PPS_THRESHOLD);

    if (total_classified_pkts > 0) {
        double avg_proc_latency_ns =
            (double)total_classification_time_ns / (double)total_classified_pkts;
        double avg_proc_latency_us = avg_proc_latency_ns / 1000.0;
        double avg_proc_latency_ms = avg_proc_latency_ns / 1000000.0;

        printf("avg_proc_latency_ns   : %.2f\n", avg_proc_latency_ns);
        printf("avg_proc_latency_us   : %.3f\n", avg_proc_latency_us);
        printf("avg_proc_latency_ms   : %.6f\n", avg_proc_latency_ms);
    }
}

static void print_top_suspects(void)
{
    printf("top malicious/active sources:\n");

    for (int round = 0; round < 10; round++) {
        uint64_t best_count = 0;
        int best_idx = -1;

        for (int i = 0; i < MAX_IP_ENTRIES; i++) {
            if (!g_ip_table[i].used)
                continue;
            if (g_ip_table[i].total_count > best_count) {
                best_count = g_ip_table[i].total_count;
                best_idx = i;
            }
        }

        if (best_idx < 0 || best_count == 0)
            break;

        char ipbuf[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &g_ip_table[best_idx].src_ip, ipbuf, sizeof(ipbuf));

        printf("  src=%s total=%llu window=%llu malicious=%u\n",
               ipbuf,
               (unsigned long long)g_ip_table[best_idx].total_count,
               (unsigned long long)g_ip_table[best_idx].window_count,
               g_ip_table[best_idx].malicious);

        g_ip_table[best_idx].total_count = 0;
    }
}

static void rx_loop(struct xsk_state *xsks)
{
    struct pollfd fds[1];
    fds[0].fd = xsk_socket__fd(xsks->xsk);
    fds[0].events = POLLIN;

    time_t start_ts = time(NULL);
    time_t last_print = start_ts;

    struct gpu_batch_host batch;
    struct gpu_result_host result;
    uint64_t recycle_addrs[CLASSIFY_BATCH_SIZE];

    uint64_t batch_start_ns = 0;
    int batch_timer_active = 0;

    reset_gpu_batch(&batch);
    memset(&result, 0, sizeof(result));

    while (running) {
        int pret = poll(fds, 1, 1000);
        if (pret < 0) {
            if (errno == EINTR)
                continue;
            perror("poll");
            break;
        }

        uint32_t idx_rx = 0;
        unsigned int rcvd = xsk_ring_cons__peek(&xsks->rx, RX_BURST_SIZE, &idx_rx);

        if (rcvd > 0) {
            for (unsigned int i = 0; i < rcvd; i++) {
                const struct xdp_desc *desc =
                    xsk_ring_cons__rx_desc(&xsks->rx, idx_rx + i);

                uint8_t *pkt = xsk_umem_get_data(xsks->umem_area, desc->addr);
                uint32_t len = desc->len;

                if (batch.count == 0 && !batch_timer_active) {
                    batch_start_ns = get_time_ns();
                    batch_timer_active = 1;
                }

                if (len > MAX_PKT_SIZE) {
                    g_stats.dropped_batch_oversize++;
                    /* still recycle */
                    uint64_t tmp = desc->addr;
                    (void)recycle_batch_addrs(xsks, &tmp, 1);
                    continue;
                }

                if (batch.count >= CLASSIFY_BATCH_SIZE ||
                    batch.total_bytes + len > MAX_BATCH_BYTES) {
                    unsigned int flushed_pkts = batch.count;

                    if (flush_gpu_batch(xsks, &batch, &result, recycle_addrs) < 0) {
                        xsk_ring_cons__release(&xsks->rx, rcvd);
                        return;
                    }

                    if (flushed_pkts > 0) {
                        uint64_t batch_end_ns = get_time_ns();
                        total_classification_time_ns += (batch_end_ns - batch_start_ns);
                        total_classified_pkts += flushed_pkts;
                        batch_timer_active = 0;
                    }

                    if (!batch_timer_active) {
                        batch_start_ns = get_time_ns();
                        batch_timer_active = 1;
                    }
                }

                batch.offsets[batch.count] = batch.total_bytes;
                batch.lengths[batch.count] = len;
                memcpy(&batch.packet_buffer[batch.total_bytes], pkt, len);
                recycle_addrs[batch.count] = desc->addr;
                batch.total_bytes += len;
                batch.count++;

                if (batch.count == CLASSIFY_BATCH_SIZE) {
                    unsigned int flushed_pkts = batch.count;

                    if (flush_gpu_batch(xsks, &batch, &result, recycle_addrs) < 0) {
                        xsk_ring_cons__release(&xsks->rx, rcvd);
                        return;
                    }

                    uint64_t batch_end_ns = get_time_ns();
                    total_classification_time_ns += (batch_end_ns - batch_start_ns);
                    total_classified_pkts += flushed_pkts;
                    batch_timer_active = 0;
                }
            }

            xsk_ring_cons__release(&xsks->rx, rcvd);
        }

        time_t now = time(NULL);

        if (batch.count > 0 && (now - last_print >= 1)) {
            unsigned int flushed_pkts = batch.count;

            if (flush_gpu_batch(xsks, &batch, &result, recycle_addrs) < 0)
                break;

            uint64_t batch_end_ns = get_time_ns();
            total_classification_time_ns += (batch_end_ns - batch_start_ns);
            total_classified_pkts += flushed_pkts;
            batch_timer_active = 0;
        }

        if (now - last_print >= 1) {
            print_periodic_stats(start_ts);
            last_print = now;
        }
    }

    if (batch.count > 0) {
        unsigned int flushed_pkts = batch.count;

        (void)flush_gpu_batch(xsks, &batch, &result, recycle_addrs);

        uint64_t batch_end_ns = get_time_ns();
        total_classification_time_ns += (batch_end_ns - batch_start_ns);
        total_classified_pkts += flushed_pkts;
        batch_timer_active = 0;
    }

    print_periodic_stats(start_ts);
    print_top_suspects();
}

static void cleanup(struct xsk_state *xsks, struct bpf_object *obj, const char *ifname)
{
    int ifindex = if_nametoindex(ifname);
    if (ifindex > 0)
        bpf_xdp_detach(ifindex, XDP_FLAGS_UPDATE_IF_NOEXIST, NULL);

    if (xsks->xsk)
        xsk_socket__delete(xsks->xsk);

    if (xsks->umem)
        xsk_umem__delete(xsks->umem);

    free(xsks->umem_area);

    if (obj)
        bpf_object__close(obj);

    gpu_cleanup();
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: %s <ifname> <queue_id>\n", argv[0]);
        return 1;
    }

    signal(SIGINT, on_sigint);
    signal(SIGTERM, on_sigint);

    const char *ifname = argv[1];
    int queue_id = atoi(argv[2]);

    struct xsk_state xsks = {
        .queue_id = queue_id,
    };

    struct bpf_object *obj = NULL;

    if (load_and_attach_xdp("./xdp_redirect_kern.o", ifname, &obj, &xsks.xsk_map_fd) < 0) {
        fprintf(stderr, "failed to load/attach XDP program\n");
        return 1;
    }

    if (setup_umem_and_socket(&xsks, ifname) < 0) {
        fprintf(stderr, "failed to setup UMEM/socket\n");
        cleanup(&xsks, obj, ifname);
        return 1;
    }

    if (register_socket_in_xskmap(&xsks) < 0) {
        fprintf(stderr, "failed to register socket in xsks_map\n");
        cleanup(&xsks, obj, ifname);
        return 1;
    }

    if (gpu_init(CLASSIFY_BATCH_SIZE, MAX_BATCH_BYTES) != 0) {
        fprintf(stderr, "failed to initialize GPU backend\n");
        cleanup(&xsks, obj, ifname);
        return 1;
    }

    printf("AF_XDP GPU-batched running on if=%s queue=%d threshold_pps=%d rx_burst=%d classify_batch=%d\n",
           ifname, queue_id, DDOS_PPS_THRESHOLD, RX_BURST_SIZE, CLASSIFY_BATCH_SIZE);

    rx_loop(&xsks);
    cleanup(&xsks, obj, ifname);
    return 0;
}