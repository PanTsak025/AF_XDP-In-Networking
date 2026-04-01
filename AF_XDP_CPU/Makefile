CLANG ?= clang
CC ?= gcc

BPF_CFLAGS := -O2 -g -target bpf
USER_CFLAGS := -O2 -g
USER_LIBS := -lxdp -lbpf -lelf -lz

all: xdp_sock_kern.o af_xdp_rx

xdp_sock_kern.o: xdp_sock_kern.c
	$(CLANG) $(BPF_CFLAGS) -c $< -o $@

af_xdp_rx: af_xdp_rx.c
	$(CC) $(USER_CFLAGS) $< -o $@ $(USER_LIBS)

clean:
	rm -f xdp_sock_kern.o af_xdp_rx