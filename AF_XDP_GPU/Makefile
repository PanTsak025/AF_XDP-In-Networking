BPF_CLANG ?= clang
BPF_CFLAGS := -O2 -g -target bpf

CC ?= gcc
CFLAGS := -O2 -g -Wall -Wextra

NVCC ?= nvcc
NVCCFLAGS := -O2 -Xcompiler -fPIC

CUDA_HOME ?= /usr/local/cuda
CUDA_INC := -I$(CUDA_HOME)/include
CUDA_LIB := -L$(CUDA_HOME)/lib64

LIBS := -lbpf -lxdp -lelf -lz -lcudart

TARGET := af_xdp_gpu
BPF_OBJ := xdp_redirect_kern.o
CPU_OBJ := af_xdp_gpu.o
GPU_OBJ := gpu_kernels.o

all: $(TARGET) $(BPF_OBJ)

$(BPF_OBJ): xdp_redirect_kern.c
	$(BPF_CLANG) $(BPF_CFLAGS) -c $< -o $@

$(CPU_OBJ): af_xdp_gpu.c gpu_kernels.h
	$(CC) $(CFLAGS) $(CUDA_INC) -c $< -o $@

$(GPU_OBJ): gpu_kernels.cu gpu_kernels.h
	$(NVCC) $(NVCCFLAGS) $(CUDA_INC) -c $< -o $@

$(TARGET): $(CPU_OBJ) $(GPU_OBJ)
	$(CC) $(CFLAGS) -o $@ $(CPU_OBJ) $(GPU_OBJ) $(CUDA_LIB) $(LIBS)

clean:
	rm -f $(TARGET) $(BPF_OBJ) $(CPU_OBJ) $(GPU_OBJ)

run: all
	sudo ./$(TARGET) $(IFACE) $(QUEUE)

.PHONY: all clean run