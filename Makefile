# LBVH build benchmark -- one binary per combination of compile-time knobs.
#
#   make                               default variant
#   make BLOCK=512 LAYOUT=soa HINT=cg REGCAP=32
#   make test                          CPU-only tests (no GPU or nvcc needed)
#   make ptxinfo                       registers used by each kernel
#   scripts/build_all.sh               all variants
#
# Knobs:
#   BLOCK   threads per block for our kernels: 128 | 256 | 512 | 1024
#   LAYOUT  node layout: aos | soa
#   HINT    cache policy of read-once loads: default | cg
#   REGCAP  register cap for our kernels: 0 (compiler default) or a number,
#           passed as -maxrregcount to kernels.cu only

NVCC ?= nvcc
CXX ?= g++
ARCH ?= sm_89
BLOCK ?= 256
LAYOUT ?= aos
HINT ?= default
REGCAP ?= 0
EXTRA_NVCCFLAGS ?=

SOA := $(if $(filter soa,$(LAYOUT)),1,0)
CG := $(if $(filter cg,$(HINT)),1,0)
TAG := b$(BLOCK)_$(LAYOUT)_$(HINT)_r$(REGCAP)
OBJ := build/$(TAG)
BIN := bin/lbvh_$(TAG)

DEFS := -DLBVH_BLOCK_SIZE=$(BLOCK) -DLBVH_LAYOUT_SOA=$(SOA) -DLBVH_STREAM_HINT_CG=$(CG) -DLBVH_REG_CAP=$(REGCAP)
NVCCFLAGS := -O3 -std=c++17 -arch=$(ARCH) -Iinclude $(DEFS) $(EXTRA_NVCCFLAGS)
# -fmad=false: no fused multiply-add, so device arithmetic matches the CPU
# reference bit for bit. The build kernels contain no multiply-add anyway.
KERNELFLAGS := -fmad=false $(if $(filter-out 0,$(REGCAP)),-maxrregcount=$(REGCAP))
HEADERS := $(wildcard include/lbvh/*.h)

all: $(BIN)

$(OBJ) bin:
	mkdir -p $@

$(OBJ)/kernels.o: src/kernels.cu $(HEADERS) | $(OBJ)
	$(NVCC) $(NVCCFLAGS) $(KERNELFLAGS) -c $< -o $@

$(OBJ)/builder.o: src/builder.cu $(HEADERS) | $(OBJ)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OBJ)/main.o: src/main.cu $(HEADERS) | $(OBJ)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OBJ)/reference.o: src/reference.cpp $(HEADERS) | $(OBJ)
	$(CXX) -O2 -std=c++17 -Iinclude $(DEFS) -c $< -o $@

$(BIN): $(OBJ)/kernels.o $(OBJ)/builder.o $(OBJ)/main.o $(OBJ)/reference.o | bin
	$(NVCC) $(NVCCFLAGS) $^ -o $@

# Prints "Used N registers ..." for every kernel in kernels.cu.
ptxinfo: | $(OBJ)
	$(NVCC) $(NVCCFLAGS) $(KERNELFLAGS) -Xptxas -v -c src/kernels.cu -o $(OBJ)/kernels_ptxinfo.o

# CPU-only tests of the kernel logic, for both node layouts.
test:
	mkdir -p build/test
	$(CXX) -O2 -std=c++17 -Wall -Wextra -pthread -Iinclude -DLBVH_LAYOUT_SOA=0 tests/test_cpu.cpp src/reference.cpp -o build/test/test_aos
	$(CXX) -O2 -std=c++17 -Wall -Wextra -pthread -Iinclude -DLBVH_LAYOUT_SOA=1 tests/test_cpu.cpp src/reference.cpp -o build/test/test_soa
	build/test/test_aos
	build/test/test_soa

clean:
	rm -rf build bin

.PHONY: all test ptxinfo clean
