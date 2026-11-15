NVCC   ?= nvcc
ARCH   ?= sm_89
LAYOUT ?= aos

ifeq ($(LAYOUT),aos)
  LAYOUT_DEF := -DLBVH_LAYOUT_SOA=0
else ifeq ($(LAYOUT),soa)
  LAYOUT_DEF := -DLBVH_LAYOUT_SOA=1
else
  $(error LAYOUT must be aos or soa)
endif

GIT_COMMIT := $(shell git rev-parse --short HEAD 2>/dev/null || echo nogit)
GIT_DIRTY  := $(shell git diff --quiet HEAD 2>/dev/null || echo -dirty)
VERSION    := $(GIT_COMMIT)$(GIT_DIRTY)

NVCCFLAGS := -arch=$(ARCH) -O3 -std=c++17 -lineinfo -Iinclude -Ibench $(LAYOUT_DEF) \
             -DGIT_COMMIT=\"$(VERSION)\" -Xcompiler -Wall,-Wextra

BIN   := bin/$(ARCH)/lbvh_build_$(LAYOUT)
STAMP := bin/$(ARCH)/.version_$(LAYOUT)
SRCS  := bench/run_build.cu src/reference.cpp
DEPS  := $(wildcard include/lbvh/*.h bench/*.h bench/*.cuh)

SMOKE_ARGS := --points uniform:12 --points duplicates:12 --points clusters:16 \
              $(if $(wildcard data/meshes/bunny.ply),--mesh data/meshes/bunny.ply) \
              --repeats 3 --warmup-ms 50 --validate

.PHONY: all build both local smoke clean FORCE

all: both

both:
	$(MAKE) build LAYOUT=aos
	$(MAKE) build LAYOUT=soa

build: $(BIN)

local:
	$(MAKE) both ARCH=sm_86

# Rebuild whenever the commit changes, so the commit stamped into every CSV row is never stale.
$(STAMP): FORCE
	@mkdir -p $(dir $@)
	@echo '$(VERSION)' | cmp -s - $@ || echo '$(VERSION)' > $@

$(BIN): $(SRCS) $(DEPS) $(STAMP)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(SRCS) -o $@

smoke: both
	./bin/$(ARCH)/lbvh_build_aos $(SMOKE_ARGS) --out data/smoke_$(ARCH).csv
	./bin/$(ARCH)/lbvh_build_soa $(SMOKE_ARGS) --out data/smoke_$(ARCH).csv

clean:
	rm -rf bin
