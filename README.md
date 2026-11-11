# LBVH build and benchmarking (CS/ECE 759 final project)
A configurable GPU build of a Linear Bounding Volume Hierarchy over a point
cloud, following Karras (HPG 2012). Every configuration produces the same tree,
which is checked against an independent CPU builder.
The driver times each configuration and writes a CSV for the learned cost model.

## Pipeline

1. Scene bounding box: `cub::DeviceReduce`
2. Morton codes (30-bit): `mortonKernel`
3. Radix sort of (code, index) pairs: `cub::DeviceRadixSort`
4. Hierarchy generation: `hierarchyKernel`, one thread per internal node
5. Bounding-box refit: `refitKernel`, one thread per leaf, bottom-up with atomic counters

## Knobs

| Knob | Values | How to select |
| --- | --- | --- |
| Block size | 128, 256, 512, 1024 | `make BLOCK=512` |
| Node layout | `aos`, `soa` | `make LAYOUT=soa` |
| Host memory | `pageable`, `pinned`, `unified`, `unified_prefetch` | `--hostmem` at run time |
| Launch mode | `separate`, `graph` | `--launch` at run time |

## Build

Needs `nvcc` (CUDA 13.x; 12.x should also work) and g++.

If the machine has no CUDA compiler install one into this directory without root access:


```bash
module load conda/miniforge gcc/13.2.0   # Python with pip, and a recent g++
bash scripts/setup_cuda_toolchain.sh     # once, on a node with internet access
source cuda-toolchain/env.sh             # in every new shell and in job scripts
```

Then:

```bash
make test                     # CPU-only tests of the kernel logic, no GPU needed
make                          # default variant -> bin/lbvh_b256_aos_default_r0
bash scripts/build_all.sh     # all 16 compile-time variants (JOBS=8 for parallel)
```

Set `ARCH` for a GPU other than the RTX 4000 Ada (`make ARCH=sm_80`).

## Run

```bash
bin/lbvh_b256_aos_default_r0 --info
bin/lbvh_b256_aos_default_r0 --n 1000000 --dist clusters --seed 1 \
    --reps 5 --warmup 2 --hostmem all --launch all --verify --csv results.csv
```
