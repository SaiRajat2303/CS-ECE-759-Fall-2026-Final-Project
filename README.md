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
## Layout

```
include/lbvh/common.h        types, knobs, Morton code, cache-policy loads
include/lbvh/nodes.h         AoS and SoA node storage behind one accessor API
include/lbvh/kernels_impl.h  per-thread logic of every kernel (GPU and CPU)
include/lbvh/kernels.h       kernel declarations
include/lbvh/builder.h       Builder: buffers, memory modes, launches, graph
include/lbvh/tree.h          host tree, CPU reference, comparison, features
include/lbvh/export.h        device layout -> host tree
include/lbvh/pointgen.h      synthetic point clouds, .xyz loader
src/kernels.cu               __global__ wrappers (register cap applies here only)
src/builder.cu               pipeline, CUB calls, CUDA Graph, timing
src/reference.cpp            independent top-down CPU builder and checks
src/main.cu                  benchmark driver, CSV output
tests/test_cpu.cpp           runs the kernel logic on the CPU against the reference
scripts/setup_cuda_toolchain.sh  installs nvcc and the CUDA runtime without root
scripts/build_all.sh         builds every compile-time variant
scripts/sweep.sbatch         Slurm job array for the full sweep
```

## Status

- Kernel logic: tested on CPU against the independent reference (298 cases per
  layout, sizes 2 to 200,000, five distributions, degenerate inputs, refit on
  real threads), also under AddressSanitizer, UBSan and ThreadSanitizer.
- CUDA code: compiles and links for all 16 variants with CUDA 13.0 and 13.5
  (`sm_89`). The generated PTX has the expected cache operators.

## Credits

Algorithm: T. Karras, "Maximizing Parallelism in the Construction of BVHs,
Octrees, and k-d Trees", HPG 2012, and "Thinking Parallel, Part III: Tree
Construction on the GPU", NVIDIA Developer Blog, 2012. This is an independent
implementation written from those two sources. ToruNiina/lbvh (MIT license) is
a second implementation of the same algorithm that is useful for comparison.