# Full LBVH build harness (`bench/`)

Builds an LBVH on the GPU from random points or a triangle mesh and times each stage with CUDA events. It writes one CSV row per timed build and can check every tree bit for bit against the CPU reference in `src/reference.cpp`. The per-thread kernel logic comes from `include/lbvh/`.

This harness is separate from the `src/builder.cu` driver described in the top-level README. Flags such as `--hostmem` and `--launch` belong to that driver, not this one.

Stages, in order (CSV column in brackets):

1. Copy primitives host to device (`ms_h2d`)
2. Scene bounding box, `cub::DeviceReduce` (`ms_bounds`)
3. Morton codes (`ms_morton`)
4. Radix sort, `cub::DeviceRadixSort` (`ms_sort`)
5. Hierarchy (`ms_hierarchy`)
6. Bounding-box refit (`ms_refit`)

## Build and run

```bash
module load nvidia/cuda/13.0.0                     # Euler only
make                                               # both layouts, sm_89 -> bin/sm_89/lbvh_build_{aos,soa}
make local                                         # both layouts, sm_86 (laptop RTX 3050)
make build LAYOUT=soa XFLAGS="-maxrregcount=32"    # one layout with extra nvcc flags

bash scripts/fetch_meshes.sh                       # Stanford meshes -> data/meshes/
bash slurm/submit.sh slurm/full_build.sh           # Euler: build on the login node, run through sbatch
```

Never run the binaries on the Euler login node. Locally:

```bash
bin/sm_86/lbvh_build_soa --points uniform:16-22 --mesh data/meshes/dragon.ply \
    --repeats 15 --validate --out data/mine.csv
python3 analysis/summarize.py data/mine.csv        # medians and spread per configuration
```

## Build-time flags (`make` variables)

| Variable | Values | Default | What it does |
|---|---|---|---|
| `LAYOUT` | `aos`, `soa` | `make` builds both | Node storage. `aos`: one struct per node. `soa`: one array per field. Sets `-DLBVH_LAYOUT_SOA`. |
| `ARCH` | any `sm_XX` | `sm_89` | GPU target. `sm_89` is the Euler RTX 4000 Ada; `make local` uses `sm_86`. |
| `XFLAGS` | any extra `nvcc` flags | empty | Added after the fixed `-O3 -std=c++17 -lineinfo`. |
| `NVCC` | compiler path | `nvcc` | |

The binary name does not change with `XFLAGS`, so a flagged build replaces the plain one. To tell them apart, the `git_commit` CSV column records the build: `<commit>`, then `-dirty-<hash>` for uncommitted edits, then `+<flags>`, e.g. `1e0c586-dirty-48890ec+--use_fast_math`. A rerun skips rows already in the CSV, and the skip check matches on this stamp, so differently flagged builds never skip each other's rows.

### `XFLAGS` values tested

Results: [`results/laptop_2026-11-20_flags/`](../results/laptop_2026-11-20_flags/). To check registers and machine code (compile only, so it is fine on the login node), run `bash scripts/check_flags.sh [ARCH]`.

| `XFLAGS` | Machine code vs plain `-O3` | Tree matches CPU reference | Time vs plain `-O3` |
|---|---|---|---|
| (none) | baseline | yes, 8 of 8 scenes | baseline |
| `--use_fast_math` | Morton kernels shorter (triangles: 248 to 104 instructions) | **no, 6 of 8 scenes** | within noise |
| `-maxrregcount=32` | slightly different | yes, 8 of 8 | within noise |
| `-Xptxas -dlcm=cg` | identical | same code, not rerun | same code, not rerun |
| `--extra-device-vectorization` | identical | same code, not rerun | same code, not rerun |

Keep plain `-O3`.

### Registers per thread

`sm_89`, CUDA 12.4, from `nvcc -Xptxas -v`. Nothing spills in any build. A `*` marks a kernel whose machine code differs from the build with no extra flags.

| Layout | `XFLAGS` | `morton` | `morton_box` | `hierarchy` | `refit` | `refit_box` |
|---|---|---|---|---|---|---|
| AoS | (none) | 25 | 25 | 22 | 24 | 28 |
| AoS | `--use_fast_math` | 19* | 22* | 22 | 24* | 28* |
| AoS | `-maxrregcount=32` | 23* | 23* | 22* | 24* | 28* |
| SoA | (none) | 25 | 25 | 24 | 32 | 32 |
| SoA | `--use_fast_math` | 19* | 22* | 24 | 32* | 32* |
| SoA | `-maxrregcount=32` | 23* | 23* | 24* | 28* | 32* |

`-Xptxas -dlcm=cg` and `--extra-device-vectorization` give the same counts as no flags, with identical machine code.

On this GPU, a kernel keeps the most threads running as long as it uses 40 registers or fewer. Every kernel is already at 32 or below, so a register limit has nothing to gain. Only one register limit was ever built: `-maxrregcount=32`. No other `-maxrregcount` value was tried, and neither was `__launch_bounds__`; the counts above ruled them out. `LBVH_REG_CAP` and `LBVH_BLOCK_SIZE` in `include/lbvh/common.h` belong to the `src/` driver and have no effect in this harness. Use `XFLAGS=-maxrregcount=N` and `--block` instead.

### Compile-time macros

`LBVH_STREAM_HINT_CG` (`0` or `1`, in `include/lbvh/common.h`) can be set with `XFLAGS=-DLBVH_STREAM_HINT_CG=1`. With `1`, the loads of point positions in Morton and refit, and of the sorted indices in refit, skip L1 and read from L2 only. Triangle-box loads are not affected. It has not been tested.

## Run-time flags

| Flag | Values | Default | What it does |
|---|---|---|---|
| `--points DIST:K[-K2]` | `DIST` = `uniform`, `gaussian`, `clusters`, `surface`, `duplicates`; `K` from 1 to 27 | none | Random points, n = 2^K (or every K from K to K2). Repeatable. |
| `--mesh FILE.ply` | any triangle PLY | none | One box per triangle. Repeatable. |
| `--block N` | multiple of 32, from 32 to 1024 | `256` | Threads per block for Morton, hierarchy and refit. CUB picks its own. |
| `--repeats N` | 1 or more | `10` | Timed builds per scene and timing mode. |
| `--warmup-ms MS` | milliseconds | `300` | Warm up for a fixed wall time first, since the idle GPU clock is low. |
| `--timing-mode` | `chain`, `ends`, `both` | `both` | `chain` records an event between every stage, giving per-stage times. `ends` records only the start and end, giving `ms_total` only. Comparing the two shows whether the extra events add overhead. |
| `--seed N` | integer | `1` | Point seed. Also seeds the shuffled run order. |
| `--validate` | flag | off | After timing, compares the whole tree (keys, links, boxes) with the CPU reference. |
| `--out PATH` | file path | `data/full_<host>_<job>_<task>.csv` | Appends. Rows already present are skipped, so a requeued job resumes. |
| `--device N` | integer | `0` | CUDA device. |

Scenes run in a shuffled order, and `chain` and `ends` builds are interleaved, so clock drift does not favour one of them.

Host memory is always pageable (`mem_mode = pageable` in every row). `bench/config.h` lists pinned and managed modes, but the harness has no flag for them yet.

## Datasets

### Stanford 3D Scanning Repository meshes (main dataset)

`bash scripts/fetch_meshes.sh` downloads them into `data/meshes/` and checks each against a SHA-256 checksum. They are not committed: the Asian Dragon is over GitHub's 100 MB file limit.

| `scene` | Source file | Triangles | vs 2^19 = 524,288 |
|---|---|---|---|
| `bunny` | `bun_zipper.ply` | 69,451 | below |
| `armadillo` | `Armadillo.ply` | 345,944 | below |
| `dragon` | `dragon_vrip.ply` | 871,414 | above |
| `happy` | `happy_vrip.ply` (Happy Buddha) | 1,087,716 | above |
| `asian_dragon` | `xyzrgb_dragon.ply` | 7,219,045 | above |

We predict that the node array outgrows the 40 MB L2 on the RTX 4000 Ada at about 2^19 primitives. The meshes sit on both sides of that size. Each triangle becomes one leaf box, and its Morton code comes from the box centroid.

### Random points (correctness and size sweeps)

Points are generated in memory by `include/lbvh/pointgen.h`. In the CSV, the `scene` is `<dist>_s<seed>`.

| `DIST` | Shape |
|---|---|
| `uniform` | unit cube |
| `gaussian` | one blob |
| `clusters` | 16 tight blobs |
| `surface` | points on a sphere, like a scanned object |
| `duplicates` | n/16 distinct positions, each repeated about 16 times |

## Results so far

All of these results come from one GPU: a GeForce RTX 3050 Laptop GPU (sm_86, 1 MB L2, CUDA 12.4). Every timed row uses pageable memory and block 256. **There are no Euler numbers yet.** The laptop's L2 is 40 times smaller, so none of these runs tests the 2^19 prediction. Each folder's README lists its own caveats.

| File | Configurations | Datasets | Repeats |
|---|---|---|---|
| [`laptop_2026-11-15/points_5dists_2e16-2e22.csv`](../results/laptop_2026-11-15/points_5dists_2e16-2e22.csv) | `aos`, `soa` | all 5 point distributions, 2^16 to 2^22 | 5 |
| [`laptop_2026-11-15/meshes_bunny_armadillo.csv`](../results/laptop_2026-11-15/meshes_bunny_armadillo.csv) | `aos`, `soa` | bunny, armadillo, uniform 2^12 | 5 |
| [`laptop_2026-11-15/meshes_dragon_happy_asian.csv`](../results/laptop_2026-11-15/meshes_dragon_happy_asian.csv) | `aos`, `soa` | dragon, happy, asian_dragon | 5 |
| [`laptop_2026-11-15/euler_job_rehearsal_aos.csv`](../results/laptop_2026-11-15/euler_job_rehearsal_aos.csv) | `aos` only, a local run of `slurm/full_build.sh` | uniform, clusters, duplicates at 2^16 to 2^23; all 5 meshes | 15 |
| [`laptop_2026-11-20_flags/timing.csv`](../results/laptop_2026-11-20_flags/timing.csv) | `aos`, `soa` x `XFLAGS` = none, `--use_fast_math`, `-maxrregcount=32`; `chain` only | armadillo, dragon, happy, asian_dragon, uniform 2^22 (a new seed each round) | 15 per round, 4 rounds |
| [`laptop_2026-11-20_flags/validation.txt`](../results/laptop_2026-11-20_flags/validation.txt) | same 6 builds, correctness only | bunny, armadillo, dragon, happy; uniform and duplicates 2^16, clusters 2^18, surface 2^20 | - |

What they show:

- Every tree built with plain `-O3` matched the CPU reference bit for bit, in both layouts.
- **Hierarchy:** SoA is 2.0 to 2.7 times faster than AoS at 2^16 and above.
- **Refit:** SoA is faster, but the size of the gap is not settled. On armadillo, dragon, happy and asian_dragon, AoS refit took 1.07 to 1.11 times as long as SoA on 11-15, and 1.14 to 1.25 times as long on 11-20. The laptop does not log its clock, so we cannot yet say why the days differ.
- **Copy:** copying the triangles to the GPU is about half of a mesh build on the laptop.
- **Flags:** no compiler flag helps. See the table above.

## Not tested yet

- Anything on Euler (sm_89, CUDA 13.0). `slurm/full_build.sh` is ready: both layouts, block 256, uniform/clusters/duplicates at 2^16 to 2^23 plus all five meshes, 15 repeats, `--validate`. It logs the GPU clock every second to `data/telem_<job>_<task>.csv`.
- Timings at any block size other than 256. Block sizes 32 to 1024 passed the CPU-reference check on 2026-11-15, but that output was not kept.
- Any register limit other than `-maxrregcount=32`, and `__launch_bounds__`.
- Pinned and unified host memory.
- `LBVH_STREAM_HINT_CG=1`.
