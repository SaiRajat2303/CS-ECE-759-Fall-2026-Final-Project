# Dynamic BVH benchmark on the Stanford Dragon

`bvh_cuda_bench` runs the LBVH from `include/lbvh/` on an animated triangle mesh.
The adapter is `lbvh_user_bvh.cuh` and the triangle kernels are in `include/lbvh/tri_impl.h`.
Each frame it keeps two BVHs:

- **rebuild**: a full build from scratch every frame.
- **update**: built once at frame 0, then only refit with the frame-0 topology.

Both trace the same camera and random rays. The tool checks that they find the same hits
and writes per-frame timings, SAH cost and nodes visited per ray to a CSV file.

`make_frames` (with `dynamic_scene.h`) turns the static scan into frames with constant topology:

| Mode | Motion |
| --- | --- |
| `rigid` | half a turn about the vertical axis plus a sideways drift |
| `twist` | rotation whose angle grows with height |
| `wave` | a travelling plane wave displaces vertices vertically |
| `explode` | every triangle flies outward and spins (worst case for refitting) |

## On Euler (Slurm): one Nsight Systems report per transformation

From the repository root:

```bash
# once, on the login node (needs internet; CMake never downloads anything)
test/3D_dragon/prefetch.sh            # tinyobjloader v2.0.0rc13 + happly v1.1.0 -> test/3D_dragon/_deps/
mkdir -p logs

sbatch test/run/bvh_bench.sh          # all modes: rigid twist wave explode
sbatch --export=ALL,MODES="explode",NUM_FRAMES=20 test/run/bvh_bench.sh
```

The job extracts `assets/graphics/dragon_recon.tar.gz` if needed, builds, and then profiles each mode.
Results go to `test/3D_dragon/results/<job id>/`:

- `bvh_<mode>.nsys-rep`: timeline with NVTX ranges for each frame, strategy (rebuild/update) and phase (build, refit, trace)
- `bvh_<mode>.csv`: the per-frame numbers
- `<mode>.log`, `build.log`, `run_info.txt` (GPU, nsys and nvcc versions, commit)

Each report also names its mode inside the file, so renaming it is safe:

- On the timeline, the top NVTX range spans the whole run, for example `explode | 2614242 verts, 871414 tris, 10 frames`.
- Each frame range is named like `explode frame 3`.
- Analysis Summary → command line shows `--label explode`.
- From a shell: `nsys stats --report nvtx_sum bvh_explode.nsys-rep`.
- In the CSV, the first column (`label`) is the mode, so CSVs from different modes can be concatenated.

Other settings (environment variables): `NUM_FRAMES` (10), `AMOUNT` (1.0), `CAM_RES` (1024),
`RND_RAYS` (1048576), `REPEATS` (5), `CUDA_ARCH` (native) and `MESH`,
plus `NCU`, `NCU_FRAMES` and `GPU_METRICS` (see below).

## Occupancy and memory alignment

An Nsight Systems timeline only shows *theoretical* occupancy. Achieved occupancy and
memory alignment come from Nsight Compute. Both are opt-in:

```bash
sbatch --export=ALL,NCU=1,GPU_METRICS=1 test/run/bvh_bench.sh
```

- `GPU_METRICS=1` samples the GPU while the normal run executes, and adds GPU Metrics rows
  to `bvh_<mode>.nsys-rep`: SM active, warp occupancy and DRAM bandwidth over time.
- `NCU=1` also writes `bvh_<mode>_ncu.ncu-rep`, made by replaying each build/refit/trace
  kernel (`traceKernel`, `refitTriKernel`, `triBoundsKernel`, `mortonKernel`, `hierarchyKernel`).
  It covers the first `NCU_FRAMES` (2) frames: frame 0 builds both trees, and frame 1 has a
  rebuild, a refit and traces with both trees. This step takes minutes per mode.

Both need access to the GPU performance counters. If a log shows `ERR_NVGPUCTRPERM`, the
node blocks counters for normal users, and only the cluster admins can change that.

Where to look:

| Question | Tool and place |
| --- | --- |
| Launch config, registers/thread, theoretical occupancy | nsys: click (or hover over) a kernel in the CUDA HW row. ncu: Launch Statistics |
| Achieved vs theoretical occupancy, and what limits it (registers, block size, shared memory) | ncu: Occupancy section |
| Occupancy over time across the frame | nsys with `GPU_METRICS=1`: GPU Metrics rows, under the NVTX ranges |
| Alignment / coalescing of global loads and stores | ncu: Memory Workload Analysis tables, *sectors per request* (4 is ideal for 32-bit loads; more means misaligned or scattered accesses) |
| Which source lines make uncoalesced accesses | ncu: Source page, or the Source Counters section ("uncoalesced global accesses"). Uses `-lineinfo`, which CMake already sets |
| Why warps wait (memory, dependencies, ...) | ncu: Warp State Statistics |

From a shell, without the GUI:

```bash
ncu --import bvh_explode_ncu.ncu-rep --page details --section Occupancy
ncu --import bvh_explode_ncu.ncu-rep --page details --section MemoryWorkloadAnalysis_Tables
```

Registers and stack size per kernel need no GPU: `make ptxinfo` (point-cloud kernels), or add
`-Xptxas -v` to the nvcc flags.

## By hand

```bash
cd test/3D_dragon
./prefetch.sh
tar xzf ../../assets/graphics/dragon_recon.tar.gz -C ../../assets/graphics
# dragon_vrip.ply has ~871K triangles; dragon_vrip_res2/3/4.ply are smaller versions for quick tests

cmake -B build -DCMAKE_BUILD_TYPE=Release        # needs CMake >= 3.24
cmake --build build -j

./build/make_frames ../../assets/graphics/dragon_recon/dragon_vrip.ply explode 60 frames/explode
./build/bvh_cuda_bench "frames/explode/frame_%04d.ply" 60 explode.csv
```

## Viewing the reports

Install Nsight Systems (for `.nsys-rep`) and Nsight Compute (for `.ncu-rep`) on your
workstation, using the [same or a newer version](https://developer.nvidia.com/nsight-systems/get-started)
than the ones in `run_info.txt`. Copy the reports over with scp or rsync and open them there.
