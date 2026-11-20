# Compiler flags, 2026-11-20 (laptop timings)

Does any nvcc flag on top of `-O3` speed up our kernels? Code is `1c865fe`, unchanged. Registers and machine code were checked for `sm_89` with `scripts/check_flags.sh` (CUDA 12.4). Correctness and timing ran on a GeForce RTX 3050 Laptop GPU (sm_86, CUDA 12.4), pageable memory, block 256.

| Flag | Kernel machine code vs `-O3` | Tree matches CPU reference | Time vs `-O3` |
|---|---|---|---|
| `-Xptxas -dlcm=cg` | identical | same code, not rerun | same code, not rerun |
| `--extra-device-vectorization` | identical | same code, not rerun | same code, not rerun |
| `-maxrregcount=32` | slightly different | yes, 8 of 8 scenes | within noise |
| `--use_fast_math` | Morton kernels shorter (triangles: 248 to 104 instructions) | **no, 6 of 8 scenes** | within noise |

- Every kernel uses 22 to 32 registers with no spills. On sm_89 a thread can use up to 40 before occupancy drops, so a register cap or `__launch_bounds__` has nothing to gain.
- `--use_fast_math` replaces the exact divide in `quantize` with an approximate one. Some Morton codes move by one cell, so the tree is still valid but no longer bit-identical to the CPU reference. It is not faster either: the Morton stage is limited by memory, not arithmetic.
- Every per-stage and total difference is inside the `-O3` build's own round-to-round spread (1 to 4.5% of the total). On the dragon, the host-to-device copy is about half the build and refit about 30%; flags only touch the smaller kernel stages.

Conclusion: keep plain `-O3`, and leave register limits out of the sweep.

| File | Contents |
|---|---|
| `timing.csv` | 4 rounds x 3 builds x 2 layouts x 5 scenes (armadillo, dragon, happy, asian_dragon, 2^22 uniform points) x 15 repeats, `chain` mode |
| `validation.txt` | `--validate` result per build and scene (bunny, armadillo, dragon, happy and four point sets) |

Notes:
- `git_commit` is `1c865fe` with the build name added by hand (`-base`, `-fastmath`, `-rreg32`). Builds from `make XFLAGS=...` now carry their flags in this column automatically.
- The builds ran as separate programs in a new shuffled order each round, which spreads clock drift across them but does not remove it.
- The point scene uses a different seed each round (`uniform_s1` to `uniform_s4`).
- `host` was replaced with `laptop-rtx3050`.
- Not yet checked with CUDA 13.0 on Euler. `bash scripts/check_flags.sh` only compiles, so it can run on the login node.
