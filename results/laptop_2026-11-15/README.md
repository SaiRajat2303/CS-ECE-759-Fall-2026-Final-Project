# Laptop runs, 2026-11-15 (provisional)

Full LBVH builds on a GeForce RTX 3050 Laptop GPU (sm_86, 1 MB L2, CUDA 12.4), pageable memory, block 256. These are not Euler numbers: the laptop's cache is 40 times smaller than the RTX 4000 Ada's, so nothing here tests the 2^19 crossover.

| File | Contents |
|---|---|
| `points_5dists_2e16-2e22.csv` | Five random-point distributions, n = 2^12 and 2^16 to 2^22, AoS and SoA, 5 repeats |
| `meshes_bunny_armadillo.csv` | Stanford bunny and armadillo, AoS and SoA, 5 repeats |
| `meshes_dragon_happy_asian.csv` | Stanford dragon, happy Buddha and Asian Dragon, AoS and SoA, 5 repeats |
| `euler_job_rehearsal_aos.csv` | Rehearsal of `slurm/full_build.sh`, AoS only |

Every tree in these runs matched the CPU reference bit for bit.

Notes:
- Use rows with `timing_mode = chain` for per-stage times; `ends` rows only have `ms_total`.
- Per-stage times below about 2^14 are dominated by event overhead and should not be used.
- The `git_commit` values predate the branch history rewrite and do not exist on GitHub; the nearest pushed code is `4393bf5`.
- AoS and SoA ran as separate programs one after the other, not interleaved.
- `host` was replaced with `laptop-rtx3050`.
