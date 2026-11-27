#!/usr/bin/env bash
# =============================================================================
# bvh_bench.sh — profile the LBVH (include/lbvh/, via test/3D_dragon's UserBVH)
# on the animated Stanford Dragon, one Nsight Systems report per transformation.
#
# For each MODE in MODES (default: rigid twist wave explode):
#   1. make_frames turns the dragon into NUM_FRAMES animated frames,
#   2. bvh_cuda_bench runs under `nsys profile`: per frame it rebuilds one BVH
#      from scratch and refits the other, traces camera + random rays with both,
#      and checks that both give the same hits.
# Output, in results/<job id>/ next to this script's sources:
#   bvh_<mode>.nsys-rep   timeline with NVTX ranges per frame / strategy / phase
#   bvh_<mode>.csv        per-frame timings, SAH cost, visits/ray, mismatches
#   <mode>.log, build.log, run_info.txt
# Optional (both off by default; both need GPU performance-counter access):
#   NCU=1          also run Nsight Compute per mode on the first NCU_FRAMES (2)
#                  frames -> bvh_<mode>_ncu.ncu-rep: achieved occupancy and its
#                  limiter, memory sectors/request and uncoalesced accesses per
#                  source line, warp stall reasons. Takes minutes per mode.
#   GPU_METRICS=1  sample SM activity / warp occupancy / DRAM bandwidth over time
#                  into the .nsys-rep (extra "GPU Metrics" rows on the timeline)
#
# Before the first submission, on the LOGIN node (needs internet):
#   test/3D_dragon/prefetch.sh
#   mkdir -p logs
# Submit from the repository root:
#   sbatch test/run/bvh_bench.sh
#   sbatch --export=ALL,MODES="explode",NUM_FRAMES=20 test/run/bvh_bench.sh
#   sbatch --export=ALL,NCU=1,GPU_METRICS=1 test/run/bvh_bench.sh
# Copy the reports to your workstation (scp/rsync) and open them in the same or
# a newer version of Nsight Systems than the one recorded in run_info.txt.
# =============================================================================

# ----------------------------- Slurm resources (EDIT ME) --------------------
#SBATCH --job-name=bvh-profile
#SBATCH --partition=instruction
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1                 # e.g. gpu:a100:1 to request a GPU type
#SBATCH --cpus-per-task=8            # used to compile in parallel
#SBATCH --mem=32G
#SBATCH --time=01:30:00              # ~45 min is enough without NCU=1
#SBATCH --output=logs/%x-%j.out      # the logs/ dir must exist before sbatch
#SBATCH --error=logs/%x-%j.out

set -euo pipefail

# ----------------------------- Modules (EDIT ME) ----------------------------
module load cuda          # provides nvcc, nsys (Nsight Systems) and ncu (Nsight Compute)
module load cmake         # needs >= 3.24
# module load gcc 2>/dev/null || true

# ----------------------------- Settings -------------------------------------
# Slurm runs a copy of this script, so locate the repository from the submit
# directory rather than from $0.
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
REPO_ROOT="${REPO_ROOT:-$(git -C "$SUBMIT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SUBMIT_DIR")}"
SRC_DIR="$REPO_ROOT/test/3D_dragon"                          # folder with CMakeLists.txt
ASSET_DIR="$REPO_ROOT/assets/graphics"
MESH="${MESH:-$ASSET_DIR/dragon_recon/dragon_vrip.ply}"       # ~871K triangles
MODES="${MODES:-rigid twist wave explode}"
NUM_FRAMES="${NUM_FRAMES:-10}"                                # frames per mode
AMOUNT="${AMOUNT:-1.0}"                                       # transformation strength
CAM_RES="${CAM_RES:-1024}"
RND_RAYS="${RND_RAYS:-1048576}"
REPEATS="${REPEATS:-5}"
CUDA_ARCH="${CUDA_ARCH:-native}"                              # or e.g. 80 (A100), 90 (H100)
NCU="${NCU:-0}"                                               # 1: add an Nsight Compute report per mode
NCU_FRAMES="${NCU_FRAMES:-2}"                                 # frames replayed under ncu (slow)
GPU_METRICS="${GPU_METRICS:-0}"                               # 1: GPU metrics sampling in nsys
if (( NCU_FRAMES > NUM_FRAMES )); then NCU_FRAMES="$NUM_FRAMES"; fi

# Kernels profiled by ncu: the LBVH build/refit/trace kernels. Left out: the
# instrumented traceVisitsKernel (untimed) and CUB's reduce/sort kernels.
NCU_KERNELS='regex:traceKernel|refitTriKernel|triBoundsKernel|mortonKernel|hierarchyKernel'

JOB_ID="${SLURM_JOB_ID:-local}"
RESULTS="$SRC_DIR/results/$JOB_ID"
BUILD_DIR="$SRC_DIR/build-gpu"
SCRATCH="${TMPDIR:-/tmp}/bvh-$JOB_ID"                         # node-local scratch for frames
mkdir -p "$RESULTS" "$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# Explains the most common profiler failure on shared clusters.
counterHint() {
    if grep -q "ERR_NVGPUCTRPERM" "$1" 2>/dev/null; then
        echo "  -> the driver blocks GPU performance counters for non-admin users on this node."
        echo "     ncu and GPU_METRICS need them; ask the cluster admins, or rerun without NCU/GPU_METRICS."
    fi
}

NSYS_EXTRA=()
if [[ "$GPU_METRICS" == 1 ]]; then NSYS_EXTRA+=(--gpu-metrics-devices=all); fi

# ----------------------------- Checks ---------------------------------------
command -v nsys > /dev/null || { echo "ERROR: nsys not found (is the cuda module loaded?)"; exit 1; }
if [[ "$NCU" == 1 ]]; then
    command -v ncu > /dev/null || { echo "ERROR: NCU=1 but ncu not found (is the cuda module loaded?)"; exit 1; }
fi
[[ -f "$SRC_DIR/CMakeLists.txt" ]] || { echo "ERROR: $SRC_DIR/CMakeLists.txt not found; submit from the repository root"; exit 1; }
[[ -d "$SRC_DIR/_deps/tinyobjloader" && -d "$SRC_DIR/_deps/happly" ]] \
    || { echo "ERROR: $SRC_DIR/_deps is missing; run test/3D_dragon/prefetch.sh manually on the login node first"; exit 1; }
for m in $MODES; do
    case "$m" in rigid|twist|wave|explode) ;; *) echo "ERROR: unknown mode '$m' (rigid|twist|wave|explode)"; exit 1 ;; esac
done
if [[ ! -f "$MESH" && -f "$ASSET_DIR/dragon_recon.tar.gz" ]]; then
    log "Extracting $ASSET_DIR/dragon_recon.tar.gz"
    tar xzf "$ASSET_DIR/dragon_recon.tar.gz" -C "$ASSET_DIR"
fi
[[ -f "$MESH" ]] || { echo "ERROR: mesh not found: $MESH"; exit 1; }

# Record what produced the reports. Your local viewer must be the SAME OR NEWER
# Nsight Systems version than the one printed here.
{
    echo "job:     $JOB_ID on $(hostname), $(date)"
    echo "commit:  $(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "mesh:    $MESH"
    echo "run:     modes=[$MODES] frames=$NUM_FRAMES amount=$AMOUNT cam=${CAM_RES}x${CAM_RES} rnd=$RND_RAYS repeats=$REPEATS"
    echo "extras:  ncu=$NCU (frames=$NCU_FRAMES) gpu_metrics=$GPU_METRICS"
    echo "nsys:    $(nsys --version)"
    if [[ "$NCU" == 1 ]]; then echo "ncu:     $(ncu --version | tail -1)"; fi
    echo "nvcc:    $(nvcc --version | tail -1)"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | sed 's/^/gpu:     /'
} | tee "$RESULTS/run_info.txt"

# ----------------------------- Build ----------------------------------------
log "Building in $BUILD_DIR"
cmake -S "$SRC_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" > "$RESULTS/build.log" 2>&1 \
    || { echo "ERROR: cmake configure failed, see $RESULTS/build.log"; tail -20 "$RESULTS/build.log"; exit 1; }
cmake --build "$BUILD_DIR" -j "${SLURM_CPUS_PER_TASK:-8}" >> "$RESULTS/build.log" 2>&1 \
    || { echo "ERROR: build failed, see $RESULTS/build.log"; tail -20 "$RESULTS/build.log"; exit 1; }
[[ -x "$BUILD_DIR/bvh_cuda_bench" ]] || { echo "ERROR: bvh_cuda_bench not built (CUDA not found?)"; exit 1; }

# ----------------------------- One profiled run per transformation -----------
failed=()
for MODE in $MODES; do
    FRAMES="$SCRATCH/frames/$MODE"
    REPORT="$RESULTS/bvh_$MODE"                                # nsys adds .nsys-rep

    log "[$MODE] generating $NUM_FRAMES frames"
    if ! "$BUILD_DIR/make_frames" "$MESH" "$MODE" "$NUM_FRAMES" "$FRAMES" --amount "$AMOUNT" \
            > "$RESULTS/$MODE.log" 2>&1; then
        echo "ERROR: make_frames failed for $MODE, see $RESULTS/$MODE.log"; failed+=("$MODE"); continue
    fi

    log "[$MODE] profiling with Nsight Systems"
    status=0
    nsys profile \
        --trace=cuda,nvtx \
        --sample=none --cpuctxsw=none \
        --force-overwrite=true \
        ${NSYS_EXTRA[@]+"${NSYS_EXTRA[@]}"} \
        --output="$REPORT" \
        "$BUILD_DIR/bvh_cuda_bench" "$FRAMES/frame_%04d.ply" "$NUM_FRAMES" "$RESULTS/bvh_$MODE.csv" \
            --cam-res "$CAM_RES" --rnd-rays "$RND_RAYS" --repeats "$REPEATS" --label "$MODE" \
        >> "$RESULTS/$MODE.log" 2>&1 || status=$?
    tr '\r' '\n' < "$RESULTS/$MODE.log" | grep -E '^frame|mismatch|error|ERROR' | tail -n "$((NUM_FRAMES + 2))" || true

    if (( status != 0 )); then
        echo "ERROR: bvh_cuda_bench exited with $status for $MODE, see $RESULTS/$MODE.log"; failed+=("$MODE")
        counterHint "$RESULTS/$MODE.log"
    elif [[ -f "$REPORT.nsys-rep" ]]; then
        log "[$MODE] report: $REPORT.nsys-rep ($(du -h "$REPORT.nsys-rep" | cut -f1))"
    else
        echo "ERROR: no report for $MODE, see $RESULTS/$MODE.log"; failed+=("$MODE")
    fi

    if [[ "$NCU" == 1 ]]; then
        log "[$MODE] Nsight Compute on $NCU_FRAMES frames (occupancy, memory, source counters)"
        NCU_REPORT="$RESULTS/bvh_${MODE}_ncu"                  # ncu adds .ncu-rep
        status=0
        ncu --target-processes all \
            --kernel-name "$NCU_KERNELS" \
            --section LaunchStats --section Occupancy --section SpeedOfLight \
            --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Tables \
            --section SourceCounters --section WarpStateStats --section SchedulerStats \
            --import-source yes --force-overwrite \
            -o "$NCU_REPORT" \
            "$BUILD_DIR/bvh_cuda_bench" "$FRAMES/frame_%04d.ply" "$NCU_FRAMES" "$NCU_REPORT.csv" \
                --cam-res "$CAM_RES" --rnd-rays "$RND_RAYS" --repeats 1 --label "$MODE" \
            > "$RESULTS/$MODE.ncu.log" 2>&1 || status=$?
        if (( status == 0 )) && [[ -f "$NCU_REPORT.ncu-rep" ]]; then
            log "[$MODE] report: $NCU_REPORT.ncu-rep ($(du -h "$NCU_REPORT.ncu-rep" | cut -f1))"
        else
            echo "ERROR: ncu failed for $MODE (exit $status), see $RESULTS/$MODE.ncu.log"; failed+=("$MODE ncu")
            counterHint "$RESULTS/$MODE.ncu.log"
        fi
    fi
    rm -rf "$FRAMES"                                            # frames can be large (explode)
done

log "Results in $RESULTS:"
ls -lh "$RESULTS"
if (( ${#failed[@]} )); then echo "FAILED modes: ${failed[*]}"; exit 1; fi
