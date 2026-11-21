#!/usr/bin/env bash
# =============================================================================
# run_bvh_bench.sh — ONE profiled run of the dynamic-BVH CUDA harness on ONE
# GPU, producing a single Nsight Systems report (.nsys-rep) to review locally
# (Nsight Systems desktop app, or in a browser via Nsight Streamer).
#
#
# Override settings at submit time without editing, e.g.:
#   sbatch --export=ALL,MODE=wave,NUM_FRAMES=20 run_bvh_bench.sh
# =============================================================================

# ----------------------------- Slurm resources (EDIT ME) --------------------
#SBATCH --job-name=bvh-profile
#SBATCH --partition=instruction
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1                 # e.g. gpu:a100:1 to request a GPU type
#SBATCH --cpus-per-task=8            # used to compile in parallel
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out      # the logs/ dir must exist before sbatch
#SBATCH --error=logs/%x-%j.out

set -euo pipefail

# ----------------------------- Modules (EDIT ME) ----------------------------

module load cuda          # provides nvcc and nsys (Nsight Systems CLI)
module load cmake         # needs >= 3.24
# module load gcc 2>/dev/null || true

# ----------------------------- Settings -------------------------------------
SRC_DIR="${SLURM_SUBMIT_DIR:-$PWD}"               # folder with CMakeLists.txt
MESH="${MESH:-$SRC_DIR/dragon_recon/dragon_vrip.ply}"
MODE="${MODE:-explode}"                           # rigid | twist | wave | explode
NUM_FRAMES="${NUM_FRAMES:-10}"                    # frames in the profiled run
CAM_RES="${CAM_RES:-1024}"
RND_RAYS="${RND_RAYS:-1048576}"
REPEATS="${REPEATS:-5}"
CUDA_ARCH="${CUDA_ARCH:-native}"                  # or e.g. 80 (A100), 90 (H100)

JOB_ID="${SLURM_JOB_ID:-local}"
RESULTS="$SRC_DIR/results/$JOB_ID"
BUILD_DIR="$SRC_DIR/build-gpu"
REPORT="$RESULTS/bvh_${MODE}_${JOB_ID}"            # nsys adds .nsys-rep
SCRATCH="${TMPDIR:-/tmp}/bvh-$JOB_ID"             # node-local scratch for frames
mkdir -p "$RESULTS" "$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT

log() { echo "[$(date '+%H:%M:%S')] $*"; }

command -v nsys > /dev/null || { echo "ERROR: nsys not found (is the cuda module loaded?)"; exit 1; }
[[ -f "$MESH" ]] || { echo "ERROR: mesh not found: $MESH"; exit 1; }
[[ -d "$SRC_DIR/_deps/tinyobjloader" && -d "$SRC_DIR/_deps/happly" ]] \
    || { echo "ERROR: $SRC_DIR/_deps is missing; run ./prefetch_deps.sh on the login node first"; exit 1; }

# Record what produced the report. Your local viewer must be the SAME OR NEWER
# Nsight Systems version than the one printed here.
{
    echo "job:     $JOB_ID on $(hostname), $(date)"
    echo "run:     mode=$MODE frames=$NUM_FRAMES cam=${CAM_RES}x${CAM_RES} rnd=$RND_RAYS repeats=$REPEATS"
    echo "nsys:    $(nsys --version)"
    echo "nvcc:    $(nvcc --version | tail -1)"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | sed 's/^/gpu:     /'
} | tee "$RESULTS/run_info.txt"

# ----------------------------- Build ----------------------------------------
log "Building"
CMAKE_ARGS=(-S "$SRC_DIR" -B "$BUILD_DIR"
            -DCMAKE_BUILD_TYPE=Release
            -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH")
cmake "${CMAKE_ARGS[@]}" > "$RESULTS/build.log" 2>&1
cmake --build "$BUILD_DIR" -j "${SLURM_CPUS_PER_TASK:-8}" >> "$RESULTS/build.log" 2>&1 \
    || { echo "ERROR: build failed, see $RESULTS/build.log"; tail -20 "$RESULTS/build.log"; exit 1; }
[[ -x "$BUILD_DIR/bvh_cuda_bench" ]] || { echo "ERROR: bvh_cuda_bench not built (CUDA not found?)"; exit 1; }

# ----------------------------- Frames ---------------------------------------
log "Generating $NUM_FRAMES '$MODE' frames"
"$BUILD_DIR/make_frames" "$MESH" "$MODE" "$NUM_FRAMES" "$SCRATCH/frames"

# ----------------------------- Single profiled run --------------------------
log "Profiling with Nsight Systems"
nsys profile \
    --trace=cuda,nvtx \
    --sample=none --cpuctxsw=none \
    --force-overwrite=true \
    --output="$REPORT" \
    "$BUILD_DIR/bvh_cuda_bench" "$SCRATCH/frames/frame_%04d.ply" "$NUM_FRAMES" \
        "$RESULTS/benchmark.csv" \
        --cam-res "$CAM_RES" --rnd-rays "$RND_RAYS" --repeats "$REPEATS" \
    2>&1 | tee "$RESULTS/run.log"

[[ -f "$REPORT.nsys-rep" ]] || { echo "ERROR: no report produced, see $RESULTS/run.log"; exit 1; }
log "Done. Report: $REPORT.nsys-rep ($(du -h "$REPORT.nsys-rep" | cut -f1))"