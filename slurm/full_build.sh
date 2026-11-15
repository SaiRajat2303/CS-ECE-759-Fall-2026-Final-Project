#!/usr/bin/env zsh
#SBATCH -J lbvh_full
#SBATCH -p instruction
#SBATCH --gres=gpu:rtx4000ada:1
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 0-00:20:00
#SBATCH --array=0-1
#SBATCH -o logs/full_%A_%a.out
#SBATCH -e logs/full_%A_%a.err

set -u

module load nvidia/cuda/13.0.0
cd "${SLURM_SUBMIT_DIR}"

TASK="${SLURM_ARRAY_TASK_ID:-0}"
JOB="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"
NODE="${SLURMD_NODENAME:-$(hostname)}"
CSV="data/full_${JOB}_${TASK}.csv"
TELEM="data/telem_${JOB}_${TASK}.csv"

# Even tasks run AoS first, odd tasks SoA first, so run order cannot favour one layout.
if (( TASK % 2 == 0 )); then LAYOUTS=(aos soa); else LAYOUTS=(soa aos); fi
for L in "${LAYOUTS[@]}"; do
    if [[ ! -x "bin/sm_89/lbvh_build_${L}" ]]; then
        echo "missing bin/sm_89/lbvh_build_${L}: submit with slurm/submit.sh" >&2
        exit 1
    fi
done

MESHES=()
for m in bunny armadillo dragon happy asian_dragon; do
    if [[ -f "data/meshes/${m}.ply" ]]; then
        MESHES+=(--mesh "data/meshes/${m}.ply")
    else
        echo "warning: data/meshes/${m}.ply missing, skipped (see scripts/fetch_meshes.sh)" >&2
    fi
done

export LBVH_COTENANTS=$(squeue -w "${NODE}" -h -t RUNNING -O "JobID" 2>/dev/null | wc -l | tr -d ' ')
export LBVH_THROTTLE=$(nvidia-smi --query-gpu=clocks_event_reasons.active,clocks.sm \
                       --format=csv,noheader 2>/dev/null | head -1 | tr ',' ';' | tr -d ' ')

# Sampled out of process so the benchmark never forks. Appends, so a requeued job keeps its history.
[[ -s "${TELEM}" ]] || echo "ts_ms,node,sm_clock_mhz,power_w,temp_c,throttle" > "${TELEM}"
( while true; do
    S=$(nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu,clocks_event_reasons.active \
        --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    echo "$(date +%s%3N),${NODE},${S}"
    sleep 1
  done ) >> "${TELEM}" &
TELEM_PID=$!
trap 'kill ${TELEM_PID} 2>/dev/null || true' EXIT

STATUS=0
for L in "${LAYOUTS[@]}"; do
    echo "=== layout ${L} on ${NODE} ==="
    ./bin/sm_89/lbvh_build_${L} \
        --points uniform:16-23 \
        --points clusters:16-23 \
        --points duplicates:16-23 \
        "${MESHES[@]}" \
        --block 256 \
        --repeats 15 \
        --warmup-ms 500 \
        --timing-mode both \
        --validate \
        --out "${CSV}" || STATUS=1
done

kill ${TELEM_PID} 2>/dev/null || true
echo "wrote ${CSV} and ${TELEM}"
python3 analysis/summarize.py "${CSV}" || true
exit ${STATUS}
