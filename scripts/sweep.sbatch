#!/usr/bin/env bash
# One array task per input point cloud. Each task runs every binary in bin/
# (the compile-time knobs) and, inside each, every runtime configuration.
#
#   mkdir -p logs results && sbatch scripts/sweep.sbatch
#
# A finished task leaves results/<name>.csv.done and is skipped if resubmitted,
# so the sweep can be restarted after a timeout or a node failure.
#SBATCH --job-name=lbvh-sweep
#SBATCH --partition=instruction
#SBATCH --gres=gpu:rtx4000ada:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:40:00
#SBATCH --array=0-79
#SBATCH --output=logs/sweep_%A_%a.out
set -euo pipefail

SIZES=(10000 30000 100000 300000 1000000 3000000 10000000 30000000)
DISTS=(uniform gaussian clusters surface duplicates)
SEEDS=(1 2)

id=${SLURM_ARRAY_TASK_ID:?run through sbatch}
size=${SIZES[$((id % 8))]}
dist=${DISTS[$(((id / 8) % 5))]}
seed=${SEEDS[$((id / 40))]}

mkdir -p logs results
out=results/sweep_${dist}_${size}_s${seed}.csv
if [ -f "$out.done" ]; then
  echo "already done: $out"
  exit 0
fi
rm -f "$out"

gpu_state() {
  nvidia-smi --query-gpu=name,uuid,clocks.current.sm,clocks_event_reasons.active,temperature.gpu,power.draw \
    --format=csv,noheader
}
echo "host=$(hostname) commit=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "gpu before: $(gpu_state)"

# Compiler runtime installed by scripts/setup_cuda_toolchain.sh, if used.
if [ -f cuda-toolchain/env.sh ]; then source cuda-toolchain/env.sh; fi

# Binaries in random order, runtime configurations in random order (--shuffle),
# so slow clock or temperature drift does not favour any configuration.
# A failing binary (exit code 3 = tree mismatch) is recorded and the task
# carries on, but the task is not marked done.
failed=0
for bin in $(ls bin/lbvh_* | shuf); do
  if ! "$bin" --n "$size" --dist "$dist" --seed "$seed" --reps 5 --warmup 2 \
    --hostmem all --launch all --verify --shuffle --csv "$out"; then
    echo "FAILED: $bin (exit code $?)"
    failed=1
  fi
done

echo "gpu after: $(gpu_state)"
if [ "$failed" -eq 0 ]; then touch "$out.done"; fi
exit "$failed"
