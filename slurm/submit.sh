#!/usr/bin/env bash
# Builds on the login node (allowed) and hands the job to sbatch. It never runs the benchmark itself.
# Usage: bash slurm/submit.sh slurm/full_build.sh
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v module >/dev/null 2>&1; then
    for f in /usr/share/lmod/lmod/init/bash /etc/profile.d/lmod.sh /etc/profile.d/modules.sh; do
        [ -f "$f" ] && . "$f" && break
    done
fi
module load nvidia/cuda/13.0.0

# A copy made with git archive has no .git, so its commit comes from a COMMIT file instead.
VERSION_ARGS=()
if ! git rev-parse --short HEAD >/dev/null 2>&1; then
    if [ ! -s COMMIT ]; then
        echo "not a git checkout and no COMMIT file; the CSV would say nogit" >&2
        exit 1
    fi
    VERSION_ARGS=(GIT_COMMIT="$(tr -d '[:space:]' < COMMIT)" GIT_DIRTY=)
fi

make both ARCH=sm_89 "${VERSION_ARGS[@]}"
mkdir -p logs data
sbatch "$@"
