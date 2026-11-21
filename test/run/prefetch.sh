#!/usr/bin/env bash
# prefetch_deps.sh — run ONCE on a LOGIN node (which has internet).
# The Slurm job (run_bvh_bench.sh) then compiles everything on the GPU node,
# using these local copies, so the compute node needs no internet access.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p _deps

# Library versions used by the build (CMakeLists.txt reads them from _deps/)
[[ -d _deps/tinyobjloader ]] || git -c advice.detachedHead=false clone --depth 1 --branch v2.0.0rc13 \
    https://github.com/tinyobjloader/tinyobjloader.git _deps/tinyobjloader

[[ -d _deps/happly ]] || git -c advice.detachedHead=false clone --depth 1 --branch v1.1.0 \
    https://github.com/nmwsharp/happly.git _deps/happly

echo "Dependencies ready in $PWD/_deps"
