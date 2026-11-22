#!/usr/bin/env bash
# Builds one binary per combination of compile-time knobs into bin/.
# Override any list from the environment, e.g.:
#   BLOCKS="256 512" LAYOUTS="soa" scripts/build_all.sh
#   REGCAPS="0 16" scripts/build_all.sh     # also build register-capped variants
#   JOBS=8 scripts/build_all.sh             # parallel compiles
set -euo pipefail
cd "$(dirname "$0")/.."

BLOCKS=${BLOCKS:-"128 256 512 1024"}
LAYOUTS=${LAYOUTS:-"aos soa"}
HINTS=${HINTS:-"default cg"}
REGCAPS=${REGCAPS:-"0"}
JOBS=${JOBS:-4}

for b in $BLOCKS; do
  for l in $LAYOUTS; do
    for h in $HINTS; do
      for r in $REGCAPS; do
        echo "BLOCK=$b LAYOUT=$l HINT=$h REGCAP=$r"
      done
    done
  done
done | xargs -P "$JOBS" -I{} sh -c 'make -s {} && echo "built: {}"'

ls bin/ | wc -l | xargs echo "binaries in bin/:"
