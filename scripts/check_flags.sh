#!/usr/bin/env bash
# Compile-only, so it is safe on the login node: registers and spills per kernel, and whether each flag set changes the kernels' machine code.
# Usage: bash scripts/check_flags.sh [ARCH]   (default sm_89)
set -euo pipefail
cd "$(dirname "$0")/.."

ARCH="${1:-sm_89}"
FLAGSETS=("" "--use_fast_math" "-Xptxas -dlcm=cg" "--extra-device-vectorization" "-maxrregcount=32")
KERNELS=(morton_kernel morton_box_kernel hierarchy_kernel refit_kernel refit_box_kernel)

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "nvcc $(nvcc --version | grep -o 'release [0-9.]*'), $ARCH, -O3"
printf '%-6s %-30s' layout flags
for k in "${KERNELS[@]}"; do printf ' %-18s' "${k%_kernel}"; done
echo

for L in aos soa; do
    S=0; [ "$L" = soa ] && S=1
    for i in "${!FLAGSETS[@]}"; do
        F="${FLAGSETS[$i]}"
        D="$TMP/${L}_$i"; mkdir -p "$D"
        nvcc -arch="$ARCH" -cubin -O3 -std=c++17 -lineinfo -Iinclude -Ibench -DLBVH_LAYOUT_SOA=$S \
             -DGIT_COMMIT=\"check\" $F -Xptxas -v bench/run_build.cu -o "$D/k.cubin" 2> "$D/ptxas.log"

        awk -v names="${KERNELS[*]}" '
            BEGIN { split(names, a, " "); for (j in a) want[a[j]] = 1 }
            /Compiling entry function/ {
                k = ""; if (match($0, /_Z[0-9]+/)) { n = substr($0, RSTART + 2, RLENGTH - 2) + 0; k = substr($0, RSTART + RLENGTH, n) }
            }
            /spill stores/ { st = $5; ld = $9 }
            /Used [0-9]+ registers/ && (k in want) { print k, $5, st, ld; k = "" }
        ' "$D/ptxas.log" > "$D/regs.txt"

        cuobjdump -sass "$D/k.cubin" | awk -v dir="$D" '
            /Function :/ { f = ""; if (match($0, /_Z[0-9]+/)) { n = substr($0, RSTART + 2, RLENGTH - 2) + 0; f = substr($0, RSTART + RLENGTH, n) } next }
            f != "" && /^[ \t]+\/\*[0-9a-f]{4,}\*\// {
                sub(/\/\*[0-9a-f]+\*\//, ""); sub(/;.*/, ""); gsub(/[ \t]+/, " "); print > (dir "/" f ".sass")
            }'

        printf '%-6s %-30s' "$L" "${F:-(none)}"
        for k in "${KERNELS[@]}"; do
            read -r _ regs st ld < <(grep "^$k " "$D/regs.txt" || echo "$k ? 0 0")
            cell="${regs}r"
            if [ "$st" != 0 ] || [ "$ld" != 0 ]; then cell+=" spill ${st}/${ld}"; fi
            if [ "$i" -gt 0 ]; then
                if cmp -s "$TMP/${L}_0/$k.sass" "$D/$k.sass"; then cell+=" same"; else cell+=" CHANGED"; fi
            fi
            printf ' %-18s' "$cell"
        done
        echo
    done
done
echo "same / CHANGED: kernel machine code compared with the build without extra flags"
