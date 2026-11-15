#!/usr/bin/env bash
# Downloads the Stanford 3D Scanning Repository meshes into data/meshes and checks each against a known checksum.
# Run it on your own machine, then copy data/meshes to Euler. Usage: bash scripts/fetch_meshes.sh
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=data/meshes
DL="${OUT}/_dl"
mkdir -p "${DL}"

PUB=https://graphics.stanford.edu/pub/3Dscanrep
DATA=https://graphics.stanford.edu/data/3Dscanrep

# name | url | path inside the archive ("-" for a single .ply.gz) | sha256 of the extracted .ply
MESHES=(
  "bunny|${PUB}/bunny.tar.gz|bunny/reconstruction/bun_zipper.ply|b1acc63bece78444aa2e15bdcc72371a201279b98c6f5d4b74c993d02f0566fe"
  "armadillo|${PUB}/armadillo/Armadillo.ply.gz|-|483563fca686b66168e1e45339c50e71a5dc8a526a04a545a16b9237d5eaa023"
  "dragon|${PUB}/dragon/dragon_recon.tar.gz|dragon_recon/dragon_vrip.ply|fea87ff48f2aba22fb53e7b67c3ff3f7b8c2a3b3a0653af62c48bba67c6d5744"
  "happy|${PUB}/happy/happy_recon.tar.gz|happy_recon/happy_vrip.ply|2283371216d748a08376a3c88698e283cc8f18d10ced348d6d133051bcf217ab"
  "asian_dragon|${DATA}/xyzrgb/xyzrgb_dragon.ply.gz|-|4a31c6b8951b0f9f4b351d183cb5d5d27e2d1a5916b27e6516acfb9a91ad7f85"
)

check() { [ -f "$1" ] && [ "$(sha256sum "$1" | cut -d' ' -f1)" = "$2" ]; }

for entry in "${MESHES[@]}"; do
    IFS='|' read -r name url inner sum <<< "${entry}"
    ply="${OUT}/${name}.ply"
    if check "${ply}" "${sum}"; then
        echo "${name}: present"
        continue
    fi
    archive="${DL}/${name}/$(basename "${url}")"
    mkdir -p "$(dirname "${archive}")"
    [ -f "${archive}" ] || curl -fsSL --retry 3 -o "${archive}" "${url}"
    if [ "${inner}" = "-" ]; then
        gzip -dc "${archive}" > "${ply}"
    else
        tar -xzOf "${archive}" "${inner}" > "${ply}"
    fi
    if ! check "${ply}" "${sum}"; then
        echo "${name}: checksum mismatch, removing ${ply}" >&2
        rm -f "${ply}"
        exit 1
    fi
    echo "${name}: ok"
done
