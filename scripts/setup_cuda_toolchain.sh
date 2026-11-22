#!/usr/bin/env bash
# Installs the CUDA compiler (nvcc), CUDA headers, CUB/Thrust and the CUDA
# runtime library into ./cuda-toolchain from NVIDIA's pip packages.
# No root access and no system CUDA toolkit needed. Run it once on a machine
# with internet access (the login node), then:
#
#   source cuda-toolchain/env.sh
#   make
#
# The version defaults to 13.0 to match a driver that reports "CUDA Version:
# 13.0" in nvidia-smi. Override with CUDA_VER=13.1 etc.
set -euo pipefail
cd "$(dirname "$0")/.."

VER=${CUDA_VER:-13.0}
DEST=${DEST:-$PWD/cuda-toolchain}

python3 -m pip install --quiet --upgrade --target "$DEST/site" \
  "nvidia-cuda-nvcc==$VER.*" "nvidia-nvvm==$VER.*" "nvidia-cuda-crt==$VER.*" \
  "nvidia-cuda-runtime==$VER.*" "nvidia-cuda-cccl==$VER.*"

nvcc_path=$(find "$DEST/site" -type f -name nvcc | head -1)
if [ -z "$nvcc_path" ]; then
  echo "nvcc not found in the installed packages" >&2
  exit 1
fi
root=$(cd "$(dirname "$nvcc_path")/.." && pwd)

# pip does not keep the executable bit or the unversioned library name.
chmod +x "$root"/bin/* "$root"/nvvm/bin/* 2>/dev/null || true
cudart=$(find "$root/lib" -maxdepth 1 -name 'libcudart.so.*' | sort | head -1)
ln -sf "$(basename "$cudart")" "$root/lib/libcudart.so"
# Some nvcc versions look for the libraries in lib64.
if [ ! -e "$root/lib64" ]; then ln -s lib "$root/lib64"; fi

cat > "$DEST/env.sh" <<ENV
export CUDA_HOME="$root"
export PATH="$root/bin:\$PATH"
export LD_LIBRARY_PATH="$root/lib:\${LD_LIBRARY_PATH:-}"
ENV

echo "installed: $("$root/bin/nvcc" --version | grep release)"
echo "next: source $DEST/env.sh && make"
