#!/bin/bash
# Build the 3D-RoPE FlashAttention-2 varlen kernel with row-indexed K/V reads (third_party/flash_attn_3d_rope_fa2)
# into third_party/site-packages. Needs CUDA 12.8 (nvcc) and the CUTLASS revision that FlashAttention v2.8.3 pins
# (its csrc/cutlass submodule). TORCH_CUDA_ARCH_LIST / FA3D_CUDA_ARCHS default to sm_120 (RTX PRO 6000 Blackwell).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
K="$SF3D_ROOT/third_party/flash_attn_3d_rope_fa2"
if [ ! -d "$K/cutlass/include" ]; then
  git clone --depth 1 --branch v2.8.3 https://github.com/Dao-AILab/flash-attention.git "$SF3D_CACHE/flash-attention-2.8.3"
  git -C "$SF3D_CACHE/flash-attention-2.8.3" submodule update --init --depth 1 csrc/cutlass
  ln -sfn "$SF3D_CACHE/flash-attention-2.8.3/csrc/cutlass" "$K/cutlass"
fi
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}" FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-120}" FA3D_CUDA_ARCHS="${FA3D_CUDA_ARCHS:-120}"
export MAX_JOBS="${MAX_JOBS:-4}" NVCC_THREADS="${NVCC_THREADS:-1}"
mkdir -p "$SF3D_CACHE/wheels" "$SF3D_ROOT/third_party/site-packages"
(cd "$K" && "$SF3D_PYTHON" -m pip wheel . --no-build-isolation --no-deps --wheel-dir "$SF3D_CACHE/wheels")
"$SF3D_PYTHON" -m pip install --no-deps --no-compile --target "$SF3D_ROOT/third_party/site-packages" "$SF3D_CACHE"/wheels/flash_attn_3d_rope_fa2-*.whl
echo KERNELS_READY
