#!/bin/bash
# Build TRELLIS.2's CUDA extensions as wheels and install them into third_party/site-packages:
#   nvdiffrast v0.4.0, CuMesh @ 12289e1, FlexGEMM @ 6dd94a8, o-voxel (TRELLIS.2 @ 75fbf01), flash-attn 2.7.3.
# third_party/site-packages comes first on the path of SuperFine3D and of the Pixal3D worker (both Python 3.10), so the
# Pixal3D environment shares these builds and needs only its own Python requirements.
# Needs CUDA 12.8 nvcc and a host gcc <= 14 (CC / CXX / CUDAHOSTCXX); sources are staged in $SF3D_CACHE/ext_src (not /tmp).
# TORCH_CUDA_ARCH_LIST / FLASH_ATTN_CUDA_ARCHS default to sm_120 (RTX PRO 6000 Blackwell).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
S="$SF3D_CACHE/ext_src"; W="$SF3D_CACHE/ext_wheels"; T="$SF3D_ROOT/third_party/site-packages"; mkdir -p "$S" "$W" "$T"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}" FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-120}" MAX_JOBS="${MAX_JOBS:-8}"
get() {   # get NAME URL REV
  [ -d "$S/$1/.git" ] || git clone --recursive "$2" "$S/$1"
  git -C "$S/$1" checkout -q "$3" && git -C "$S/$1" submodule update --init --recursive
}
get nvdiffrast https://github.com/NVlabs/nvdiffrast.git v0.4.0
get CuMesh https://github.com/JeffreyXiang/CuMesh.git 12289e1
get FlexGEMM https://github.com/JeffreyXiang/FlexGEMM.git 6dd94a8
git -C "$SF3D_TRELLIS2_REPO" submodule update --init --recursive          # o-voxel needs its Eigen submodule
for d in "$S/nvdiffrast" "$S/CuMesh" "$S/FlexGEMM" "$SF3D_TRELLIS2_REPO/o-voxel"; do
  "$SF3D_PYTHON" -m pip wheel "$d" --no-build-isolation --no-deps -w "$W"
done
FLASH_ATTENTION_FORCE_BUILD=TRUE "$SF3D_PYTHON" -m pip wheel flash-attn==2.7.3 --no-binary flash-attn --no-build-isolation --no-deps -w "$W"
"$SF3D_PYTHON" -m pip install --no-deps --no-compile --upgrade --target "$T" \
  "$W"/nvdiffrast-*.whl "$W"/cumesh-*.whl "$W"/flex_gemm-*.whl "$W"/o_voxel-*.whl "$W"/flash_attn-2.7.3-*.whl
"$SF3D_PYTHON" -c "import sys; sys.path.insert(0, '$T'); import nvdiffrast.torch, cumesh, flex_gemm, o_voxel, flash_attn; print('extensions import OK')"
echo EXTENSIONS_READY
