# SuperFine3D environment. Edit the locations marked "set me", then:  source setup/env.sh
# Every script in this repository reads its install locations from these variables.
export SF3D_ROOT="${SF3D_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

# ---- runtime (inference)
export SF3D_PYTHON="${SF3D_PYTHON:-$(command -v python)}"                    # Python 3.10 with requirements.txt
export SF3D_CKPT="${SF3D_CKPT:-$SF3D_ROOT/checkpoints}"                     # SuperFine3D weights (see README)
export SF3D_TRELLIS2_REPO="${SF3D_TRELLIS2_REPO:-$SF3D_ROOT/third_party/TRELLIS.2}"   # github.com/microsoft/TRELLIS.2 @ 75fbf01
export SF3D_TRELLIS2_WEIGHTS="${SF3D_TRELLIS2_WEIGHTS:-}"                   # set me: local snapshot of microsoft/TRELLIS.2-4B
export SF3D_PIXAL3D="${SF3D_PIXAL3D:-$SF3D_ROOT/third_party/Pixal3D}"       # contains repo/ (TencentARC/Pixal3D @ f7cf384) and weights/
export SF3D_PIXAL3D_PYTHON="${SF3D_PIXAL3D_PYTHON:-$SF3D_PIXAL3D/venv/bin/python}"
export SF3D_PIXAL3D_HF_CACHE="${SF3D_PIXAL3D_HF_CACHE:-$SF3D_PIXAL3D/hf_cache}"
export SF3D_STABLENORMAL="${SF3D_STABLENORMAL:-$SF3D_ROOT/third_party/StableNormal}"  # github.com/hugoycj/StableNormal @ 1b7203c
export SF3D_NIRNE_WEIGHTS="${SF3D_NIRNE_WEIGHTS:-}"                         # set me: local snapshot of Stable-X/yoso-normal-v1-8-1
export SF3D_CACHE="${SF3D_CACHE:-$SF3D_ROOT/.cache}"
# optional (only for switches that are off in the default configuration)
export SF3D_MOGE="${SF3D_MOGE:-$SF3D_ROOT/third_party/MoGe}" SF3D_UTILS3D="${SF3D_UTILS3D:-$SF3D_ROOT/third_party/utils3d}"
export SF3D_MOGE_WEIGHTS="${SF3D_MOGE_WEIGHTS:-}"

# ---- training
export SF3D_DATA="${SF3D_DATA:-}"                                           # set me: paired 512/1024 latents + condition renders
export SF3D_RUNS="${SF3D_RUNS:-$SF3D_ROOT/runs}"
export SF3D_BLENDER="${SF3D_BLENDER:-$(command -v blender)}"                # Blender 4.2 (estimator data rendering)

# ---- evaluation
export SF3D_TOYS4K_EVAL="${SF3D_TOYS4K_EVAL:-}"                             # set me: Toys4K test assets (GT meshes, baseline meshes)
export SF3D_TOYS4K_4096="${SF3D_TOYS4K_4096:-}"                             # set me: 4096-px input renders of the test views
export SF3D_MESHYBENCH="${SF3D_MESHYBENCH:-$SF3D_ROOT/third_party/meshybench}"   # github.com/meshy-dev/meshybench @ 5cce370

# ---- library path and runtime switches (same as the timed runs)
SP="$SF3D_ROOT/third_party/site-packages"          # CUDA extension wheels (setup/build_extensions.sh) and our kernel (setup/build_kernels.sh)
export PYTHONPATH="$SP:$SF3D_ROOT/superfine3d/refiner/code:$SF3D_ROOT/superfine3d/refiner/decode_code:$SF3D_TRELLIS2_REPO"     # o_voxel is the installed (compiled) package
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export ATTN_BACKEND=flash_attn SPARSE_ATTN_BACKEND=flash_attn PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" MEM_FRACTION="${MEM_FRACTION:-.95}"
mkdir -p "$SF3D_CACHE/triton" "$SF3D_CACHE/cuda" "$SF3D_CACHE/hf_modules"
# FlexGEMM kernel choices of our runs (keyed by GPU name; other GPUs autotune on first use). With them, on an
# RTX PRO 6000 Blackwell the 512^3 start and camera fit match ours exactly; a fresh autotune picks other kernels and they differ slightly.
[ -f "$SF3D_CACHE/flex_autotune.json" ] || cp "$SF3D_ROOT/setup/flex_autotune.json" "$SF3D_CACHE/flex_autotune.json"
export FLEX_GEMM_AUTOTUNE_CACHE_PATH="$SF3D_CACHE/flex_autotune.json" TRITON_CACHE_DIR="$SF3D_CACHE/triton" CUDA_CACHE_PATH="$SF3D_CACHE/cuda" HF_MODULES_CACHE="$SF3D_CACHE/hf_modules"
export PY="$SF3D_PYTHON"
