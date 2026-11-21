#!/bin/bash
# Single image -> 4096^3 mesh (Pixal3D 512^3 start, recursive refinement 1024 -> 2048 -> 4096, image-faithful detail
# injection, fused DiT kernels). The OPT_* speed switches below are exact (same output).
# usage: scripts/infer.sh INPUT_RGBA.png OUT_DIR [RES] [START]
#   RES   = 4096 (default) | 2048 | 1024
#   START = pixal3d (default: Pixal3D native 512^3 output placed in the input frame)
#         | trellis2 (TRELLIS.2 512^3 shape; Pixal3D is not needed)        also settable as SF3D_START
#   -> OUT_DIR/final_RES.ply, OUT_DIR/report.json (stage times, peak GPU memory)
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/setup/env.sh"
IN=$(realpath "$1"); OUT=$(realpath -m "$2"); RES=${3:-4096}; export SF3D_START=${4:-${SF3D_START:-pixal3d}}; mkdir -p "$OUT"
export OPT_PARVOX=16 OPT_CAMTHREADS=4 OPT_PRELOAD=1 OPT_ROUTE_FAST=1 OPT_DEC_FAST=1 OPT_VOX_MIN=512 OPT_EXPORT_LEAN=1 \
       OPT_DEC_GPU=1 OPT_CAMFAST=1 OPT_NOINIT=1 OPT_PXWARM=1 OPT_ENCPIPE=48 OPT_PRELOAD_SPLIT=1
"$SF3D_PYTHON" "$SF3D_ROOT/superfine3d/detail/bench/pipeline.py" "$IN" "$RES" "$OUT/report.json" "$OUT"
echo "mesh ($SF3D_START start): $OUT/final_${RES}.ply"
