#!/bin/bash
# Refiner training data, built with TRELLIS.2's data_toolkit (commit 75fbf01; the ObjaverseXL loader is in
# training/trellis2_variants/data_toolkit/datasets) for the Objaverse-XL (Sketchfab) assets of training/refiner/data/:
#   metadata -> download -> mesh dump -> dual grids 256/512/1024 -> shape latents 256/512/1024 -> 16 condition views,
# then the layout the training configs read (training/refiner/configs/train_roots.json and stage 2):
#   $SF3D_DATA/metadata/metadata.csv  $SF3D_DATA/shape_latent_1024  $SF3D_DATA/shape_latent_512  $SF3D_DATA/renders_cond
#   $SF3D_DATA/data256/shape_latents/shape_enc_next_dc_f16c32_fp16_256
# The toolkit's working tree is $SF3D_DATA/objaverse_sketchfab. Blender (4.5.1 for the mesh dump, 3.0.1 for the views) is
# downloaded by the toolkit into $SF3D_BLENDER_DIR (default $SF3D_CACHE/blender) instead of /tmp.
# usage: training/refiner/build_data.sh [RANK WORLD_SIZE]    (after setup/prepare_trellis2_variants.sh; one GPU for the encoder)
#        SF3D_REFINER_ASSETS=LIST.txt selects other assets (default training/refiner/data/train_10000.txt).
set -euo pipefail
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; source "$R/setup/env.sh"
RANK=${1:-0}; WS=${2:-1}; E=shape_enc_next_dc_f16c32_fp16
K="$R/training/trellis2_variants/data_toolkit"
[ -f "$K/dump_mesh.py" ] && [ -f "$K/datasets/ObjaverseXL.py" ] || { echo "run setup/prepare_trellis2_variants.sh first"; exit 1; }
L="$(realpath "${SF3D_REFINER_ASSETS:-$R/training/refiner/data/train_10000.txt}")"
T="$SF3D_DATA/objaverse_sketchfab"; export SF3D_BLENDER_DIR="${SF3D_BLENDER_DIR:-$SF3D_CACHE/blender}"
mkdir -p "$T" "$SF3D_BLENDER_DIR" "$SF3D_DATA/metadata" "$SF3D_DATA/data256/shape_latents"
for f in dump_mesh.py dump_pbr.py render_cond.py; do      # local copies of the upstream files (not tracked): Blender location
  sed -i "s|^BLENDER_INSTALLATION_PATH = '/tmp'$|BLENDER_INSTALLATION_PATH = os.environ.get('SF3D_BLENDER_DIR', '/tmp')|" "$K/$f"; done
P="$SF3D_PYTHON"; A=(--rank "$RANK" --world_size "$WS" --instances "$L"); cd "$K"
export HF_HUB_OFFLINE=0                                    # metadata (TRELLIS-500K) and the Objaverse downloads
[ -f "$T/metadata.csv" ] || "$P" build_metadata.py ObjaverseXL --source sketchfab --root "$T"
[ -f "$SF3D_DATA/metadata/metadata.csv" ] || cp "$T/metadata.csv" "$SF3D_DATA/metadata/metadata.csv"   # TRELLIS-500K rows
"$P" download.py ObjaverseXL --root "$T" "${A[@]}"
"$P" build_metadata.py ObjaverseXL --root "$T"
"$P" dump_mesh.py ObjaverseXL --root "$T" "${A[@]}"
"$P" build_metadata.py ObjaverseXL --root "$T"
"$P" dual_grid.py ObjaverseXL --root "$T" --resolution 256,512,1024 "${A[@]}"
"$P" build_metadata.py ObjaverseXL --root "$T"
export HF_HUB_OFFLINE=1
for r in 256 512 1024; do
  "$P" encode_shape_latent.py --root "$T" --resolution $r --enc_pretrained "$SF3D_TRELLIS2_WEIGHTS/ckpts/$E" "${A[@]}"; done
"$P" render_cond.py ObjaverseXL --root "$T" "${A[@]}"
"$P" build_metadata.py ObjaverseXL --root "$T"
ln -sfn "$T/shape_latents/${E}_1024" "$SF3D_DATA/shape_latent_1024"
ln -sfn "$T/shape_latents/${E}_512" "$SF3D_DATA/shape_latent_512"
ln -sfn "$T/shape_latents/${E}_256" "$SF3D_DATA/data256/shape_latents/${E}_256"
ln -sfn "$T/renders_cond" "$SF3D_DATA/renders_cond"
n=$(grep -c True "$SF3D_DATA/shape_latent_1024/metadata.csv" || true); echo "REFINER_DATA_DONE ($n assets with 1024 latents) $SF3D_DATA"
