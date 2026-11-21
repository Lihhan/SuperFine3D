#!/bin/bash
# Detail-estimator training data under $SF3D_DATA/estimator, from the asset and crop lists in training/estimator/data:
#   z1    (z1.json: 492 Objaverse assets, 1734 crops) and
#   train (train.json: 18 dense scanned/sculpted assets, 332 crops):
#     GLB download -> Blender mesh dump -> 512 latent of every crop + crop records (pairs.py encode) -> grey crop views
#     (hires/<split>) -> GT normal targets -> coarse-normal conditions (cond_lr) -> textured + lit views (tex)
#   v2    (v2/v2b0-7.json: 200 relief / scan assets): build_v2.sh (crops drawn per asset with a fixed seed, detail filter).
# usage: training/estimator/build_data.sh GPU_A GPU_B [V2_BATCH ...]     (V2 batches default 0-7)
# Needs setup/prepare_trellis2_variants.sh (the toolkit's Blender scripts for the mesh dump and the textured views) and
# Blender 4.2 ($SF3D_BLENDER). SF3D_EST_LISTS selects another list directory.
set -euo pipefail
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; source "$R/setup/env.sh"
GA=$1; GB=$2; shift 2; BATCHES=("$@"); [ ${#BATCHES[@]} -gt 0 ] || BATCHES=(0 1 2 3 4 5 6 7)
C=$R/training/estimator; LISTS=${SF3D_EST_LISTS:-$C/data}; X=$SF3D_DATA/estimator; Z=$X/objaverse; PYS="$R/setup/python.sh"
[ -f "$R/training/trellis2_variants/data_toolkit/blender_script/dump_mesh.py" ] || { echo "run setup/prepare_trellis2_variants.sh first"; exit 1; }
mkdir -p "$X/ft_nirne" "$X/hires"
for s in z1 train; do
  echo "== $s $(date +%H:%M)"
  "$SF3D_PYTHON" "$C/objaverse_download.py" "$LISTS/$s.json" 2>&1 | tail -n 1
  "$SF3D_PYTHON" "$C/objaverse_dump.py" "$LISTS/$s.json" 2>&1 | tail -n 1
  CUDA_VISIBLE_DEVICES=$GA bash "$PYS" "$R/superfine3d/encode/pairs.py" encode "$LISTS/$s.json" "$Z/crops/$s" 2>&1 | grep -E "ENCODE_DONE|Error|Traceback" | tail -n 1
  CUDA_VISIBLE_DEVICES=$GA bash "$PYS" "$C/hires_render.py" "$X/hires/$s" "$Z/crops/$s/records" 2>&1 | grep -E "HIRES_DONE|Error|Traceback" | tail -n 1
done
SPLITS=z1,train CUDA_VISIBLE_DEVICES=$GA bash "$PYS" "$C/make_targets.py" 0/1 2>&1 | grep -E "DONE|Error|Traceback" | tail -n 1
CUDA_VISIBLE_DEVICES=$GA bash "$PYS" "$C/make_cond.py" 0/1 2>&1 | grep -E "COND_DONE|Error|Traceback" | tail -n 1
ls "$X/hires/z1" "$X/hires/train" | grep -E '_z[0-9]+_[0-9]+$' | LC_ALL=C sort > "$X/ft_nirne/tex_v1.txt"
"$SF3D_PYTHON" "$C/tex_render.py" "$X/ft_nirne/tex_v1.txt" "$GA,$GB" 2 2>&1 | grep -cE " ok " | sed 's/^/tex assets ok /'
bash "$C/build_v2.sh" "$GA" "$GB" "${BATCHES[@]}"
echo "ESTIMATOR_DATA_DONE $X"
