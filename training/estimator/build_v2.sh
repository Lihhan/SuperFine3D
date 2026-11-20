#!/bin/bash
# Estimator data v2 (relief / scan assets), per batch of Objaverse uids ($SF3D_DATA/estimator/objaverse/v2bN.json):
# download -> Blender mesh dump -> crop records -> grey crop views (hires/v2) -> GT normal targets -> detail filter
# -> textured + lit Blender renders of the kept crops.   usage: build_v2.sh GPU_A GPU_B BATCH_IDS...
set -euo pipefail
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; source "$R/setup/env.sh"
GA=$1; GB=$2; shift 2
C=$R/training/estimator; X=$SF3D_DATA/estimator; Z=$X/objaverse; V=$X/ft_nirne/v2; REC=$Z/zoom_data_v2relief/records
mkdir -p "$V" "$REC" "$X/hires/v2"
for b in "$@"; do
  B=$Z/v2b$b.json; echo "== batch $b $(date +%H:%M)"
  "$SF3D_PYTHON" "$C/objaverse_download.py" "v2b$b.json" 2>&1 | tail -n 1
  "$SF3D_PYTHON" "$C/objaverse_dump.py" "v2b$b.json" 2>&1 | tail -n 1
  bash "$R/setup/python.sh" "$C/v2/records.py" "$B" "$REC" 12 2>&1 | grep -cE "crops$" | sed 's/^/records /'
  CUDA_VISIBLE_DEVICES=$GA bash "$R/setup/python.sh" "$C/hires_render.py" "$X/hires/v2" "$REC" 2>&1 | grep -E "HIRES_DONE|Error|Traceback" | tail -n 1
  SPLITS=v2 CUDA_VISIBLE_DEVICES=$GA bash "$R/setup/python.sh" "$C/make_targets.py" 0/1 2>&1 | grep -E "DONE|Error|Traceback" | tail -n 1
  "$SF3D_PYTHON" "$C/v2/filter.py" 0.05 0.4 8 2>&1 | tail -n 1
  "$SF3D_PYTHON" -c "import json,sys;print('\n'.join(x['uid'] for x in json.load(open(sys.argv[1]))))" "$B" > "$V/b$b.uids"
  grep -F -f "$V/b$b.uids" "$V/keep.txt" > "$V/tex_$b.txt" || true
  "$SF3D_PYTHON" "$C/tex_render.py" "$V/tex_$b.txt" "$GA,$GB" 2 2>&1 | grep -cE " ok " | sed 's/^/tex assets ok /'
  echo "batch $b done: kept views $(wc -l < "$V/tex_$b.txt")"
done
echo V2_BUILD_DONE
