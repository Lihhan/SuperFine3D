#!/bin/bash
# Train one stage of the recursive refiner. Every stage fine-tunes the previous one (setup: README, "Training").
#   1  spatial-cell attention (HR LoRA, boundary-prioritized cross-cell routing)   5000 steps, from TRELLIS.2 1024
#   2  cross-scale guidance (LR adapters on the 512^3 parent)                        3000 steps
#   3  joint HR/LR fine-tuning                                                       3000 steps
#   4  refinement objective (keep/detail on decoded geometry)                        3000 steps
#   5  geometry objective (edge/plane terms) -> released refiner                     2000 steps
# usage: training/refiner/run_stage.sh STAGE [key.path=value ...]   (config overrides, e.g. trainer.args.max_steps=10)
set -euo pipefail
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; source "$R/setup/env.sh"
S=$1; shift
case $S in
  1) ENTRY=(-m tf6_2.train);                          CFG=stage1_tf6_2_hr.json;         NAME=tf6_2_hr;   CK=();;
  2) ENTRY=(-m tf8.train);                            CFG=stage2_tf8_B.json;            NAME=tf8_B;      CK=();;
  3) ENTRY=("$R/training/refiner/stages/sc2_train.py"); CFG=stage3_sc2_B_high.json;     NAME=sc2_B_high; CK=(--ckpt none);;
  4) ENTRY=("$R/training/refiner/refine_train.py");   CFG=stage4_R2.json;               NAME=R2;         CK=(--ckpt none);;
  5) ENTRY=("$R/training/refiner/r3_geometry_train.py"); CFG=stage5_R3_geometry_v2.json; NAME=R3_geometry_v2; CK=(--ckpt none);;
  *) echo "STAGE must be 1..5"; exit 2;;
esac
C=$("$SF3D_PYTHON" "$R/setup/render_config.py" "$R/training/refiner/configs/$CFG" "$@")
ROOTS=$("$SF3D_PYTHON" -c 'import os,sys;print(os.path.expandvars(open(sys.argv[1]).read()).strip())' "$R/training/refiner/configs/train_roots.json")
O="$SF3D_RUNS/stage${S}_$NAME"; mkdir -p "$O"
bash "$R/setup/python.sh" -u "${ENTRY[@]}" --trellis-root "$R/training/trellis2_variants" --tf1-root "$R/superfine3d/refiner/code" \
  --config "$C" --output_dir "$O" "${CK[@]}" --data_dir "$ROOTS" --num_gpus 1 --auto_retry 0 --master_port $((29870 + S)) 2>&1 | tee "$O/train.log"
mkdir -p "$SF3D_CKPT/stages"; ln -sfn "$O/ckpts" "$SF3D_CKPT/stages/$NAME"       # the next stage starts from here
[ "$S" = 5 ] && { mkdir -p "$SF3D_CKPT/refiner"; ln -sfn "$O/ckpts/denoiser_step0002000.pt" "$SF3D_CKPT/refiner/denoiser_step0002000.pt"; }
echo "STAGE_${S}_DONE $O"
