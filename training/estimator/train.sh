#!/bin/bash
# Fine-tune the one-step diffusion normal estimator (NiRNE, Hi3DGen yoso-normal-v1-8-1) for zoomed-in crops.
#   grey : grey-shaded crops, high-pass loss (LHP=3)                                  -> $SF3D_CKPT/estimator/grey
#   rgb  : grey -> 50% textured + lit views -> mesh-conditioned (UNet starts from the coarse-normal latent)
#          -> v2 relief/scan data with conditioning misalignment/dropout and albedo/photometric/exposure augmentation
#                                                                                     -> $SF3D_CKPT/estimator/rgb
# Data: see README ("Estimator data"). usage: training/estimator/train.sh grey|rgb   (GPU via CUDA_VISIBLE_DEVICES)
set -euo pipefail
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; source "$R/setup/env.sh"
cd "$R/superfine3d/detail"; export PYTHONPATH= HF_HUB_OFFLINE=1
FT="$R/training/estimator/ft_train.py"; E="$SF3D_CKPT/estimator"
grey(){ STEPS=4000 LHP=3 "$SF3D_PYTHON" "$FT" "$E/grey"; }
rgb(){
  [ -f "$E/grey/ckpt_04000.pt" ] || grey
  STEPS=4000 LHP=3 TEX=0.5 INIT="$E/grey/ckpt_04000.pt" "$SF3D_PYTHON" "$FT" "$E/tex50"
  STEPS=4000 BS=4 LHP=3 TEX=0.75 COND=1 COND_LP=0.5 INIT="$E/tex50/ckpt_03000.pt" "$SF3D_PYTHON" "$FT" "$E/C_cond"
  NV2=$(wc -l < "$SF3D_DATA/estimator/ft_nirne/v2/keep.txt"); W=$("$SF3D_PYTHON" -c "print(round(1961*0.79/max($NV2,1),2))")
  SPLITS=z1,train,v2 W_V2=$W HP4_MIN=0.05 STEPS=4000 BS=4 LR=1e-5 LHP=3 TEX=0.75 COND=1 COND_LP=0.5 CLPS=8,16,32,64 \
  CSHIFT=0.06 CROT=5 CSCALE=0.08 CWARP=0.03 CDROP=0.15 PHOTO=0.5 ALB=0.3 EXPO=0.9 SAVE=last \
  INIT="$E/C_cond/ckpt_04000.pt" "$SF3D_PYTHON" "$FT" "$E/rgb"
}
"$1"
