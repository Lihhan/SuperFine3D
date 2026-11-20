#!/bin/bash
# Table-1 image-space fidelity of the Toys4k baselines (fitted-camera protocol): for each asset, the image-only camera of each
# baseline mesh (TRELLIS.2 1536, Hunyuan3D-2.1 1024, Pixal3D 1536) exactly as for the relief baselines (supp_queue.sh base():
# fit_input_camera.py + camera_refine.py on the raw mesh); when the refined silhouette IoU < 0.85 the Z-up variant of zup_cam.sh
# (unit-normalised, Y-up -> Z-up, full-circle fit_wide.py + guarded refine) is fitted too and the higher IoU is kept (recorded).
# Then fid_t4k_img.py on the asset. usage: cam_t4k.sh LIST   -> t4kcam/KEY/<method>/cam_*.json, t4kimg/KEY.json
LIST=$1;E=${SF3D_TOYS4K_EVAL};W=${SF3D_ROOT}/evaluation;P=${SF3D_ROOT}/superfine3d/detail
PYU=${SF3D_PYTHON};TF=${SF3D_ROOT}/setup/python.sh;ZT=${SF3D_CACHE}/t4kzup;mkdir -p $W/t4kimg
export OMP_NUM_THREADS=4 CAMFIT_THREADS=4 CAMREF_THREADS=4 CAMREF_HULL=1 CAMREF_LEAN=1
gpu(){ nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1; }
pick(){ ls $1/mesh_*.ply 2>/dev/null | grep -v "_384\|_unit\|_zup" | sort -t_ -k2 -n | tail -n 1; }
iou(){ python3 -c "import json;print(json.load(open('$1')).get('iou',0))" 2>/dev/null || echo 0; }
while read K; do [ -f $W/t4kimg/$K.json ] && continue; IMG=$E/$K/input_data/input.png; A=()
  for m in TRELLIS2 Hunyuan21 Pixal3D; do M=$(pick $E/$K/baselines/$m); [ -n "$M" ] || continue; d=$W/t4kcam/$K/$m; mkdir -p $d
    [ -f $d/cam_fit.json ] || nice -n 10 $PYU $P/fit_input_camera.py $IMG $M $d/cam_fit.json > $d/fit.log 2>&1
    [ -f $d/cam_refined.json ] || CUDA_VISIBLE_DEVICES=$(gpu) nice -n 10 bash $TF $P/whole/camera_refine.py $IMG $M $d/cam_fit.json $d/cam_refined.json > $d/refine.log 2>&1
    use="$M@$d";i1=$(iou $d/cam_refined.json)
    if python3 -c "import sys;sys.exit(0 if float('$i1')<0.85 else 1)"; then z=$d/zup;ZM=$ZT/$K/$m/mesh_zup.ply;mkdir -p $z $ZT/$K/$m   # Z-up copy on /tmp (1-3 GB for 4096 meshes), deleted after the asset
      [ -f $ZM ] || $PYU -c "
import trimesh,numpy as np
m=trimesh.load('$M',process=False,force='mesh');v=np.asarray(m.vertices);c=(v.max(0)+v.min(0))/2;s=np.ptp(v,0).max();v=(v-c)/s
m.vertices=np.stack([v[:,0],-v[:,2],v[:,1]],1);m.export('$ZM')"
      [ -f $z/cam_fit.json ] || nice -n 10 $PYU $P/dualcand/fit_wide.py $IMG $ZM $z/cam_fit.json > $z/fit.log 2>&1
      [ -f $z/cam_refined.json ] || CAM_IOU_REL=0.1 CUDA_VISIBLE_DEVICES=$(gpu) nice -n 10 bash $TF $P/whole/camera_refine.py $IMG $ZM $z/cam_fit.json $z/cam_refined.json > $z/refine.log 2>&1
      i2=$(iou $z/cam_refined.json); python3 -c "import sys;sys.exit(0 if float('$i2')>float('$i1') else 1)" && use="$ZM@$z"
      echo "$K $m raw IoU $i1, zup IoU $i2 -> ${use##*@}" >> $W/t4kcam/choices.log; fi
    A+=("$m=$use"); done
  (cd /tmp && CUDA_VISIBLE_DEVICES=$(gpu) nice -n 10 bash $TF $W/fid_t4k_img.py $K $W/t4kimg/$K.json "${A[@]}" > $W/t4kimg/$K.log 2>&1)
  rm -rf "${ZT:?}/${K:?}"; echo "$(date +%T) $K $([ -f $W/t4kimg/$K.json ] && echo ok || echo FAIL)"
done < $LIST; echo "$(date +%T) CAM_T4K_LANE_DONE $LIST"
