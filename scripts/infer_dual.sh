#!/bin/bash
# Two photos of one object -> 4096^3 mesh (the pipeline of the paper's two-view figure).
# usage: scripts/infer_dual.sh VIEW0 VIEW1 OUT_DIR [FOCAL_PX] [RES]
#   VIEWi    RGBA image (alpha masks the object; RGB = the full-resolution photo), or PHOTO.jpg:MASK.png
#            Both photos from the same camera and lens (one focal length in pixels). VIEW0 is the detail view.
#   FOCAL_PX focal length of the photos in pixels (from EXIF: f_mm / sensor_width_mm * width_px). Default: the MASt3R
#            pointmap focal of the pair (printed by step 3).
#   RES      4096 (default) | 2048 | 1024
# Steps: 1 single-view Pixal3D 512 on VIEW0 (defines the world)  2 PnP of VIEW0 in that world  3 MASt3R dense matches
#   4 rectified views with the true focal, relative pose from the essential matrix  5 Pixal3D multi-view 512 start (seed 42)
#   6 exact camera of VIEW0 in the multi-view world (no fit)  7 scripts/infer.sh with that start (SF3D_START=given).
#   -> OUT_DIR/final_RES.ply; intermediate files in OUT_DIR/dual/
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/setup/env.sh"
OUT=$(realpath -m "$3"); FT=${4:-}; RES=${5:-4096}; W=$OUT/dual; mkdir -p "$W"; DU=$SF3D_ROOT/superfine3d/dual
view(){ # VIEW -> RGBA png path
  if [[ "$1" == *:* ]]; then "$SF3D_PYTHON" -c "import sys,numpy as np;from PIL import Image
p,m,o=sys.argv[1:4];a=np.asarray(Image.open(p).convert('RGB'));k=np.asarray(Image.open(m).convert('L'))
assert k.shape==a.shape[:2],(k.shape,a.shape);Image.fromarray(np.dstack([a,k])).save(o)" "${1%%:*}" "${1#*:}" "$2"; echo "$2"
  else realpath "$1"; fi; }
IMG0=$(view "$1" "$W/view0_input.png"); IMG1=$(view "$2" "$W/view1_input.png")
PXENV=(env HF_HUB_OFFLINE="${PIXAL_HF_OFFLINE:-1}" HF_HUB_CACHE="$SF3D_PIXAL3D_HF_CACHE" PYTHONPATH="$SF3D_ROOT/third_party/site-packages")
echo "== 1 single-view Pixal3D 512 on view 0"
[ -f "$W/single/mesh_512_native.ply" ] || (cd "$SF3D_ROOT/superfine3d/detail/fig5" && "${PXENV[@]}" "$SF3D_PIXAL3D_PYTHON" pixal512_worker.py "$IMG0" "$W/single" > "$W/single.log" 2>&1)
echo "== 2 PnP of view 0"
"$SF3D_PYTHON" "$DU/pnp_view.py" "$W/single" "$IMG0" "$W/p0" | grep -E "PnP|IoU"
echo "== 3 MASt3R matches view 0 - view 1"
env HF_HUB_OFFLINE="${MAST3R_HF_OFFLINE:-1}" HF_HUB_CACHE="${SF3D_MAST3R_HF_CACHE:-$HF_HUB_CACHE}" PYTHONPATH="$SF3D_ROOT/third_party/site-packages:${SF3D_MAST3R_DEPS:-}" \
  "$SF3D_PIXAL3D_PYTHON" "$DU/mast3r_match.py" "$IMG0" "$IMG0" "$IMG1" "$IMG1" "$W/m1_m3.npz" | grep -E "matches"
[ -n "$FT" ] || FT=$("$SF3D_PYTHON" -c "import numpy as np;print(round(float(np.load('$W/m1_m3.npz')['f0_m'])))")
echo "   focal $FT px"
echo "== 4 rectified views"
V0="$W/single" P0POSE="$W/p0_pose.json" "$SF3D_PYTHON" "$DU/make_views_tf.py" "$W/views" "$FT" "$IMG0" "$IMG0" "$W/m1_m3.npz:$IMG1:$IMG1" | grep -E "^view|true f"
echo "== 5 Pixal3D multi-view 512 start"
[ -f "$W/mv/mesh_512_native.ply" ] || "${PXENV[@]}" SEED="${SEED:-42}" "$SF3D_PIXAL3D_PYTHON" "$DU/pixal512mv_worker.py" "$W/views" "$W/mv" > "$W/mv.log" 2>&1
grep -h PIXAL512MV_DONE "$W/mv.log"
echo "== 6 start and exact camera of view 0"
mkdir -p "$W/start"
"$SF3D_PYTHON" -c "import numpy as np,trimesh                 # Y-up -> Z-up, written as inject_lib.write_ply (binary ply, float xyz, uchar+int faces)
m=trimesh.load('$W/mv/mesh_512_native.ply',process=False);v=np.asarray(m.vertices,dtype=np.float32);v=np.stack([v[:,0],-v[:,2],v[:,1]],1).astype(np.float32);f=np.asarray(m.faces).astype(np.int32)
with open('$W/start/mesh_512.ply','wb') as fh:
    fh.write(f'ply\nformat binary_little_endian 1.0\nelement vertex {len(v)}\nproperty float x\nproperty float y\nproperty float z\nelement face {len(f)}\nproperty list uchar int vertex_indices\nend_header\n'.encode())
    fh.write(v.tobytes());fa=np.zeros(len(f),dtype=[('n','u1'),('i','<i4',(3,))]);fa['n']=3;fa['i']=f;fh.write(fa.tobytes())"
"$SF3D_PYTHON" "$DU/chain_cam.py" "$W/start" "$W/views" "$FT" "$IMG0" "$IMG0" 2048 | grep -E "IoU"
echo "== 7 refinement to ${RES}^3"
SF3D_START=given SF3D_START_DIR="$W/start" bash "$SF3D_ROOT/scripts/infer.sh" "$IMG0" "$OUT" "$RES" given
