#!/bin/bash
# Pixal3D 512 + its image-only camera, exactly start.sh's pixal512_M + pixal512_M_cam steps (same scripts and thread settings),
# run as one background chain by bench_t1 so that it overlaps the TRELLIS.2 reference frame and its camera fit.
# usage: pixal_chain.sh INPUT OUT   -> OUT/{mesh_512_native.ply, zup/mesh_512.ply, zup/cam_fit.json, zup/cam_refined.json, chain_times.json}
INP=$1;O=$2;P=${SF3D_ROOT}/superfine3d/detail;TF=${SF3D_ROOT}/setup/python.sh;ULTRA=${SF3D_PYTHON}
PXV=${SF3D_PIXAL3D_PYTHON};mkdir -p $O/zup
now(){ date +%s.%N; };t0=$(now)
(cd $P/fig5 && $PXV $P/bench/pixal_peak.py $INP $O) > $O/worker.log 2>&1 || { echo "PIXAL_WORKER_FAILED"; exit 1; }
t1=$(now);cp $INP $O/zup/input.png
NT=${PXCHAIN_THREADS:-4}   # thread count of the camera fit/refine (exact speed-up; 4 = the queue's lib.sh)
export OMP_NUM_THREADS=4 CAMFIT_THREADS=$((NT<4?NT:4)) CAMREF_THREADS=$NT CAMREF_HULL=1 CAMREF_LEAN=1;unset PYTHONPATH   # lib.sh settings of the queue
[ "$PXCHAIN_FAST" = 1 ] && export CAMFIT_FASTNORM=2 CAMREF_THREADS=12 CAMREF_LOCKSTEP=1 CAMREF_FNCACHE=1 CAMREF_MEMO=1 CAMREF_GRIDBATCH=72 CAMREF_FASTNORM=1   # exact camera speed-ups (bench/camab)
$ULTRA $P/fit_input_camera.py $O/zup/input.png $O/zup/mesh_512.ply $O/zup/cam_fit.json > $O/fit.log 2>&1 || { echo "PIXAL_FIT_FAILED"; exit 1; }
t2=$(now)
bash $TF $P/whole/camera_refine.py $O/zup/input.png $O/zup/mesh_512.ply $O/zup/cam_fit.json $O/zup/cam_refined.json > $O/refine.log 2>&1 || { echo "PIXAL_REFINE_FAILED"; exit 1; }
t3=$(now)
python3 -c "import json;json.dump(dict(pixal3d_512=$t1-$t0,camera_fit=$t2-$t1,camera_refine=$t3-$t2,total=$t3-$t0),open('$O/chain_times.json','w'),indent=1)"
