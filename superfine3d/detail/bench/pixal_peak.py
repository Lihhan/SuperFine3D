"""Run fig5/pixal512_worker.py (argv: INPUT OUT) in this process, record its peak allocated GPU memory and wall time
(OUT/peak.json), then write the Z-up copy OUT/zup/mesh_512.ply exactly as start.sh's pixal512_M_cam step (v -> [x, -z, y])."""
from os import environ as _E   # install locations, set by setup/env.sh (SF3D_ROOT, SF3D_CKPT, ...)
import sys,runpy,json,time,os
t=time.time()
try:runpy.run_path(f'{_E["SF3D_ROOT"]}/superfine3d/detail/fig5/pixal512_worker.py',run_name='__main__')
finally:
    import torch
    json.dump(dict(peak_gib=torch.cuda.max_memory_allocated()/2**30 if torch.cuda.is_available() else None,seconds=time.time()-t),open(sys.argv[2]+'/peak.json','w'))
import numpy as np,trimesh
O=sys.argv[2];m=trimesh.load(f'{O}/mesh_512_native.ply',process=False);v=np.asarray(m.vertices,dtype=np.float32);v=np.stack([v[:,0],-v[:,2],v[:,1]],1)
os.makedirs(f'{O}/zup',exist_ok=True);trimesh.Trimesh(v,np.asarray(m.faces),process=False).export(f'{O}/zup/mesh_512.ply')
