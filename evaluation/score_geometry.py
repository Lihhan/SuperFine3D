"""Score an extra mesh for a toys4k test asset with the frozen Table-1 protocol (detail100): v4 normalization, v4.fit
rigid alignment (prepare: 100k samples, seeds 101/102), CD-L1/P95 + F-scores (2M samples/direction, point-to-triangle),
DQ = meshybench mesh_details quality_4096 (Z-up -> Y-up rotation as the worker). Writes OUT.json; touches nothing in the eval tree.
usage: score_extra.py KEY MESH OUT.json"""
from os import environ as _E   # install locations, set by setup/env.sh (SF3D_ROOT, SF3D_CKPT, ...)
import sys,json,time,numpy as np,torch
E=f'{_E["SF3D_TOYS4K_EVAL"]}'
sys.path[:0]=[E+'/detail_pilot_TC002/meshybench/src',E+'/code']
import toys_metrics_v4 as v4,detail100_distance as dx,detail_batched_render
from fair_alignment import transform_mesh
from meshybench import mesh_details
from meshybench.mesh_details import metric
metric.raster_view=detail_batched_render.raster_view;torch.set_grad_enabled(False);torch.set_num_threads(int(__import__("os").environ.get("SCORE_THREADS","4")))
key,mesh,out=sys.argv[1:4];t0=time.time()
gt,_=v4.normalize(v4.load(__import__('os').environ.get('GT_PLY') or f'{E}/{key}/input_data/GT_normalized.ply'));m,n=v4.normalize(v4.load(mesh))   # GT_PLY: non-Toys4k assets
r,t,a=v4.fit(v4.base.sampled(m,100000,101),v4.base.sampled(gt,100000,102));m=transform_mesh(m,r,t)
p=v4.base.sampled(m,2000000,101);g=v4.base.sampled(gt,2000000,102)
pd=v4.base.mesh_distances_squared(gt,p);gd=v4.base.mesh_distances_squared(m,g)
st=dx.stats(pd,gd);fs={str(tau):v4.base.fscore_squared(pd,gd,tau*tau) for tau in [.0001,.001,.002,.01]};del p,g,pd,gd
m.vertices=np.asarray(m.vertices)@np.array([[1,0,0],[0,0,-1],[0,1,0]])
dq=mesh_details.evaluate(m).to_dict()
res=dict(asset=key,mesh=mesh,alignment=dict(rotation=np.asarray(r).tolist(),translation=np.asarray(t).tolist(),objective=a.get('objective')),
 table=dict(CD_L1_x1e3=st['CD_L1']*1e3,P95_x1e3=st['P95']*1e3,F002=fs['0.002']['fscore']*100,F01=fs['0.01']['fscore']*100,DQ=dq['values']['quality_4096']),
 distance=st,fscore=fs,richness=dq['values'],seconds=time.time()-t0)
json.dump(res,open(out,'w'),indent=1);print(json.dumps(res['table']))
