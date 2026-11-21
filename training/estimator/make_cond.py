"""Conditioning normal maps for the mesh-conditioned estimator: the crop's LR 512 latent decoded to a mesh (crop frame,
object X = center + v/zoom) and rendered with the view camera at 1024 px (same raster/normal code as everywhere else).
Views: hires/{z1,train}; crop latents: objaverse/crops/<split>/shape_latent_512 (pairs.py encode).
Output ft_nirne/cond_lr/<cid>.npz {n (1024,1024,3) f16 object frame, m bool}
usage: make_cond.py SHARD/NSHARD"""
from os import environ as _E   # install locations, set by setup/env.sh (SF3D_ROOT, SF3D_CKPT, ...)
import sys,os,json,numpy as np,torch
sys.path.insert(0,f'{_E["SF3D_ROOT"]}/superfine3d/encode');sys.path.insert(0,f'{_E["SF3D_ROOT"]}/superfine3d/detail');import pairs
import trellis2.models as models
from trellis2.modules.sparse import SparseTensor
from hfdetail import camera,raster
torch.set_grad_enabled(False)
D=f'{_E["SF3D_DATA"]}/estimator';Z=f'{D}/objaverse/crops';OUT=f'{D}/ft_nirne/cond_lr';a_,b_=map(int,sys.argv[1].split('/'));os.makedirs(OUT,exist_ok=True)
items=[(s,c) for s in ('z1','train') if os.path.isdir(f'{D}/hires/{s}') for c in sorted(os.listdir(f'{D}/hires/{s}'))][a_::b_]
dec=models.from_pretrained(pairs.CACHE+'/ckpts/shape_dec_next_dc_f16c32_fp16').eval().cuda();dec.set_resolution(512);n=0
for s,cid in items:
    o=f'{OUT}/{cid}.npz';lat=f'{Z}/{s}/shape_latent_512/{cid}.npz'
    if os.path.exists(o) or not os.path.exists(lat):continue
    cam=json.load(open(f'{D}/hires/{s}/{cid}/cam.json'))
    with np.load(lat) as a:c=np.concatenate([np.zeros((len(a['coords']),1),np.int32),a['coords']],1);m=dec(SparseTensor(torch.from_numpy(a['feats']).float().cuda(),torch.from_numpy(c).int().cuda()))[0]
    X=torch.tensor(cam['center'],device='cuda').float()+m.vertices.float()/cam['zoom'];F=m.faces.long()
    tri=raster.tri_ids(raster.rasterize(camera.clip_coords(X,cam['P']),F,1024));N,mk=raster.normal_map(X,F,tri,cam['fwd'])
    np.savez_compressed(o,n=N.half().cpu().numpy(),m=mk.cpu().numpy());n+=1
    if n%100==0:print(a_,n,flush=True)
print('COND_DONE',a_,n,flush=True)
