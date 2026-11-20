"""v2 crop records for relief/scan assets (no TRELLIS latents needed): crop centres = area-weighted random face centroids,
zoom 4 (x3) or 2 (x1). usage: records.py BATCH.json OUT_RECORD_DIR [N=12]"""
from os import environ as _E   # install locations, set by setup/env.sh (SF3D_ROOT, SF3D_CKPT, ...)
import sys,os,json,numpy as np,torch
sys.path.insert(0,f'{_E["SF3D_ROOT"]}/superfine3d/encode');import pairs
B,O=sys.argv[1:3];N=int(sys.argv[3]) if len(sys.argv)>3 else 12;os.makedirs(O,exist_ok=True)
for x in json.load(open(B)):
    u=x['sha256'];o=f'{O}/{u}.json'
    if os.path.exists(o) or not os.path.exists(f'{pairs.R}/mesh_dumps/{u}.pickle'):continue
    try:v,f=pairs.load_dump(u)
    except Exception as e:print(u[:10],'load fail',e,flush=True);continue
    T=v[f];a=torch.cross(T[:,1]-T[:,0],T[:,2]-T[:,0],dim=1).norm(dim=1).double();c=T.mean(1)
    g=torch.Generator().manual_seed(int(u[:8],16));idx=torch.multinomial(a/a.sum(),N,replacement=False,generator=g)
    crops=[dict(id=f'{u}_z{z}_{k}',zoom=z,center=c[i].tolist(),kind='v2_area') for k,i in enumerate(idx.tolist()) for z in [2 if k%4==3 else 4]]
    json.dump(dict(sha256=u,crops=crops,name=x.get('name')),open(o,'w'));print(u[:10],len(f),'faces',len(crops),'crops',flush=True)
print('RECORDS_DONE',flush=True)
