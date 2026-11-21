"""Mesh -> TRELLIS.2 shape latent, mirroring TRELLIS.2 data_toolkit (dual_grid.py + encode_shape_latent.py): load_dump,
encode, crop_mesh (exact clipping of a mesh to a zoom crop cube). Used by the pipeline and by the estimator data build.
usage (estimator data): pairs.py encode SPLIT.json OUT_DIR [SHARD NSHARDS]
  SPLIT.json = training/estimator/data/<split>.json ({sha256: {uid, crops: [{id, zoom, center}]}}) ->
  OUT_DIR/shape_latent_512/<crop id>.npz (the crop's 512 latent, decoded by make_cond.py) and OUT_DIR/records/<sha256>.json
  (the crops, read by hires_render.py). Mesh dumps from objaverse_dump.py.
"""
from os import environ as _E   # install locations, set by setup/env.sh (SF3D_ROOT, SF3D_CKPT, ...)
import os,sys,json,time,pickle
import numpy as np,torch
R=f'{_E["SF3D_DATA"]}/estimator/objaverse'
CACHE=f'{_E["SF3D_TRELLIS2_WEIGHTS"]}'
torch.set_grad_enabled(False)
import o_voxel,trimesh
import trellis2.models as models
from trellis2.modules.sparse import SparseTensor
ENC=None
def encoder():
    global ENC
    if ENC is None:ENC=models.from_pretrained(CACHE+'/ckpts/shape_enc_next_dc_f16c32_fp16').eval().cuda()
    return ENC

def load_dump(sha):
    with open(f'{R}/mesh_dumps/{sha}.pickle','rb') as f:dump=pickle.load(f)
    vs=[];fs=[];start=0
    for o in dump['objects']:
        if o['vertices'].size==0 or o['faces'].size==0:continue
        vs.append(o['vertices']);fs.append(o['faces']+start);start+=len(o['vertices'])
    v=torch.from_numpy(np.concatenate(vs)).float();f=torch.from_numpy(np.concatenate(fs)).long()
    lo=v.min(0)[0];hi=v.max(0)[0];v=(v-(lo+hi)/2)*(0.99999/(hi-lo).max())
    assert torch.all(v>=-.5) and torch.all(v<=.5)
    return v,f

def encode(v,f,res):
    idx,dual,inter=o_voxel.convert.mesh_to_flexible_dual_grid(vertices=v,faces=f,grid_size=res,aabb=[[-.5]*3,[.5]*3],
        face_weight=1.0,boundary_weight=0.2,regularization_weight=1e-2,timing=False)
    dual=dual*res-idx;assert torch.all(dual>=-1e-3) and torch.all(dual<=1+1e-3)
    dual=(dual.clamp(0,1)*255).to(torch.uint8).float()/255
    bits=(inter[:,0:1]+2*inter[:,1:2]+4*inter[:,2:3]).to(torch.uint8)
    inter=torch.cat([bits%2,bits//2%2,bits//4%2],-1).bool()
    co=torch.cat([torch.zeros_like(idx[:,:1]),idx],-1).int().cuda()
    x=SparseTensor(dual.cuda(),co);z=encoder()(x,x.replace(inter.cuda()))
    return z.coords[:,1:].cpu().numpy().astype(np.int32),z.feats.float().cpu().numpy(),len(idx)

def clip_soup(T,axis,sign,h):
    """Keep the half-space sign*x[axis] <= h of a triangle soup T (n,3,3); exact, orientation preserving."""
    d=h-sign*T[...,axis];inside=d>=0;n=inside.sum(1)
    keep=[T[n==3]]
    for want in (1,2):
        S=T[n==want];dd=d[n==want];ins=inside[n==want]
        if not len(S):continue
        # rotate so the odd vertex (the lone inside for want=1, lone outside for want=2) is first; cyclic order preserved
        odd=np.argmax(ins if want==1 else ~ins,1)
        idx=(odd[:,None]+np.arange(3)[None])%3
        S=np.take_along_axis(S,idx[...,None],1);dd=np.take_along_axis(dd,idx,1)
        A,Bv,C=S[:,0],S[:,1],S[:,2];da,db,dc=dd[:,0:1],dd[:,1:2],dd[:,2:3]
        AB=A+(Bv-A)*(da/(da-db));AC=A+(C-A)*(da/(da-dc))
        if want==1:keep.append(np.stack([A,AB,AC],1))
        else:keep.append(np.stack([AB,Bv,C],1));keep.append(np.stack([AB,C,AC],1))
    return np.concatenate(keep)

def crop_mesh(v,f,center,zoom,h=0.49999):
    V=(v.numpy().astype(np.float64)-center)*zoom;T=V[f.numpy()]
    lo=T.min(1);hi=T.max(1);T=T[((hi>=-h)&(lo<=h)).all(1)]
    for ax in range(3):
        for s in (1,-1):
            if not len(T):return None
            T=clip_soup(T,ax,s,h)
    a=np.linalg.norm(np.cross(T[:,1]-T[:,0],T[:,2]-T[:,0]),axis=1);T=T[a>1e-14]
    if not len(T):return None
    assert np.abs(T).max()<=.5
    return torch.from_numpy(T.reshape(-1,3)).float(),torch.arange(len(T)*3).reshape(-1,3).long()

if __name__=='__main__':
    assert sys.argv[1]=='encode',f'unknown mode {sys.argv[1]}'
    src,D=sys.argv[2],sys.argv[3];shard,ns=(int(sys.argv[4]),int(sys.argv[5])) if len(sys.argv)>5 else (0,1)
    sel=sorted(json.load(open(src)).items())[shard::ns]
    for d in ('records','shape_latent_512'):os.makedirs(f'{D}/{d}',exist_ok=True)
    for s,a in sel:
        rf=f'{D}/records/{s}.json'
        if os.path.exists(rf):continue
        if not os.path.exists(f'{R}/mesh_dumps/{s}.pickle'):print(s[:12],'no mesh dump',flush=True);continue
        t=time.time();v,f=load_dump(s);n=0
        for c in a['crops']:
            o=f"{D}/shape_latent_512/{c['id']}.npz"
            if os.path.exists(o):continue
            lc,lf,_=encode(*crop_mesh(v,f,np.array(c['center']),c['zoom']),512);np.savez_compressed(o,coords=lc,feats=lf);n+=1
        json.dump(dict(sha256=s,uid=a['uid'],crops=a['crops']),open(rf,'w'),indent=1)   # written last: marks the asset done
        print(s[:12],len(a['crops']),'crops',n,'encoded',round(time.time()-t,1),'s',flush=True)
    print('ENCODE_DONE',flush=True)
