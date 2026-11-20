"""Zoom pair construction, mirroring TRELLIS.2 data_toolkit (dual_grid.py + encode_shape_latent.py).
usage:
  pairs.py gate  N          -> zoom=1 re-encode of first N dumped assets vs existing training latents
  pairs.py crops SHARD NSHARDS -> zoom crops for selected500 (sharded)
"""
from os import environ as _E   # install locations, set by setup/env.sh (SF3D_ROOT, SF3D_CKPT, ...)
import os,sys,json,time,pickle,hashlib,traceback
import numpy as np,torch
R=f'{_E["SF3D_DATA"]}/estimator/objaverse';B=f'{_E["SF3D_DATA"]}'
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

def compare(c,fe,path):
    with np.load(path) as a:c0=a['coords'];f0=a['feats']
    k=lambda c:(c[:,0].astype(np.int64)*4096+c[:,1])*4096+c[:,2]
    k1,k0=k(c),k(c0);common,i1,i0=np.intersect1d(k1,k0,return_indices=True)
    d=np.abs(fe[i1]-f0[i0])
    return dict(tokens_new=len(c),tokens_old=len(c0),common=len(common),jaccard=len(common)/len(np.union1d(k1,k0)),
        feat_max_abs=float(d.max()) if len(d) else None,feat_mean_abs=float(d.mean()) if len(d) else None,old_feat_std=float(f0.std()))

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

def pick_centers(sha,rng,per_zoom=2,zooms=(2,4),cap=30000):
    """Detail-weighted crop centers from existing 1024/512 training latents (object frame, normalized [-.5,.5])."""
    with np.load(f'{B}/shape_latent_1024/{sha}.npz') as a:hr=(a['coords'][:,-3:]+.5)/64-.5
    with np.load(f'{B}/shape_latent_512/{sha}.npz') as a:lr=(a['coords'][:,-3:]+.5)/32-.5
    out=[]
    for z in zooms:
        half=.5/z;cands=hr[rng.choice(len(hr),min(256,len(hr)),replace=False)];scored=[]
        for c in cands:
            nh=int((np.abs(hr-c)<=half).all(1).sum());nl=int((np.abs(lr-c)<=half).all(1).sum())
            est=nh*z*z  # surface tokens scale ~ zoom^2
            if nh<16 or est>cap:continue
            scored.append((nh/max(nl,1),c,est))
        if not scored:continue
        scored.sort(key=lambda x:-x[0])
        picks=[scored[0]]+([scored[rng.integers(len(scored))]] if len(scored)>1 else [])
        for k,(ratio,c,est) in enumerate(picks[:per_zoom]):
            c=np.clip(np.round(c*32*z)/(32*z),-.5+half,.5-half)  # grid-aligned: crop voxels coincide with a global (1024*z) grid
            out.append(dict(zoom=z,k=k,center=c.tolist(),ratio=float(ratio),est_tokens=int(est),kind='top_ratio' if k==0 else 'random_valid'))
    return out

def pick_centers_detail(sha,v,f,rng,per_zoom=2,zooms=(2,4),cap=30000):
    """Z2: regions whose surface has normal variation below the LR voxel scale (cell = 2 LR voxels in crop frame)."""
    from detail_map import cell_dispersion
    with np.load(f'{B}/shape_latent_1024/{sha}.npz') as a:hr=(a['coords'][:,-3:]+.5)/64-.5
    out=[]
    for z in zooms:
        half=.5/z;cells,disp,w=cell_dispersion(v.numpy(),f.numpy(),n=3000000,cell=1/(256*z),seed=z)
        if not len(cells):continue
        pr=(disp*w)+1e-9;cand=cells[rng.choice(len(cells),min(300,len(cells)),replace=False,p=pr/pr.sum())]
        scored=[]
        for c in cand:
            c=np.clip(np.round(c*32*z)/(32*z),-.5+half,.5-half)
            m=(np.abs(cells-c)<=half).all(1);nh=int((np.abs(hr-c)<=half).all(1).sum());est=nh*z*z
            if m.sum()<32 or nh<16 or est>cap:continue
            scored.append((float((disp[m]*w[m]).sum()/w[m].sum()),c,est))
        scored.sort(key=lambda x:-x[0]);picked=[]
        for sc,c,est in scored:
            if all(np.abs(c-q).max()>=half for _,q,_ in picked):picked.append((sc,c,est))
            if len(picked)==per_zoom:break
        for k,(sc,c,est) in enumerate(picked):out.append(dict(zoom=z,k=k,center=c.tolist(),detail=sc,est_tokens=int(est),kind='detail_top%d'%k))
    return out

if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='gate':
        n=int(sys.argv[2]);sel=json.load(open(f'{R}/selected500.json'))
        sel=[x for x in sel if os.path.exists(f"{R}/mesh_dumps/{x['sha256']}.pickle")][:n];out=[]
        for x in sel:
            s=x['sha256'];t=time.time();v,f=load_dump(s);rec=dict(sha256=s)
            for res,d in [(1024,'shape_latent_1024'),(512,'shape_latent_512')]:
                c,fe,nv=encode(v,f,res);rec[str(res)]=compare(c,fe,f'{B}/{d}/{s}.npz');rec[str(res)]['voxels']=nv
            rec['seconds']=time.time()-t;out.append(rec);print(json.dumps(rec),flush=True)
        json.dump(out,open(f'{R}/gate_zoom1.json','w'),indent=1)
    elif mode=='crops':
        Z2=os.environ.get('CROP_SET')=='z2'
        shard,ns=int(sys.argv[2]),int(sys.argv[3]);sel=json.load(open(f'{R}/'+('z2_selected500.json' if Z2 else 'selected500.json')))[shard::ns]
        if os.environ.get('REVERSE')=='1':sel=sel[::-1]
        if os.environ.get('MEM_CAP'):torch.cuda.set_per_process_memory_fraction(float(os.environ['MEM_CAP']))
        D=R+('/zoom_data_z2' if Z2 else '/zoom_data_v2');[os.makedirs(f'{D}/{d}',exist_ok=True) for d in ['shape_latent_1024','shape_latent_512','records']]
        for x in sel:
            s=x['sha256'];rf=f'{D}/records/{s}.json'
            if os.path.exists(rf) or not os.path.exists(f'{R}/mesh_dumps/{s}.pickle'):continue
            t=time.time();recs=[]
            try:
                v,f=load_dump(s);rng=np.random.default_rng(int(s[:8],16))
                for p in (pick_centers_detail(s,v,f,rng) if Z2 else pick_centers(s,rng)):
                    cid=f"{s}_z{p['zoom']}_{p['k']}";cm=crop_mesh(v,f,np.array(p['center']),p['zoom'])
                    if cm is None:recs.append(dict(p,id=cid,status='empty'));continue
                    rec=dict(p,id=cid,faces=len(cm[1]))
                    hc,hf,hv=encode(*cm,1024)
                    if len(hc)>32768 or len(hc)<64:recs.append(dict(rec,status='token_cap',tokens_1024=len(hc)));continue
                    lc,lf,lv=encode(*cm,512)
                    np.savez_compressed(f'{D}/shape_latent_1024/{cid}.npz',coords=hc,feats=hf)
                    np.savez_compressed(f'{D}/shape_latent_512/{cid}.npz',coords=lc,feats=lf)
                    recs.append(dict(rec,status='ok',tokens_1024=len(hc),tokens_512=len(lc),voxels_1024=hv))
            except Exception:recs.append(dict(status='error',error=traceback.format_exc()[-800:]))
            json.dump(dict(sha256=s,seconds=time.time()-t,crops=recs),open(rf,'w'),indent=1);print(s[:12],round(time.time()-t,1),[(r.get('id','')[-6:],r['status'],r.get('tokens_1024')) for r in recs],flush=True)
