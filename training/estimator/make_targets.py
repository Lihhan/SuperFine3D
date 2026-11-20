"""Training targets for the NiRNE detail fine-tune: for every grey crop view in hires/z1 and hires/train (assets disjoint
from the 11 held-out test assets), rasterize the FULL source mesh with the view camera at 1024 px:
  n  (1024,1024,3) f16  face normal (object/dump frame, flipped toward the camera), 0 where no hit
  m  (1024,1024) bool   pixel hit (all visible surface, not only the crop cube)
  mc (1024,1024) bool   hit AND visible point inside the crop cube (same definition as the test oracle)
usage: make_targets.py SHARD/NSHARD   -> ft_nirne/targets/<cid>.npz"""
from os import environ as _E   # install locations, set by setup/env.sh (SF3D_ROOT, SF3D_CKPT, ...)
import sys,os,json,numpy as np,torch
sys.path.insert(0,f'{_E["SF3D_ROOT"]}/superfine3d/encode');import pairs
import nvdiffrast.torch as dr
D=f'{_E["SF3D_DATA"]}/estimator';OUT=f'{D}/ft_nirne/targets';a_,b_=map(int,sys.argv[1].split('/'))
glctx=dr.RasterizeCudaContext()
def raster(v,f,P,res,near=.05,far=8.):
    h=torch.cat([v,torch.ones(len(v),1,device=v.device)],1)@torch.tensor(P,device=v.device).float().T
    w=h[:,2];a=(far+near)/(far-near);b=-2*far*near/(far-near)
    clip=torch.stack([2*h[:,0]-w,2*h[:,1]-w,a*w+b,w],1)[None].contiguous()
    out=None;CH=8_000_000
    for a0 in range(0,len(f),CH):
        r,_=dr.rasterize(glctx,clip,f[a0:a0+CH].int().contiguous(),(res,res));r=r[0].double();hit=r[...,3]>0
        r[...,3]=torch.where(hit,r[...,3]+a0,0.)
        if out is None:out=r;continue
        better=hit&((out[...,3]==0)|(r[...,2]<out[...,2]));out=torch.where(better[...,None],r,out)
    return out
SPL=os.environ.get('SPLITS','z1,train').split(',')   # heldout only for the information probe (never used in training)
items=[(c.split('_')[0],c,f'{D}/hires/{s}/{c}') for s in SPL for c in sorted(os.listdir(f'{D}/hires/{s}'))]
shas=sorted({x[0] for x in items})[a_::b_];items=[x for x in items if x[0] in set(shas)]
cur=None;n_done=0
for sha,cid,p in sorted(items):
    o=f'{OUT}/{cid}.npz'
    if os.path.exists(o) or not os.path.exists(p+'/cam.json'):continue
    if sha!=cur:
        try:v,f=pairs.load_dump(sha);v=v.cuda();f=f.cuda();cur=sha
        except Exception as e:print('no dump',sha[:12],e,flush=True);cur=None;continue
    cam=json.load(open(p+'/cam.json'));rast=raster(v,f,cam['P'],1024);tri=rast[...,3].long()-1;hit=tri>=0
    T=v[f[tri.clamp_min(0)]];bu,bv=rast[...,0:1].float(),rast[...,1:2].float()
    X=bu*T[...,0,:]+bv*T[...,1,:]+(1-bu-bv)*T[...,2,:]
    inside=((X-torch.tensor(cam['center'],device='cuda'))*cam['zoom']).abs().max(-1).values<=.5
    n=torch.cross(T[...,1,:]-T[...,0,:],T[...,2,:]-T[...,0,:],dim=-1);n=n/n.norm(dim=-1,keepdim=True).clamp_min(1e-12)
    V=-torch.tensor(cam['fwd'],device='cuda').float();n=n*torch.where((n@V)<0,-1.,1.)[...,None];n=n*hit[...,None]
    np.savez_compressed(o,n=n.half().cpu().numpy(),m=hit.cpu().numpy(),mc=(hit&inside).cpu().numpy());n_done+=1
    if n_done%50==0:print(a_,n_done,flush=True)
print('TARGETS_DONE',a_,n_done,flush=True)
