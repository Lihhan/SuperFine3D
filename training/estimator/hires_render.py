"""High-resolution crop condition views (upper-bound test: the image really contains HR detail).
For every zoom crop, render the *full* source mesh (dump frame = latent frame, so no frame fit is needed) with a
perspective camera aimed at the crop centre whose field of view just covers the crop cube: 1024 px over the crop
extent, i.e. ~1 px per HR(1024-crop) voxel. Grey two-light shading (geometry only, no texture), 2x supersampled.
The view direction is the one (of 64) that sees the most in-crop surface.
usage: hires_render.py OUT_ROOT RECORD_DIR [instance_list]   -> OUT_ROOT/<crop id>/{view.png,cam.json}"""
from os import environ as _E   # install locations, set by setup/env.sh (SF3D_ROOT, SF3D_CKPT, ...)
import sys,os,json,glob,math,numpy as np,torch
sys.path.insert(0,f'{_E["SF3D_ROOT"]}/superfine3d/encode')
import pairs
import nvdiffrast.torch as dr
from PIL import Image
OUT,RECD=sys.argv[1],sys.argv[2];keep=set(open(sys.argv[3]).read().split()) if len(sys.argv)>3 else None
RES=1024;SS=2;D=1.5;MARGIN=1.12
glctx=dr.RasterizeCudaContext()
def fib(n):
    i=np.arange(n)+.5;phi=np.arccos(1-2*i/n);th=np.pi*(1+5**.5)*i
    return np.stack([np.cos(th)*np.sin(phi),np.sin(th)*np.sin(phi),np.cos(phi)],1)
def camera(center,d,zoom):
    e=center+D*d;fwd=-d;up=np.array([0,0,1.]) if abs(d[2])<.95 else np.array([0,1.,0])
    r=np.cross(fwd,up);r/=np.linalg.norm(r);u=np.cross(r,fwd)
    half=MARGIN*0.5/zoom
    f=0.5/(half/D)                         # normalized focal: half-extent at distance D maps to u=0 / 1
    M=np.stack([f*r+.5*fwd,-f*u+.5*fwd,fwd]);P=np.c_[M,-M@e]
    return P,dict(eye=e.tolist(),r=r.tolist(),u=u.tolist(),fwd=fwd.tolist(),f=f)
def render(v,f,P,res,near=.05,far=4.):
    h=torch.cat([v,torch.ones(len(v),1,device=v.device)],1)@torch.from_numpy(P).float().cuda().T  # (u*w,v*w,w)
    w=h[:,2];a=(far+near)/(far-near);b=-2*far*near/(far-near)
    clip=torch.stack([2*h[:,0]-w,2*h[:,1]-w,a*w+b,w],1)[None]   # ndc = 2u-1 ; row i <-> v=(i+.5)/res
    out=None;CH=8_000_000             # nvdiffrast triangle ids are float32 (exact below 2^24): rasterize in chunks, z-merge
    for a0 in range(0,len(f),CH):
        r,_=dr.rasterize(glctx,clip,f[a0:a0+CH].int().contiguous(),(res,res));r=r[0].double();hit=r[...,3]>0
        r[...,3]=torch.where(hit,r[...,3]+a0,0.)
        if out is None:out=r;continue
        better=hit&((out[...,3]==0)|(r[...,2]<out[...,2]));out=torch.where(better[...,None],r,out)
    return out
def shade(v,f,rast,cam):
    tri=rast[...,3].long()-1;m=tri>=0
    T=v[f];n=torch.cross(T[:,1]-T[:,0],T[:,2]-T[:,0],dim=1);n=n/n.norm(dim=1,keepdim=True).clamp_min(1e-12)
    V=-torch.tensor(cam['fwd'],device=v.device).float();R=torch.tensor(cam['r'],device=v.device).float();U=torch.tensor(cam['u'],device=v.device).float()
    nn_=n[tri.clamp_min(0)];nn_=nn_*torch.where((nn_@V)<0,-1.,1.)[...,None]
    L=V+.5*U-.3*R;L=L/L.norm()
    img=(.08+.9*(.65*(nn_@L).clamp(0,1)+.35*(nn_@V).clamp(0,1)))*m
    return img,m
os.makedirs(OUT,exist_ok=True);DIRS=fib(64)
recs=[json.load(open(p)) for p in sorted(glob.glob(RECD+'/*.json'))]
for r in recs:
    crops=[c for c in r['crops'] if (keep is None or c['id'] in keep) and not os.path.exists(f"{OUT}/{c['id']}/cam.json")]
    if not crops:continue
    v,f=pairs.load_dump(r['sha256']);v=v.cuda();f=f.cuda();cen=v[f].mean(1)
    for c in crops:
        ctr=np.array(c['center'],dtype=np.float64);z=c['zoom']
        inb=((cen-torch.from_numpy(ctr).float().cuda()).abs()*z<=.5).all(1)
        best=None
        for d in DIRS:
            P,cam=camera(ctr,d,z);rast=render(v,f,P,256);tri=rast[...,3].long()-1
            s=int(inb[tri[tri>=0]].sum()) if (tri>=0).any() else 0
            if best is None or s>best[0]:best=(s,d)
        P,cam=camera(ctr,best[1],z);rast=render(v,f,P,RES*SS);img,m=shade(v,f,rast,cam)
        g=torch.nn.functional.avg_pool2d(torch.stack([img,m.float()])[None],SS)[0]
        rgb=(g[0]*255).clamp(0,255).byte().cpu().numpy();al=(g[1]*255).clamp(0,255).byte().cpu().numpy()
        o=f"{OUT}/{c['id']}";os.makedirs(o,exist_ok=True)
        Image.fromarray(np.dstack([rgb,rgb,rgb,al])).save(o+'/view.png')
        json.dump(dict(P=P.tolist(),center=ctr.tolist(),zoom=z,dir=best[1].tolist(),visible_crop_px256=best[0],**cam),open(o+'/cam.json','w'))
    print(r['sha256'][:12],len(crops),flush=True)
print('HIRES_DONE',flush=True)
