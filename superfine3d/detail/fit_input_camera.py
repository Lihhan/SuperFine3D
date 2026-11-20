"""Fit the condition camera of a real input image to the generated 1024 mesh (inputs only, no GT):
P maps object points to the preprocess_image crop (normalized). Camera looks at the origin from
direction (yaw, pitch) at the training-render distance for the fov; roll free; 2D similarity (s, du, dv) free.
usage: fit_input_camera.py INPUT_PNG MESH_PLY OUT_JSON"""
from os import environ as _E   # install locations, set by setup/env.sh (SF3D_ROOT, SF3D_CKPT, ...)
import sys,json,numpy as np,trimesh
sys.path.insert(0,f'{_E["SF3D_ROOT"]}/superfine3d/detail');import pixcam
from PIL import Image
from scipy.ndimage import binary_closing,binary_fill_holes
from scipy.optimize import minimize
inp,meshp,outp=sys.argv[1:4];S=256
im=Image.open(inp)
if max(im.size)>1024:sc=1024/max(im.size);im=im.resize((int(im.width*sc),int(im.height*sc)),Image.Resampling.LANCZOS)
from scipy.ndimage import label,binary_opening
if im.mode=='RGBA' and not (np.array(im)[:,:,3]==255).all():al=np.array(im)[:,:,3];fg=al>127
else:  # no alpha: the pipeline's rembg is a no-op here, so the model saw the whole frame; object mask by brightness
    g=np.array(im.convert('L')).astype(float);fg=binary_fill_holes(binary_opening(binary_closing(g>40,iterations=3),iterations=2))
    lab,n=label(fg);fg=lab==(np.argmax(np.bincount(lab.ravel())[1:])+1);al=np.full(g.shape,255,np.uint8)
im=im.convert('RGBA');box=pixcam.preprocess_box(al)
def crop(arr):return np.array(Image.fromarray(arr.astype(np.uint8)*255).crop(box).resize((S,S),Image.Resampling.NEAREST))>127
A=crop(fg);VALID=crop(np.ones_like(fg))
m=trimesh.load(meshp,process=False,force='mesh');X=np.asarray(m.vertices,dtype=np.float64);X=X[np.random.default_rng(0).choice(len(X),min(len(X),int(__import__('os').environ.get('CAMFIT_NPTS','150000'))),replace=False)]
def cam(yaw,pitch,roll,fov):
    fov=float(np.clip(fov,8,70));y,p=np.radians(yaw),np.radians(pitch);d=np.array([np.cos(p)*np.cos(y),np.cos(p)*np.sin(y),np.sin(p)]);r=np.sqrt(3)/2/np.sin(np.radians(fov)/2)
    z=d;x=np.cross([0,0,1.],z);x/=np.linalg.norm(x);yv=np.cross(z,x);a=np.radians(roll)
    x,yv=np.cos(a)*x+np.sin(a)*yv,-np.sin(a)*x+np.cos(a)*yv
    Rw=np.stack([x,yv,z]);w2c=np.c_[Rw,-Rw@(d*r)];f=0.5/np.tan(np.radians(fov)/2)
    K=np.array([[f,0,0.5],[0,-f,0.5],[0,0,1.]])@np.diag([1,1,-1.]);return K@w2c
XH=X
if __import__('os').environ.get('CAMFIT_HULL','0')=='1':   # opt-in exact speed-up: the projected bounding box is attained at hull vertices
    from scipy.spatial import ConvexHull;XH=X[ConvexHull(X).vertices]
_FN=__import__('os').environ.get('CAMFIT_FASTNORM','0')   # opt-in speed-up: '1' = per-column 1-D min/max only (exact: numpy's axis-0 reduce
_FN,_FNH=_FN in ('1','2'),_FN=='2'                         # of an (N,2) array is ~35x slower, same values); '2' also builds the homogeneous
if _FNH:XHh=np.c_[XH,np.ones(len(XH))];Xh=np.c_[X,np.ones(len(X))]   # coordinates once (same values, but BLAS may round differently on another buffer)
def normalize(P):
    if _FN:
        h=(XHh if _FNH else np.c_[XH,np.ones(len(XH))])@P.T;uv=h[:,:2]/h[:,2:3];lo=np.array([uv[:,0].min(),uv[:,1].min()]);hi=np.array([uv[:,0].max(),uv[:,1].max()])
    else:
        uv,_=pixcam.project(P,XH) if False else (lambda h:(h[:,:2]/h[:,2:3],0))(np.c_[XH,np.ones(len(XH))]@P.T)
        lo,hi=uv.min(0),uv.max(0)
    c=(lo+hi)/2;s=1/max(hi-lo)
    return np.array([[s,0,.5-s*c[0]],[0,s,.5-s*c[1]],[0,0,1]])@P
def sil(P):
    h=(Xh if _FNH else np.c_[X,np.ones(len(X))])@P.T;uv=h[:,:2]/h[:,2:3];px=np.clip((uv*S).astype(int),0,S-1);mm=np.zeros((S,S),bool);mm[px[:,1],px[:,0]]=1
    return binary_fill_holes(binary_closing(mm,iterations=2))
def iou(P):M=sil(P)&VALID;return (M&A).sum()/max(1,(M|A).sum())
best=[];_NT=int(__import__('os').environ.get('CAMFIT_THREADS','0'))   # opt-in exact speed-up: independent evaluations in threads
def _g(a):
    yaw,pitch,fov=a;P=normalize(cam(yaw,pitch,0,fov));o=[]
    for sc_ in (1.,1.15,1.3,1.5):   # photo crops the pillar: projection may overfill the frame
        Q=np.array([[sc_,0,.5-.5*sc_],[0,sc_,.5-.5*sc_],[0,0,1]])@P;o.append((iou(Q),yaw,pitch,fov,sc_))
    return o
_ga=[(yaw,pitch,fov) for fov in (15,30,45) for yaw in range(30,61,5) for pitch in range(-15,31,5)]
_gf=__import__('os').environ.get('CAMFIT_GRIDFILE')   # opt-in exact: the grid is computed once and shared (JSON round-trips float64)
if _NT>0:import concurrent.futures as _cf
if _gf and __import__('os').path.exists(_gf):best=[tuple(x) for x in json.load(open(_gf))]
else:
    if _NT>0:
        with _cf.ThreadPoolExecutor(_NT) as _ex:_go=list(_ex.map(_g,_ga))
    else:_go=[_g(a) for a in _ga]
    for o in _go:best+=o
    if _gf:json.dump([[float(b_[0]),*b_[1:]] for b_ in best],open(_gf,'w'))
if __import__('os').environ.get('CAMFIT_GRIDONLY')=='1':sys.exit(0)
best.sort(reverse=True);print('grid top',best[:5],flush=True)
res=[];import os as _os
_starts=best[:int(_os.environ.get('CAMFIT_NSTART','4'))];_only=_os.environ.get('CAMFIT_ONLY');_resd=_os.environ.get('CAMFIT_RESULTS')
if _only is not None:_starts=[_starts[int(_only)]]
if _resd:
    from types import SimpleNamespace as _NS
    for _i in range(len(_starts)):
        _d=json.load(open(f'{_resd}/part{_i}.json'));res.append(_NS(fun=_d['fun'],x=np.array(_d['x'])))
    _starts=[]
def _st(a):
    _,yaw,pitch,fov,sc_=a
    f=lambda q:1-iou(np.array([[np.exp(q[4]),0,q[5]],[0,np.exp(q[4]),q[6]],[0,0,1]])@normalize(cam(q[0],q[1],q[2],q[3])))
    return minimize(f,np.array([yaw,pitch,0,fov,np.log(sc_),.5-.5*sc_,.5-.5*sc_]),method='Powell',options=dict(xtol=1e-3,ftol=1e-5,maxfev=int(__import__('os').environ.get('CAMFIT_MAXFEV','1500'))))
if _NT>0 and _only is None and _starts:
    with _cf.ThreadPoolExecutor(_NT) as _ex:_rs=list(_ex.map(_st,_starts))
else:_rs=None
for _j,_a in enumerate(_starts):
    r=_rs[_j] if _rs is not None else _st(_a);res.append(r)
    if _only is not None:json.dump(dict(fun=float(r.fun),x=[float(x) for x in r.x]),open(f"{_os.environ['CAMFIT_PARTDIR']}/part{_only}.json",'w'));sys.exit(0)
r=min(res,key=lambda r:r.fun);q=r.x;P=np.array([[np.exp(q[4]),0,q[5]],[0,np.exp(q[4]),q[6]],[0,0,1]])@normalize(cam(q[0],q[1],q[2],q[3]))
print('fit yaw %.1f pitch %.1f roll %.1f fov %.1f  IoU %.3f'%(q[0],q[1],q[2],q[3],1-r.fun))
json.dump(dict(P=P.tolist(),yaw=q[0],pitch=q[1],roll=q[2],fov=q[3],iou=1-r.fun,box=[float(b) for b in box],size=im.size),open(outp,'w'),indent=1)
M=sil(P);vis=np.zeros((S,S,3),np.uint8);M=M&VALID;vis[A&M]=[0,180,0];vis[M&~A]=[220,0,0];vis[A&~M]=[0,0,220]
Image.fromarray(np.concatenate([np.array(im.crop(box).convert('RGB').resize((S,S))),vis],1)).save(outp.replace('.json','.png'))
