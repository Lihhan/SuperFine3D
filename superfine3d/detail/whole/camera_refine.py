"""Appearance-aware input-camera fit (fixes fit_input_camera.py's silhouette ambiguities, e.g. a box seen from above vs
below has nearly the same outline). Parametrisation as fit_input_camera.py: camera looks at the origin from (yaw, pitch)
at the training-render distance for the fov, roll free, 2D similarity (s, du, dv) on the preprocess crop.
Objective = (1 - silhouette IoU) + LAM * (1 - NCC(edges of the input crop, edges of the mesh's shaded render)), both at
512 px; edges = Sobel magnitude. Candidates: the silhouette fit and its mirrors (pitch -> -pitch, yaw -> yaw+180, both),
each locally optimised (Powell); best objective wins.
usage: camera_refine.py INPUT_PNG MESH1024 FIT_JSON OUT_JSON   (FIT_JSON from fit_input_camera.py)"""
from os import environ as _E   # install locations, set by setup/env.sh (SF3D_ROOT, SF3D_CKPT, ...)
import os,sys,json,math,numpy as np,torch,torch.nn.functional as F
sys.path.insert(0,f'{_E["SF3D_ROOT"]}/superfine3d/detail');import pixcam
from PIL import Image
from scipy.optimize import minimize
import trimesh
from hfdetail import raster,camera
INP,M1,FJ,OJ=sys.argv[1:5];S=512;LAM=float(__import__('os').environ.get('LAM',1.0))
fit=json.load(open(FJ));src=Image.open(INP)
sc=max(src.size)/max(fit['size']);box=[b*sc for b in fit['box']]
g=np.array(src.convert('L').crop(tuple(int(round(b)) for b in box)).resize((S,S),Image.Resampling.LANCZOS)).astype(np.float32)/255
m=trimesh.load(M1,process=False,force='mesh');V=torch.from_numpy(np.asarray(m.vertices)).float().cuda();Fc=torch.from_numpy(np.asarray(m.faces)).long().cuda()
def cam(yaw,pitch,roll,fov):                                          # = fit_input_camera.cam
    fov=float(np.clip(fov,8,70));y,p=np.radians(yaw),np.radians(pitch);d=np.array([np.cos(p)*np.cos(y),np.cos(p)*np.sin(y),np.sin(p)]);r=np.sqrt(3)/2/np.sin(np.radians(fov)/2)
    z=d;x=np.cross([0,0,1.],z);x/=np.linalg.norm(x);yv=np.cross(z,x);a=np.radians(roll);x,yv=np.cos(a)*x+np.sin(a)*yv,-np.sin(a)*x+np.cos(a)*yv
    Rw=np.stack([x,yv,z]);w2c=np.c_[Rw,-Rw@(d*r)];f=0.5/np.tan(np.radians(fov)/2);K=np.array([[f,0,0.5],[0,-f,0.5],[0,0,1.]])@np.diag([1,1,-1.]);return K@w2c,(x,yv,-d)
Xs=V[torch.randperm(len(V),generator=torch.Generator(device='cuda').manual_seed(0),device='cuda')[:150000]].double().cpu().numpy()
if os.environ.get('CAMREF_HULL','0')=='1':   # exact speed-up: projected bounding boxes are attained at convex-hull vertices
    from scipy.spatial import ConvexHull;Xs=Xs[ConvexHull(Xs).vertices]
_FNORM=os.environ.get('CAMREF_FASTNORM','0')=='1'   # opt-in exact speed-up: homogeneous coordinates once, per-column 1-D min/max (same values)
if _FNORM:Xsh=np.c_[Xs,np.ones(len(Xs))]
def normalize(P):
    h=(Xsh if _FNORM else np.c_[Xs,np.ones(len(Xs))])@P.T;uv=h[:,:2]/h[:,2:3]
    if _FNORM:lo=np.array([uv[:,0].min(),uv[:,1].min()]);hi=np.array([uv[:,0].max(),uv[:,1].max()])
    else:lo,hi=uv.min(0),uv.max(0)
    c=(lo+hi)/2;s=1/max(hi-lo)
    return np.array([[s,0,.5-s*c[0]],[0,s,.5-s*c[1]],[0,0,1]])@P
def fullP(q):
    P,ax=cam(q[0],q[1],q[2],q[3]);P=np.array([[np.exp(q[4]),0,q[5]],[0,np.exp(q[4]),q[6]],[0,0,1]])@normalize(P)
    return P*(2/(P@np.array([0,0,0,1.]))[2]),ax                       # depth 2 at the origin (inside the far plane)
fg=torch.from_numpy(g>0.0).cuda()
def sobel(x):
    kx=torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1.]],device='cuda')[None,None];x=x[None,None]
    return (F.conv2d(x,kx,padding=1)**2+F.conv2d(x,kx.transpose(2,3),padding=1)**2).sqrt()[0,0]
gi=torch.from_numpy(g).cuda();Eg=sobel(gi)
# silhouette of the input: alpha or brightness, as in fit_input_camera.py (re-derived at 512)
al=np.array(src.convert('RGBA'))[:,:,3]
if not (al==255).all():A=np.array(Image.fromarray(al).crop(tuple(int(round(b)) for b in box)).resize((S,S)))>127
else:
    from scipy.ndimage import binary_closing,binary_fill_holes,binary_opening,label
    G=np.array(src.convert('L').crop(tuple(int(round(b)) for b in box)).resize((S,S))).astype(float);A=binary_fill_holes(binary_opening(binary_closing(G>40,iterations=3),iterations=2))
    lab,n=label(A);A=lab==(np.argmax(np.bincount(lab.ravel())[1:])+1)
A=torch.from_numpy(A).cuda()
_LEAN=os.environ.get('CAMREF_LEAN','0')=='1'   # opt-in exact speed-up: same values, fewer host<->device round trips
_KX=torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1.]],device='cuda')[None,None]
_FN=raster.face_normals(V,Fc) if (_LEAN and os.environ.get('CAMREF_FNCACHE','0')=='1') else None   # opt-in exact: face normals do not depend on the camera
def sobel_lean(x):
    x=x[None,None];return (F.conv2d(x,_KX,padding=1)**2+F.conv2d(x,_KX.transpose(2,3),padding=1)**2).sqrt()[0,0]
import threading as _thr
_LSL=_thr.local();_LSC=_thr.Condition();_LSP=[];_LSA=[0]   # opt-in exact (CAMREF_LOCKSTEP): the running Powell starts (threads)
def _ls_flush():                                          # rasterize their pending cameras in ONE nvdiffrast call; each start
    import nvdiffrast.torch as _dr                        # then shades/compares its own image with the usual code. Per-image
    rb,_=_dr.rasterize(raster.ctx(),torch.cat([p_['clip'] for p_ in _LSP],0).contiguous(),_LSFi,(S,S))   # rasters are bitwise
    for i_,p_ in enumerate(_LSP):p_['tri']=rb[i_].double()[...,3].long()-1;p_['done']=True             # equal to single calls
    torch.cuda.current_stream().synchronize();_LSP.clear();_LSC.notify_all()                           # (test_rasterbatch.py)
def _tri_ls(clip):
    torch.cuda.current_stream().synchronize();p_={'clip':clip,'done':False}
    with _LSC:
        _LSP.append(p_)
        if len(_LSP)>=_LSA[0]:_ls_flush()
        else:_LSC.wait_for(lambda:p_['done'])
    t_=p_['tri'];t_.record_stream(torch.cuda.current_stream())   # allocated on the flushing thread's stream: keep its block alive until
    return t_                                                      # this stream's queued kernels are done (else reuse races -> NaN)
def render(q):
    P,ax=fullP(q)
    if _LEAN:
        AX=torch.tensor(np.stack([ax[0],ax[1],ax[2]]),device='cuda').float()
        _cl=camera.clip_coords(V,P.tolist())
        tri=_tri_ls(_cl) if getattr(_LSL,'on',False) else raster.tri_ids(raster.rasterize(_cl,Fc,S))
        if _FN is not None:mk=tri>=0;N=raster.orient(_FN[tri.clamp_min(0)],-AX[2])*mk[...,None]   # = raster.normal_map with cached face normals
        else:N,mk=raster.normal_map(V,Fc,tri,AX[2])
        Vd=-AX[2];U=AX[1];R=AX[0]
    else:
        tri=raster.tri_ids(raster.rasterize(camera.clip_coords(V,P.tolist()),Fc,S));N,mk=raster.normal_map(V,Fc,tri,ax[2])
        Vd=-torch.tensor(ax[2],device='cuda').float();U=torch.tensor(ax[1],device='cuda').float();R=torch.tensor(ax[0],device='cuda').float()
    L=Vd+.5*U-.3*R;L=L/L.norm()
    return (.08+.9*(.65*(N@L).clamp(0,1)+.35*(N@Vd).clamp(0,1)))*mk,mk
def objective(q,parts=False):
    if _LEAN:
        sh,mk=render(q);it=(mk&A).sum()/(mk|A).sum().clamp_min(1);E=sobel_lean(sh);ix=(mk|A).flatten().nonzero().squeeze(1)
        ea=Eg.flatten()[ix];eb=E.flatten()[ix];a_=ea-ea.mean();b_=eb-eb.mean();nt=(a_*b_).sum()/(a_.norm()*b_.norm()).clamp_min(1e-9)
        iou,ncc=(float(x) for x in torch.stack([it,nt]).cpu())
    else:
        sh,mk=render(q);iou=float((mk&A).sum()/(mk|A).sum().clamp_min(1));E=sobel(sh);b=(mk|A)
        a_=Eg[b]-Eg[b].mean();b_=E[b]-E[b].mean();ncc=float((a_*b_).sum()/(a_.norm()*b_.norm()).clamp_min(1e-9))
    o=(1-iou)+LAM*(1-ncc);return (o,iou,ncc) if parts else o
# recover the similarity of the silhouette fit: P_fit ~ A @ normalize(cam(..)) (least squares, up to scale)
P0=np.array(fit['P']);B=normalize(cam(fit['yaw'],fit['pitch'],fit['roll'],fit['fov'])[0]);B=B/ B[2,3]*P0[2,3]
s_=np.linalg.lstsq(np.stack([B[0],B[2]],1),P0[0],rcond=None)[0];t_=np.linalg.lstsq(np.stack([B[1],B[2]],1),P0[1],rcond=None)[0]
base=np.array([fit['yaw'],fit['pitch'],fit['roll'],float(np.clip(fit['fov'],8,70)),math.log(max(s_[0],1e-3)),s_[1],t_[1]])
cands=[]
for dyaw in (0,180):
    for ps in (1,-1):
        q=base.copy();q[0]+=dyaw;q[1]*=ps;cands.append(q)
# global grid (fit_input_camera.py only searches yaw 30-60, pitch -15..30): silhouette IoU at 128 px, top 8 join
Ac=F.interpolate(A.float()[None,None],size=(128,128),mode='area')[0,0]>.5;grid=[]
_NT=int(os.environ.get('CAMREF_THREADS','0'))   # opt-in exact speed-up: independent evaluations / Powell starts in threads,
if _NT>0:                                        # each with its own CUDA stream and nvdiffrast context (same ops per start)
    import threading,concurrent.futures as _cf;_tl=threading.local();torch.cuda.synchronize()
    def _ctx(device='cuda'):
        if not hasattr(_tl,'ctx'):_tl.ctx=__import__('nvdiffrast.torch',fromlist=['x']).RasterizeCudaContext(device=device)
        return _tl.ctx
    raster.ctx=_ctx
    def _instream(fn):
        def w(*a):
            if not hasattr(_tl,'st'):_tl.st=torch.cuda.Stream()
            with torch.cuda.stream(_tl.st):r=fn(*a);torch.cuda.current_stream().synchronize();return r
        return w
    def _pmap(fn,xs):
        with _cf.ThreadPoolExecutor(_NT) as ex:return list(ex.map(_instream(fn),xs))
def _gridq():
    for fov in (15.,30.,45.):
        for yaw in range(0,360,15):
            for pitch in range(-30,51,10):
                for ls in (0.,math.log(1.15)):yield np.array([yaw,pitch,0.,fov,ls,.5-.5*math.exp(ls),.5-.5*math.exp(ls)])
def _grid1(q):
    P,ax=fullP(q);tri=raster.tri_ids(raster.rasterize(camera.clip_coords(V,P.tolist()),Fc,128));mk=tri>=0
    return (float((mk&Ac).sum()/(mk|Ac).sum().clamp_min(1)),q)
_gf=os.environ.get('CAMREF_GRIDFILE')                # opt-in exact: the global grid is computed once and shared (JSON round-trips float64)
if _gf and os.path.exists(_gf):grid=[(g_[0],np.array(g_[1])) for g_ in json.load(open(_gf))]
else:
    _GB=int(os.environ.get('CAMREF_GRIDBATCH','0'))  # opt-in exact: rasterize the grid views in batches (one nvdiffrast call per
    if _GB>0:                                          # batch; per-view clip coordinates from the same clip_coords call as _grid1)
        import nvdiffrast.torch as _dr;_qs=list(_gridq());grid=[];_Fi=Fc.int().contiguous()
        for _b in range(0,len(_qs),_GB):
            _qb=_qs[_b:_b+_GB];_cl=torch.cat([camera.clip_coords(V,fullP(q_)[0].tolist()) for q_ in _qb],0).contiguous()
            _rb,_=_dr.rasterize(raster.ctx(),_cl,_Fi,(128,128))
            for _j,q_ in enumerate(_qb):
                mk=(_rb[_j].double()[...,3].long()-1)>=0;grid.append((float((mk&Ac).sum()/(mk|Ac).sum().clamp_min(1)),q_))
    else:grid=_pmap(_grid1,list(_gridq())) if _NT>0 else [_grid1(q) for q in _gridq()]
    if _gf:json.dump([[float(i_),[float(x) for x in q_]] for i_,q_ in grid],open(_gf,'w'))
if os.environ.get('CAMREF_GRIDONLY')=='1':sys.exit(0)
grid.sort(key=lambda x:-x[0]);print('grid top IoU',[round(x[0],3) for x in grid[:8]],[(int(x[1][0]),int(x[1][1]),int(x[1][3])) for x in grid[:8]],flush=True)
cands+=[x[1] for x in grid[:int(os.environ.get('CAMREF_NGRID','8'))]]
res=[]
_k=int(os.environ.get('CAMREF_TOPK','0'))           # opt-in: optimise only the k starts with the lowest initial objective
if _k>0:cands=sorted(cands,key=lambda q:objective(q))[:_k]
_only=os.environ.get('CAMREF_ONLY');_resd=os.environ.get('CAMREF_RESULTS')   # exact parallel mode: one start per process, then combine
if _only is not None:cands=[cands[int(_only)]]
if _resd:
    for _i in range(len(cands)):
        _d=json.load(open(f'{_resd}/part{_i}.json'));res.append((_d['o'],np.array(_d['q']),tuple(_d['o1'])))
    cands=[]
def _start(q):
    if os.environ.get('CAMREF_MEMO','0')=='1':      # opt-in exact: the objective is deterministic, so repeated parameter vectors
        memo={}                                       # (Powell re-evaluations, the initial and final points) reuse the stored value
        def obj(x,parts=False):
            k=np.asarray(x,np.float64).tobytes()
            if k not in memo:memo[k]=objective(x,True)
            return memo[k] if parts else memo[k][0]
    else:obj=objective
    o0=obj(q,True);r=minimize(obj,q,method='Powell',options=dict(xtol=1e-3,ftol=1e-5,maxfev=int(os.environ.get('CAMREF_MAXFEV','1200'))));o1=obj(r.x,True)
    return o0,r,o1
_SPL=os.environ.get('CAMREF_SPLIT')   # opt-in exact: 'k/n' = this process optimises the starts with index % n == k; k>0 writes
_ALLC=cands                             # CAMREF_SPLITDIR/part{i}.json, k=0 waits for them and selects over all starts in the
if _SPL and _only is None and not _resd:   # original order (JSON round-trips float64; per-start results do not depend on grouping)
    _sk,_sn=(int(x) for x in _SPL.split('/'));_SD=os.environ['CAMREF_SPLITDIR'];os.makedirs(_SD,exist_ok=True)
    _MINE=[i for i in range(len(_ALLC)) if i%_sn==_sk];cands=[_ALLC[i] for i in _MINE]
else:_SPL=None
_LSON=os.environ.get('CAMREF_LOCKSTEP','0')=='1' and _NT>=len(cands)>0 and _only is None
if _LSON:
    _LSFi=Fc.int().contiguous();_LSA[0]=len(cands)
    def _start_ls(q):
        _LSL.on=True
        try:return _start(q)
        finally:
            _LSL.on=False
            with _LSC:
                _LSA[0]-=1
                if _LSP and len(_LSP)>=_LSA[0]:_ls_flush()
_done=_pmap(_start_ls if _LSON else _start,cands) if (_NT>0 and _only is None) else None
if _SPL:
    if _done is None:_done=[_start(q) for q in cands]
    _pk=lambda o0,r,o1:dict(o0=[float(v) for v in o0],x=[float(v) for v in r.x],o1=[float(v) for v in o1])
    if _sk>0:
        for _i,_d in zip(_MINE,_done):
            json.dump(_pk(*_d),open(f'{_SD}/part{_i}.tmp','w'));os.replace(f'{_SD}/part{_i}.tmp',f'{_SD}/part{_i}.json')
        sys.exit(0)
    from types import SimpleNamespace as _NS2;import time as _tm;_got=dict(zip(_MINE,_done))
    for _i in range(len(_ALLC)):
        if _i in _got:continue
        while not os.path.exists(f'{_SD}/part{_i}.json'):_tm.sleep(.01)
        _d=json.load(open(f'{_SD}/part{_i}.json'));_got[_i]=(tuple(_d['o0']),_NS2(x=np.array(_d['x'])),tuple(_d['o1']))
    cands=_ALLC;_done=[_got[_i] for _i in range(len(_ALLC))]
for _j,q in enumerate(cands):
    o0,r,o1=_done[_j] if _done is not None else _start(q)
    if _only is not None:
        json.dump(dict(o=o1[0],q=[float(x) for x in r.x],o1=[float(x) for x in o1]),open(f"{os.environ['CAMREF_PARTDIR']}/part{_only}.json",'w'));sys.exit(0)
    res.append((o1[0],r.x,o1));print('start yaw %.0f pitch %.0f: obj %.3f (IoU %.3f NCC %.3f) -> %.3f (IoU %.3f NCC %.3f)'%(q[0],q[1],*o0,*o1),flush=True)
if os.environ.get('CAMREF_DUMP'):json.dump([dict(o1=[float(v) for v in x[2]],q=[float(v) for v in x[1]]) for x in res],open(os.environ['CAMREF_DUMP'],'w'))
_fl=float(os.environ.get('CAM_IOU_FLOOR','0'))   # opt-in: ignore degenerate optima (low IoU, field of view at the clamp)
_rel=float(os.environ.get('CAM_IOU_REL','0'))   # opt-in: only candidates within _rel of the best silhouette IoU
if _rel>0:_fl=max(_fl,max(x[2][1] for x in res)-_rel)
_ok=[x for x in res if x[2][1]>=_fl and 8.<x[1][3]<70.] if _fl>0 else res
if os.environ.get('CAM_PITCH_MIN') is not None:   # opt-in: viewpoint side known (e.g. a rendered input seen from above); a near-symmetric
    _pm=float(os.environ['CAM_PITCH_MIN']);_ok=[x for x in (_ok or res) if x[1][1]>=_pm]   # outline otherwise lets the mirrored view win
best=min(_ok or res,key=lambda x:x[0]);q=best[1];P,ax=fullP(q)
json.dump(dict(P=P.tolist(),r=list(ax[0]),u=list(ax[1]),fwd=list(ax[2]),yaw=q[0],pitch=q[1],roll=q[2],fov=float(np.clip(q[3],8,70)),iou=best[2][1],ncc=best[2][2],
               box=fit['box'],size=fit['size']),open(OJ,'w'),indent=1)
sh,mk=render(q);ov=np.stack([g,sh.cpu().numpy(),g],-1);Image.fromarray((np.concatenate([np.stack([g]*3,-1),ov],1)*255).clip(0,255).astype(np.uint8)).save(OJ.replace('.json','.png'))
print('best: yaw %.1f pitch %.1f roll %.1f fov %.1f  IoU %.3f NCC %.3f'%(q[0],q[1],q[2],q[3],best[2][1],best[2][2]))
