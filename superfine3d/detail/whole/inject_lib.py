"""Shared implementation of one image-detail injection (inject.py CLI and the single-process pipeline_v3.py driver).
cfg: dict of the inject.py environment keys (strings), e.g. {'UNIT': '1/2048', 'MAXD': '3', 'K': '64', 'GAIN': '1.1'}."""
from os import environ as _E   # install locations, set by setup/env.sh (SF3D_ROOT, SF3D_CKPT, ...)
import os,sys,time,json,numpy as np,torch
sys.path.insert(0,f'{_E["SF3D_ROOT"]}/superfine3d/detail')
from hfdetail import pipeline,heightfield as hf,estimator
CKPT=f'{_E["SF3D_CKPT"]}/estimator/C_cond/ckpt_04000.pt'
def parse(cfg):
    E_=lambda k,d:type(d)(cfg.get(k,d));n,d=(cfg.get('UNIT','1/4096')+'/1').split('/')[:2]
    o=lambda k,f:f(cfg[k]) if cfg.get(k) else None
    return dict(CK=cfg.get('CKPT',CKPT),UNIT=float(n)/float(d),MAXD=E_('MAXD',3.0),MODE=cfg.get('MODE','resid'),K=E_('K',32),GAIN=E_('GAIN',.9),
                COND=E_('COND',1),IT=E_('ITERS',800),LAP=E_('LAP',.01),OUTL=o('OUTLIER',float),ORING=E_('ORINGS',1),SLOPE=o('SLOPE',float),
                FK=o('FK',int),FG=E_('FG',.9),SUB=E_('SUBDIV',0),RES=o('RES',int),BR=o('BR',int),AMAX=E_('AMAX',2.0),ALIGN=E_('ALIGN',0),ASIG=E_('ALIGN_SIGMA',6.0),AMAXS=E_('ALIGN_MAX',96.0),LAPH=o('LAPH',float),EDGE=o('EDGE',float),LEVELS=E_('LEVELS',1),DEPTH=E_('DEPTH',0),DEPTHW=E_('DEPTHW',1.0),DEPTHF=E_('DEPTHF',32),ORIENT=E_('ORIENT',0),FOLD=o('FOLD',float),
                AGREE=o('AGREE',float),GRAZE=o('GRAZE',float),CONF=cfg.get('CONF'),GRAZE_W=E_('GRAZE_W',0.3),GRAZE_R=E_('GRAZE_R',8),RGB_FK=o('RGB_FK',int),TTA=E_('TTA',0),ROUTE=E_('ROUTE',0),RGB_THR=E_('RGB_THR',1.0),RGB_GAIN=E_('RGB_GAIN',1.1),RGB_KPX=o('RGB_KPX',float),MIN_PPV=E_('MIN_PPV',0.5),RGB_CK=cfg.get('RGB_CKPT',CKPT))
def chroma(img):
    """Mean |channel - grey| (0-255) over the foreground; 0 for grey-model renders."""
    a=img.astype(np.float32);fg=a.max(-1)>5
    return float(np.abs(a-a.mean(-1,keepdims=True))[fg].mean()) if fg.any() else 0.
def ppv(v,cam,unit,W):
    """Image pixels spanned by one voxel (size unit) at the mesh centre, along the camera's right axis."""
    P=np.array(cam['P'],dtype=np.float64);c=v.double().mean(0).cpu().numpy();r=np.array(cam['r'],dtype=np.float64)
    pr=lambda X:(P[:2]@np.append(X,1))/(P[2]@np.append(X,1))*W
    return float(np.linalg.norm(pr(c+unit*r)-pr(c)))
def route(c,img,st):
    """ROUTE=1: RGB (textured) inputs use the mesh-conditioned estimator (C_cond, COND=1), no fine pass, gain <= RGB_GAIN;
    RGB_KPX: residual band K scaled to the image size (K * width / RGB_KPX, min 8). Grey inputs keep the given config."""
    ch=chroma(img);st['chroma']=ch;st['route']='grey'
    if not c['ROUTE'] or ch<c['RGB_THR']:return c
    c=dict(c,CK=c['RGB_CK'],COND=1,FK=c['RGB_FK'],GAIN=min(c['GAIN'],c['RGB_GAIN']));st['route']='rgb'   # RGB_FK (opt-in): keep the fine band (gain FG)
    if c['RGB_KPX']:c['K']=max(8,int(round(c['K']*img.shape[0]/c['RGB_KPX'])))
    st['K']=c['K'];return c
def gated(c,v,cam,img,st):
    """ROUTE=1: True when the image is coarser than MIN_PPV pixels per voxel at this level (injection would only add
    estimator noise: the mesh already resolves more than the image)."""
    if not c['ROUTE']:return False
    st['ppv']=ppv(v,cam,c['UNIT'],img.shape[0]);return st['ppv']<c['MIN_PPV']
def load_image(path):
    from PIL import Image
    rgba=np.array(Image.open(path).convert('RGBA')).astype(np.float32)/255;return (rgba[...,:3]*rgba[...,3:4]*255).clip(0,255).astype(np.uint8)
def load_cam(path):
    cam=json.load(open(path));return dict(P=cam['P'],r=cam['r'],u=cam['u'],fwd=cam['fwd'])
_MODELS={}
_DEPTH={}
def moge_depth(img):
    """MoGe-2 (ViT-L) depth on the input image grid (image only; NaN outside its mask); cached per image."""
    key=(img.shape,int(img[::97,::97].sum()))
    if key not in _DEPTH:
        sys.path.insert(0,f'{_E["SF3D_UTILS3D"]}');sys.path.insert(0,f'{_E["SF3D_MOGE"]}')
        from moge.model.v2 import MoGeModel
        if 'moge' not in _MODELS:_MODELS['moge']=MoGeModel.from_pretrained(f'{_E["SF3D_MOGE_WEIGHTS"]}/model.pt').cuda().eval()
        x=torch.from_numpy(np.ascontiguousarray(img)).float().permute(2,0,1).cuda()/255.
        with torch.no_grad():o=_MODELS['moge'].infer(x,resolution_level=9)
        d=o['depth'].float();_DEPTH.clear();_DEPTH[key]=torch.where(o['mask'].bool()&torch.isfinite(d),d,torch.tensor(float('nan'),device='cuda'))
    return _DEPTH[key]
def model(ck):
    if ck not in _MODELS:_MODELS[ck]=estimator.NiRNE(ck)
    return _MODELS[ck]
_EST_ORIG=estimator.estimate
def _est_cached(image,cam,res,model,**kw):
    """Opt-in (env EST_CACHE=file.npz): the image normal estimate is computed once and reused by later injections of the same
    image / camera / resolution (all levels, also across processes). With a mesh-conditioned estimator the conditioning map is the
    one of the first call (the coarsest mesh)."""
    import hashlib
    if os.environ.get('EST_FLATCOND')=='1' and kw.get('cond') is not None:      # opt-in: mesh-independent estimate (conditioning map = all pixels facing the camera,
        fc=torch.zeros_like(kw['cond']);fc[...,2]=1;kw=dict(kw,cond=fc)           # as the condition-dropout case in training) -> identical at every level, cacheable exactly
    path=os.environ.get('EST_CACHE')
    if not path:return _EST_ORIG(image,cam,res,model,**kw)
    grp=''                                                            # EST_CACHE_LEVELS='512,2048': re-estimate at these levels (each estimate is reused until the next listed level)
    if os.environ.get('EST_CACHE_LEVELS'):
        lv=sorted(int(x) for x in os.environ['EST_CACHE_LEVELS'].split(','));cur=_EST_LEVEL[0];grp='_L%d'%max([l for l in lv if l<=cur] or [lv[0]])
    key=hashlib.sha256(image[::7,::7].tobytes()+np.asarray(cam['P'],dtype=np.float64).tobytes()+str((res,sorted((k,str(x)) for k,x in kw.items() if k!='cond'))).encode()).hexdigest()[:16]
    if os.environ.get('EST_COND_PER_LEVEL')=='1' and kw.get('cond') is not None:grp+='_C%d'%_EST_LEVEL[0]   # mesh-conditioned: per level
    f=path.replace('.npz','')+'_'+key+grp+'.npz'
    if os.path.exists(f):
        a=np.load(f);return torch.from_numpy(a['E']).cuda(),torch.from_numpy(a['m']).cuda()
    E,Em=_EST_ORIG(image,cam,res,model,**kw);np.savez(f,E=E.cpu().numpy().astype(np.float32 if os.environ.get('EST_CACHE_F32')=='1' else np.float16),m=Em.cpu().numpy());return E,Em   # EST_CACHE_F32=1: exact reuse
estimator.estimate=_est_cached
_EST_LEVEL=[0]
_CC={}
_AOTI={}
def _aoti(m):
    """AOTInductor packages of the re-estimation networks (setup/build_estimator_aot.sh), used when EST_AOTI points to them and
    they were built on this GPU / torch version for this estimator checkpoint; otherwise None (eager networks)."""
    d=os.environ.get('EST_AOTI')
    if not d or getattr(m,'ckpt',None) is None:return None
    k=(d,m.ckpt)
    if k not in _AOTI:
        meta=json.load(open(f'{d}/meta.json'));ok=(meta.get('gpu')==torch.cuda.get_device_name() and meta.get('torch')==torch.__version__
            and meta.get('unet_ckpt')==estimator.ckpt_fingerprint(m.ckpt))
        _AOTI[k]={n:torch._inductor.aoti_load_package(f'{d}/{n}.pt2') for n in ('enc','unet','dec')} if ok else None
        if not ok:print('estimator AOT packages not used (built for another GPU / torch / checkpoint):',d,flush=True)
    return _AOTI[k]
_NIRNE_CALL=estimator.NiRNE.__call__
@torch.no_grad()
def _nirne_call_cc(self,tiles,conds=None):
    """Opt-in exact speed-up (env EST_CTRL_CACHE=1): NiRNE.__call__ with the image-only part of the network reused.
    Preprocessing, the image VAE encode (prepare_latents) and the ControlNet depend only on the image tiles of the batch, not on
    the mesh conditioning, so their results are stored per batch (key = the tiles' pixels) and reused by later injections of the
    same image; the UNet, the conditioning encode and the decode run as before with the same inputs."""
    if os.environ.get('EST_CTRL_CACHE')!='1' or conds is None:return _NIRNE_CALL(self,tiles,conds)
    import hashlib
    p=self.pipe;dev,dt=p._execution_device,p.dtype
    key=hashlib.blake2b(b''.join(t.tobytes() for t in tiles),digest_size=16).digest()
    ent=_CC.get(key)
    if ent is None:
        img,padding,orig=p.image_processor.preprocess(tiles,p.default_processing_resolution,'bilinear',dev,dt)
        lat,_pred,_=p.prepare_latents(img,None,None,1,len(tiles))
        emb=torch.zeros(len(tiles),257,1024,device=dev,dtype=dt)
        dn,mid=p.controlnet(lat.detach(),p.t_start,encoder_hidden_states=emb,conditioning_scale=1.0,guess_mode=False,return_dict=False)
        ent=(padding,orig,tuple(dn),mid);_CC[key]=ent
    padding,orig,dn,mid=ent
    c=torch.stack([torch.as_tensor(x) for x in conds]).to(dev,dt)
    A=_aoti(self)
    if A is not None and len(tiles)<=8:                               # AOTInductor networks (static batch 8: partial batches padded, extra outputs dropped)
        n=len(tiles);sc=p.vae.config.scaling_factor;pad=lambda u:torch.cat([u,u[:1].expand(8-n,*u.shape[1:])]).contiguous() if n<8 else u.contiguous()
        one=lambda r:r[0] if isinstance(r,(list,tuple)) else r
        pred=one(A['enc'](pad(c)))*sc;emb8=torch.zeros(8,257,1024,device=dev,dtype=dt)
        out=one(A['unet'](pred,emb8,pad(mid),*[pad(d) for d in dn]))
        y=one(A['dec'](out/sc))[:n]
        if not bool(torch.isfinite(y).all()):A=None                    # non-finite output (fp16 overflow): this batch with the eager networks
        else:y=p.normalize_normals(y);y=p.image_processor.unpad_image(y,padding)
    if A is None or len(tiles)>8:
        pred=p.vae.encode(c).latent_dist.mode()*p.vae.config.scaling_factor
        emb=torch.zeros(len(tiles),257,1024,device=dev,dtype=dt)
        out=p.unet(pred,p.t_start,encoder_hidden_states=emb,down_block_additional_residuals=tuple(dn),mid_block_additional_residual=mid,return_dict=False)[0]
        y=p.decode_prediction(out);y=p.image_processor.unpad_image(y,padding)
    y=p.image_processor.resize_antialias(y,orig,'bilinear',is_aa=False);y=p.normalize_normals(y)
    y=p.image_processor.pt_to_numpy(y)
    return [y[i] for i in range(len(tiles))]
estimator.NiRNE.__call__=_nirne_call_cc
def inject(v,F,img,cam,cfg,keep_model=True):
    """v (V,3) float32 cuda, F (M,3) long cuda -> (out vertices, F, stats); identical calls to the former inject.py."""
    try:_EST_LEVEL[0]=int(str(cfg.get('UNIT','1/1024')).split('/')[-1])
    except Exception:_EST_LEVEL[0]=0
    c=parse(cfg);st={};c=route(c,img,st)
    from hfdetail import targets as _tg                                # opt-in evidence gate (cfg SHRINK=deg, SHRINK_R=px); unset = unchanged
    _tg.SHRINK=(float(cfg['SHRINK']),int(cfg.get('SHRINK_R',2))) if cfg.get('SHRINK') else None;_tg.SHRINK_STATS.clear()
    from hfdetail import pipeline as _pl;_pl.AUTO_MIN_PASS=float(cfg['AUTO']) if cfg.get('AUTO') else None   # opt-in evidence-adaptive levels (needs SHRINK)
    _pl.MESH_ALIGN=dict(zip(('sigma','max_shift','ds'),(float(x) for x in cfg['MALIGN'].split(':')))) if cfg.get('MALIGN') else None   # opt-in: move the mesh onto the image
    _pl.EVMASK=int(cfg['EVMASK']) if cfg.get('EVMASK') else None;_tg.KEEP_S=_pl.EVMASK is not None;_tg.SHRINK_S[0]=None   # opt-in evidence-masked displacement
    if gated(c,v,cam,img,st):st.update(seconds=0.,moved=0.,skipped='image coarser than mesh');return v,F,st
    if c['SUB']:
        from hfdetail import subdivide;v,F=subdivide.midpoint(v,F)
    RES=c['RES'] or img.shape[0];t=time.time()
    m=model(c['CK']) if keep_model else estimator.NiRNE(c['CK'])
    out=pipeline.refine(v,F,hf.Frame((0.,0.,0.),1.,c['UNIT']),cam,image=img,model=m,res=RES,mode=c['MODE'],k=c['K'],gain=c['GAIN'],
                        params=hf.HFParams(iters=c['IT'],maxd=c['MAXD'],lap=c['LAP'],slope=c['SLOPE'],lap_huber=c['LAPH'],levels=c['LEVELS'],depth_f=c['DEPTHF'],orient_init=bool(c['ORIENT']),fold_tau=c['FOLD'],lp_w=(float(cfg['LPW']) if cfg.get('LPW') else None),lp_k=int(cfg.get('LPK',64))),est_kw=dict(batch=8,**(dict(shifts=(0,384),flip=True) if c['TTA'] else {})),stats=st,condition=bool(c['COND']),tol_obj=(float(cfg['TOL'])/1024 if cfg.get('TOL') else 2/1024),
                        outlier_deg=c['OUTL'],outlier_rings=c['ORING'],fine_k=c['FK'],fine_gain=c['FG'] if c['FK'] else None,blend_r=c['BR'],blend_amax=c['AMAX'],align=bool(c['ALIGN']),align_kw=dict(sigma=c['ASIG'],max_shift=c['AMAXS']),edge_tau=c['EDGE'],
                        depth=moge_depth(img) if c['DEPTH'] else None,depth_w=c['DEPTHW'],agree_deg=c['AGREE'],graze=(c['GRAZE'],c['GRAZE_W'],int(c['GRAZE_R'])) if c['GRAZE'] else None,conf=tuple(float(x) for x in c['CONF'].split(',')) if c['CONF'] else None)
    (st.update({'shrink_'+k:x for k,x in _tg.SHRINK_STATS.items()}) if _tg.SHRINK is not None else None)
    (st.update({'fold_'+k:x for k,x in hf.FOLD_STATS.items()}) if c['FOLD'] is not None else None);st['seconds']=time.time()-t;st['moved']=float(((out-v).abs().sum(1)>0).float().mean());return out,F,st
def write_ply(path,v,F):
    vo=v.cpu().numpy().astype(np.float32);fo=F.cpu().numpy().astype(np.int32)
    with open(path,'wb') as fh:                                       # binary little-endian ply (vertices float x,y,z; faces uchar+int)
        fh.write(f'ply\nformat binary_little_endian 1.0\nelement vertex {len(vo)}\nproperty float x\nproperty float y\nproperty float z\nelement face {len(fo)}\nproperty list uchar int vertex_indices\nend_header\n'.encode())
        fh.write(vo.tobytes());fa=np.zeros(len(fo),dtype=[('n','u1'),('i','<i4',(3,))]);fa['n']=3;fa['i']=fo;fh.write(fa.tobytes())
src_=open(f'{_E["SF3D_ROOT"]}/superfine3d/detail/whole/whole_run.py').read();exec(src_[src_.index('def read_ply'):src_.index("t=time.time();v,Fn=read_ply")])
def read_mesh(path):
    if path.endswith('.npz'):
        with np.load(path) as a:return a['v'],a['f'].astype(np.int64)
    return read_ply(path)
def fmt(st):return json.dumps({k:round(x,2) if isinstance(x,float) else x for k,x in st.items() if k not in ('seconds','moved')})
_TP_ORIG=estimator.tiled_predict
def _tiled_predict_gpu(pil, predict, up=3., ts=768, batch=1, cond=None, offset=0, fg=None):
    """Opt-in exact speed-up (env EST_GPU_ACC=1): estimator.tiled_predict with the full-resolution blending (tent-weighted
    accumulation, empty-pixel fill, division and renormalisation) done on the GPU instead of in numpy. The same float32
    operations are applied in the same order (elementwise multiply/add, division, x*x summed a0+a1+a2, IEEE sqrt), so
    the result is bit-identical; the upsampling, tiling, network calls and the final area downsample are unchanged.
    The conditioning map is renormalised on the GPU as well (same float32 ops, bit-identical)."""
    if os.environ.get('EST_GPU_ACC')!='1':return _TP_ORIG(pil,predict,up,ts,batch,cond=cond,offset=offset,fg=fg)
    from PIL import Image as _I;import torch.nn.functional as F_
    W0=pil.size[0];assert pil.size[1]==W0,'square images only'
    W=int(round(W0*up));big=pil.resize((W,W),_I.Resampling.LANCZOS);pos=estimator.tile_positions(W,ts,offset)
    if cond is not None:
        assert cond.shape[:2]==(W0,W0),cond.shape
        cb=F_.interpolate(torch.as_tensor(cond).float().permute(2,0,1)[None],size=(W,W),mode='bilinear',align_corners=False)[0]
        cb=cb.cuda();cb=cb/cb.norm(dim=0,keepdim=True).clamp_min(1e-6)  # bilinear stays on the CPU (the GPU kernel rounds differently); the strided
                                                                         # CPU norm took 15 s at 8856 px, the GPU norm is bit-identical (eff_opt/r1_exact.log)
    dev=torch.device('cuda');acc=torch.zeros((W,W,3),dtype=torch.float32,device=dev);wsum=torch.zeros((W,W,1),dtype=torch.float32,device=dev)
    win=torch.from_numpy(estimator.tent_window(ts)).to(dev)
    xy=[(y,x) for y in pos for x in pos]
    if fg is not None:
        fg=np.asarray(fg,bool);assert fg.shape==(W0,W0),fg.shape
        def has_fg(y,x):
            a,b=max(int(np.floor(y/up))-1,0),min(int(np.ceil((y+ts)/up))+1,W0)
            c,d=max(int(np.floor(x/up))-1,0),min(int(np.ceil((x+ts)/up))+1,W0)
            return bool(fg[a:b,c:d].any())
        n=len(xy);sizes=[batch if i<n-n%batch else n%batch for i in range(n)];groups={}
        for p_,sz in zip(xy,sizes):
            if has_fg(*p_):groups.setdefault(sz,[]).append(p_)
        plan=[]
        for sz,ps in groups.items():
            for i in range(0,len(ps),sz):
                ch=ps[i:i+sz];plan.append((ch,ch+[ch[0]]*(sz-len(ch))))
    else:plan=[(xy[i:i+batch],xy[i:i+batch]) for i in range(0,len(xy),batch)]
    for keep,ch in plan:
        tiles=[big.crop((x,y,x+ts,y+ts)) for y,x in ch]
        preds=predict(tiles) if cond is None else predict(tiles,[cb[:,y:y+ts,x:x+ts] for y,x in ch])
        for (y,x),p_ in zip(keep,preds[:len(keep)]):
            pt=torch.from_numpy(np.ascontiguousarray(p_)).to(dev);acc[y:y+ts,x:x+ts]+=pt*win;wsum[y:y+ts,x:x+ts]+=win
    if fg is not None:
        empty=wsum[...,0]==0;acc[empty]=torch.tensor((0.,0.,1.),device=dev);wsum[empty]=1.
    n=acc/wsum;del acc,wsum;s=n*n;r=s[...,0]+s[...,1];r=r+s[...,2];del s;r=torch.sqrt(r).clamp_min(1e-6)[...,None];n=n/r;del r
    n=n.cpu()
    return F_.interpolate(n.permute(2,0,1)[None],size=(W0,W0),mode='area')[0].permute(1,2,0).numpy()
estimator.tiled_predict=_tiled_predict_gpu
