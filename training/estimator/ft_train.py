"""Fine-tune NiRNE (Hi3DGen yoso-normal-v1-8-1: one-step image->normal, VAE + controlnet + UNet) into a fine-detail
normal estimator for zoomed-in crops, used exactly like the tiled inference that already helped:
  sample a region of TS/U px from a grey crop view (U ~ log-uniform [UMIN,UMAX]), upsample it to TS (768, the model's
  native processing size), predict, area-downsample the prediction back to the region size, compare with the true
  normals of that region (full-mesh render at 1024, hires/z1 + hires/train views; assets disjoint from the test set).
Loss on hit pixels:  (1-cos)  +  LHP * sum_k mean |hp_k(pred) - hp_k(target)|^2   (k = 4, 8 px masked box high-pass)
Frame: NiRNE camera convention on these cameras is n_obj = -x r + y u + z (-fwd)  (sign_search.py).
Trainable: UNet + controlnet (fp32 master weights, bf16 autocast); VAE frozen (gradients pass through its decoder).
Env: COND (0/1) mesh-conditioned: the UNet starts from the VAE latent of a coarse normal map (LR 512 mesh render,
  ft_nirne/cond_lr, or with prob COND_LP a random 8/16/32 px low-pass of the GT normals) instead of the image latent;
  the image still drives the controlnet. TEX p (0: share of samples from the textured+lit Blender views ft_nirne/tex/<cid>/view_l{0,1}.png) INIT ckpt  STEPS (4000) BS (4) LR (1e-5) LHP (1.0) UMIN (2) UMAX (4) TS (768) TRAIN (unet,controlnet) OUT dir
usage: ft_train.py OUT_DIR"""
from os import environ as _E   # install locations, set by setup/env.sh (SF3D_ROOT, SF3D_CKPT, ...)
import sys,os,json,math,random,time,numpy as np,torch,torch.nn.functional as F
from PIL import Image
import diffusers.models.controlnets.controlnet as _cn;sys.modules.setdefault('diffusers.models.controlnet',_cn)
sys.path.insert(0,f'{_E["SF3D_STABLENORMAL"]}');from stablenormal.pipeline_yoso_normal import YOSONormalsPipeline
D=f'{_E["SF3D_DATA"]}/estimator';OUT=sys.argv[1];os.makedirs(OUT,exist_ok=True)
E=lambda k,d:type(d)(os.environ.get(k,d))
STEPS=E('STEPS',4000);BS=E('BS',4);LR=E('LR',1e-5);LHP=E('LHP',1.0);UMIN=E('UMIN',2.0);UMAX=E('UMAX',4.0);TS=E('TS',768)
TRAIN=os.environ.get('TRAIN','unet,controlnet').split(',');TEX=E('TEX',0.0);INIT=os.environ.get('INIT','');COND=E('COND',0);COND_LP=E('COND_LP',0.5);SEED=E('SEED',0)
CSHIFT=E('CSHIFT',0.0);CROT=E('CROT',0.0);CSCALE=E('CSCALE',0.0);CWARP=E('CWARP',0.0);CDROP=E('CDROP',0.0);CLPS=[int(k) for k in os.environ.get('CLPS','8,16,32').split(',')]
PHOTO=E('PHOTO',0.0);ALB=E('ALB',0.0);EXPO=E('EXPO',0.0);SAVE=os.environ.get('SAVE','every')
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
# ---------------- data
items=[];SPLITS=os.environ.get('SPLITS','z1,train').split(',');W_V2=E('W_V2',1.0);HP4_MIN=E('HP4_MIN',0.0)
KEEP_V2=set(open(f'{D}/ft_nirne/v2/keep.txt').read().split()) if 'v2' in SPLITS else set()
HPST=json.load(open(f'{D}/ft_nirne/v2/stats_all.json')) if HP4_MIN>0 else {}
for s in SPLITS:
    for c in sorted(os.listdir(f'{D}/hires/{s}')):
        if s=='v2' and c not in KEEP_V2:continue
        if HP4_MIN>0 and s!='v2' and HPST.get(c,1.)<HP4_MIN:continue        # drop near-flat crops of the old set
        if os.path.exists(f'{D}/ft_nirne/targets/{c}.npz') and (TEX==0 or os.path.exists(f'{D}/ft_nirne/tex/{c}/view_l1.png')) and (not COND or s=='v2' or os.path.exists(f'{D}/ft_nirne/cond_lr/{c}.npz')):items.append((c.split('_')[0],c,f'{D}/hires/{s}/{c}'))
shas=sorted({x[0] for x in items});rng=random.Random(123);val_sh=set(rng.sample([s for s in shas if any(x[0]==s and '/z1/' in x[2] for x in items)],30))
v2sh=sorted({x[0] for x in items if '/hires/v2/' in x[2]});val_sh|=set(rng.sample(v2sh,min(12,len(v2sh)//8)))   # held-out v2 (relief) assets
tr=[x for x in items if x[0] not in val_sh];va=[x for x in items if x[0] in val_sh]
json.dump(dict(train=[x[1] for x in tr],val=[x[1] for x in va]),open(f'{OUT}/split.json','w'))
TRW=[W_V2 if '/hires/v2/' in x[2] else 1.0 for x in tr]
print('views train',len(tr),'val',len(va),'assets',len(shas),flush=True)
cache={};icache={}
def image(it,var):                                                    # var: 'g' grey view, 'l0'/'l1' textured+lit (premultiplied on black)
    key=(it[1],var)
    if key in icache:return icache[key]
    if var=='g':im=Image.open(it[2]+'/view.png').convert('RGB')
    else:
        a=np.array(Image.open(f'{D}/ft_nirne/tex/{it[1]}/view_{var}.png').convert('RGBA')).astype(np.float32)/255
        im=Image.fromarray((a[...,:3]*a[...,3:4]*255).clip(0,255).astype(np.uint8))
    if len(icache)<4500:icache[key]=im
    return im
def load(it):
    if it[1] in cache:return cache[it[1]]
    im=None;cam=json.load(open(it[2]+'/cam.json'))
    with np.load(f'{D}/ft_nirne/targets/{it[1]}.npz') as a:n=a['n'].astype(np.float32);m=a['m']
    R,U,B=np.array(cam['r']),np.array(cam['u']),-np.array(cam['fwd'])
    t=np.stack([-(n@R),n@U,n@B],-1)*m[...,None]                     # NiRNE camera frame
    if COND:
        if os.path.exists(f'{D}/ft_nirne/cond_lr/{it[1]}.npz'):
            with np.load(f'{D}/ft_nirne/cond_lr/{it[1]}.npz') as a:cn=a['n'].astype(np.float32);cm=a['m']
            cc=np.stack([-(cn@R),cn@U,cn@B],-1);cc[~cm]=(0,0,1);t=(t,cc.astype(np.float16))
        else:t=(t,None)                                                # no LR latent (v2 assets): low-pass conds only       # uncovered: facing the camera
    r=(im,t,m)
    if len(cache)<3000:cache[it[1]]=r
    return r
def cond_map(t,m,y,x,S,src,k):
    if src=='lr':c=torch.from_numpy(t[1][y:y+S,x:x+S].astype(np.float32)).permute(2,0,1)
    else:                                                             # GT low-pass (masked box, k px), uncovered -> camera-facing
        g=torch.from_numpy(t[0].astype(np.float32)).permute(2,0,1)[:,max(0,y-k):y+S+k,max(0,x-k):x+S+k];mm=torch.from_numpy(m)[max(0,y-k):y+S+k,max(0,x-k):x+S+k].float()
        num=F.avg_pool2d((g*mm)[None],k+1,1,k//2,count_include_pad=False)[0];den=F.avg_pool2d(mm[None,None],k+1,1,k//2,count_include_pad=False)[0]
        c=num/den.clamp_min(1e-6);c=torch.where(den>0,c,torch.tensor([0.,0,1])[:,None,None]);oy,ox=y-max(0,y-k),x-max(0,x-k);c=c[:,oy:oy+S,ox:ox+S]
    c=F.interpolate(c[None],size=(TS,TS),mode='bilinear',align_corners=False)[0];return c/c.norm(dim=0,keepdim=True).clamp_min(1e-6)
def misalign(c,rs):
    """Cond corruption (test-time base meshes are generated + camera-fitted, not aligned with the image): random
    shift / rotation / scale (fractions of the tile) + smooth elastic warp; border values extend. c (3,TS,TS)."""
    if not (CSHIFT or CROT or CSCALE or CWARP):return c
    a=math.radians(rs.uniform(-CROT,CROT));sc=1+rs.uniform(-CSCALE,CSCALE);tx,ty=rs.uniform(-CSHIFT,CSHIFT)*2,rs.uniform(-CSHIFT,CSHIFT)*2
    th=torch.tensor([[sc*math.cos(a),-sc*math.sin(a),tx],[sc*math.sin(a),sc*math.cos(a),ty]],dtype=torch.float32)[None]
    g=F.affine_grid(th,(1,3,TS,TS),align_corners=False)
    if CWARP:
        n=rs.choice([3,4,6]);gen=torch.Generator().manual_seed(rs.randrange(2**31))
        w=torch.randn(1,2,n,n,generator=gen)*CWARP*2;g=g+F.interpolate(w,size=(TS,TS),mode='bicubic',align_corners=True).permute(0,2,3,1)
    c=F.grid_sample(c[None],g,mode='bilinear',padding_mode='border',align_corners=False)[0];return c/c.norm(dim=0,keepdim=True).clamp_min(1e-6)
def photo(img,rs,tex):
    """Photometric augmentation (3,TS,TS) in [0,1]: gamma / contrast / colour gain, and (ALB) a smooth random albedo field
    multiplied in, so colour and brightness patches stop implying geometry."""
    if EXPO and tex and rs.random()<EXPO:                             # auto-exposure: dark scan renders -> foreground p95 in [.6,.95]
        L=img.mean(0);fg=L>0.004
        if fg.sum()>100:img=img*(rs.uniform(.6,.95)/torch.quantile(L[fg],.95).clamp_min(1e-3))
    if PHOTO and rs.random()<PHOTO:
        img=img.clamp(0,1)**rs.uniform(.7,1.4);img=(img-img.mean())*rs.uniform(.7,1.3)+img.mean()
        img=img*torch.tensor([rs.uniform(.8,1.2) for _ in range(3)])[:,None,None]
    if ALB and rs.random()<ALB:
        gen=torch.Generator().manual_seed(rs.randrange(2**31));n=rs.choice([2,4,8,16,32])
        f=torch.rand(1,3 if rs.random()<.5 else 1,n,n,generator=gen);f=F.interpolate(f,size=(TS,TS),mode='bicubic',align_corners=True)[0].clamp(0,1)
        img=img*(rs.uniform(.35,.8)+rs.uniform(.2,.65)*f)
    return img.clamp(0,1)
def sample(it,rs,var=None,csrc=None):
    _,t,m=load(it);var=var or ('g' if rs.random()>=TEX else rs.choice(['l0','l1']));im=image(it,var)
    tt=t[0] if COND else t
    for _ in range(20):
        u=math.exp(rs.uniform(math.log(UMIN),math.log(UMAX)));S=int(round(TS/u));x=rs.randint(0,1024-S);y=rs.randint(0,1024-S)
        mm=m[y:y+S,x:x+S]
        if mm.mean()>0.6:break
    img=np.array(im.crop((x,y,x+S,y+S)).resize((TS,TS),Image.Resampling.LANCZOS)).astype(np.float32)/255
    img=torch.from_numpy(img).permute(2,0,1)
    if csrc is None:img=photo(img,rs,var!='g')
    if COND:
        src=csrc or ('lp' if rs.random()<COND_LP else 'lr');src='lp' if t[1] is None else src;cm_=cond_map(t,m,y,x,S,src,rs.choice(CLPS))
        if csrc is None:
            cm_=misalign(cm_,rs)
            if rs.random()<CDROP:cm_=torch.tensor([0.,0,1])[:,None,None].expand(3,TS,TS).clone()
        img=torch.cat([img,cm_])   # (6,TS,TS): rgb + cond normals
    return img,torch.from_numpy(tt[y:y+S,x:x+S].astype(np.float32)).permute(2,0,1),torch.from_numpy(mm),S
# ---------------- model
pipe=YOSONormalsPipeline.from_pretrained(f'{_E["SF3D_NIRNE_WEIGHTS"]}',variant='fp16',torch_dtype=torch.float32,trust_remote_code=True,safety_checker=None).to('cuda')
unet,ctl,vae=pipe.unet,pipe.controlnet,pipe.vae
for mdl in (unet,ctl,vae):mdl.requires_grad_(False)
params=[]
for name,mdl in (('unet',unet),('controlnet',ctl)):
    if name in TRAIN:mdl.requires_grad_(True);mdl.train();params+=list(mdl.parameters())
    if hasattr(mdl,'enable_gradient_checkpointing'):mdl.enable_gradient_checkpointing()
vae.eval()
try:vae.enable_gradient_checkpointing()
except Exception:pass
opt=torch.optim.AdamW(params,lr=LR,weight_decay=0.0)
emb=torch.zeros(1,257,1024,device='cuda');sf=vae.config.scaling_factor
def forward(img):                                                     # img (B,3|6,TS,TS): rgb in [0,1] (+ cond normals in [-1,1])
    x=img[:,:3].cuda()*2-1
    with torch.autocast('cuda',dtype=torch.bfloat16):
        with torch.no_grad():
            lat=vae.encode(x).latent_dist.mode()*sf
            start=vae.encode(img[:,3:6].cuda()).latent_dist.mode()*sf if COND else lat     # UNet start latent
        e=emb.expand(len(x),-1,-1)
        dn,mid=ctl(lat,pipe.t_start,encoder_hidden_states=e,conditioning_scale=1.0,guess_mode=False,return_dict=False)
        out=unet(start,pipe.t_start,encoder_hidden_states=e,down_block_additional_residuals=dn,mid_block_additional_residual=mid,return_dict=False)[0]
        p=vae.decode(out/sf,return_dict=False)[0]
    p=p.float();return p/p.norm(dim=1,keepdim=True).clamp_min(1e-6)
def hp(N,m,k):                                                        # N (3,H,W), m (H,W) float
    w=F.avg_pool2d(m[None,None],k+1,1,k//2,count_include_pad=False)[0].clamp_min(1e-6)
    return N-F.avg_pool2d((N*m)[None],k+1,1,k//2,count_include_pad=False)[0]/w
def loss_one(p,t,m,S):
    p=F.interpolate(p[None],size=(S,S),mode='area')[0];p=p/p.norm(dim=0,keepdim=True).clamp_min(1e-6)
    t=t.cuda();m=m.cuda().float();c=(1-(p*t).sum(0))
    lc=(c*m).sum()/m.sum().clamp_min(1);lh=0
    for k in (4,8):lh=lh+((((hp(p,m,k)-hp(t,m,k))**2).sum(0))*m).sum()/m.sum().clamp_min(1)
    return lc,lh
if INIT:
    sd0=torch.load(INIT,map_location='cuda')
    for name,mdl in (('unet',unet),('controlnet',ctl)):
        if name in sd0:mdl.load_state_dict({a:b.float() for a,b in sd0[name].items()})
    print('init from',INIT,flush=True)
rs=random.Random(SEED)
def vset(var,pool=None):pool=pool or [x for x in va if '/hires/v2/' not in x[2]];r_=random.Random(7);return [sample(pool[i%len(pool)],r_,var,'lr') for i in range(64)]
VB={'grey':vset('g')}
if TEX>0:VB['tex_l0']=vset('l0');VB['tex_l1']=vset('l1')
VA2=[x for x in va if '/hires/v2/' in x[2]]
if VA2:VB['v2_grey']=vset('g',VA2);VB['v2_tex_l0']=vset('l0',VA2) if TEX>0 else None
VB={k:v for k,v in VB.items() if v}
def validate():
    unet.eval();ctl.eval();res=[]
    with torch.no_grad():
        for nm,vb in VB.items():
            a=b=0
            for i in range(0,len(vb),8):
                ch=vb[i:i+8];p=forward(torch.stack([x[0] for x in ch]))
                for j,(img,t,m,S) in enumerate(ch):lc,lh=loss_one(p[j],t,m,S);a+=float(lc);b+=float(lh)
            res+=[a/len(vb),b/len(vb)]
    unet.train() if 'unet' in TRAIN else None;ctl.train() if 'controlnet' in TRAIN else None
    return tuple(res)
log=open(f'{OUT}/log.txt','a');VN=' | '.join(f'{k} cos/hp' for k in VB);v0=validate();print('val step 0 ',VN,' '.join('%.5f'%x for x in v0),flush=True);log.write('0 val '+' '.join('%.6f'%x for x in v0)+'\n')
t0=time.time()
for step in range(1,STEPS+1):
    lr=LR*min(1,step/200);[g.__setitem__('lr',lr) for g in opt.param_groups]
    batch=[sample(rs.choices(tr,weights=TRW)[0],rs) for _ in range(BS)]
    p=forward(torch.stack([b[0] for b in batch]));L=0;LC=LH=0
    for j,(img,t,m,S) in enumerate(batch):lc,lh=loss_one(p[j],t,m,S);L=L+(lc+LHP*lh)/BS;LC+=float(lc)/BS;LH+=float(lh)/BS
    opt.zero_grad(set_to_none=True);L.backward();torch.nn.utils.clip_grad_norm_(params,1.0);opt.step()
    if step%50==0:print(f'step {step} cos {LC:.5f} hp {LH:.5f} {(time.time()-t0)/step:.2f}s/it mem {torch.cuda.max_memory_allocated()/1e9:.1f}G',flush=True)
    if step%500==0 or step==STEPS:
        v=validate();print('val step %d '%step,VN,' '.join('%.5f'%x for x in v),flush=True);log.write('%d val '%step+' '.join('%.6f'%x for x in v)+'\n');log.flush()
        sd={k:v_ for name,mdl in (('unet',unet),('controlnet',ctl)) if name in TRAIN for k,v_ in [(name,{a:b.half() for a,b in mdl.state_dict().items()})]}
        torch.save(sd,f'{OUT}/ckpt_last.pt' if SAVE=='last' else f'{OUT}/ckpt_{step:05d}.pt')
        if SAVE=='last':open(f'{OUT}/ckpt_last.step','w').write(str(step))
print('FT_DONE',flush=True)
