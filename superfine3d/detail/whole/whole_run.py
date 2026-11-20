"""Whole-object mode of hfdetail on a full 4096 mesh (one height field over the whole 4096 px input view, no crops).
Arms (all from the same refined 4096 mesh, object frame, Frame(zoom 1, unit 1/4096), visibility tolerance 2/1024,
i.e. the zoom-4 crop setting):
  base        the refined mesh as generated
  smooth      control: residual target with gain 0 (mesh's own 32 px low-pass only, no image detail)
  ft_resid    fine-tuned NiRNE (tiled x3) detail above 32 px on the mesh's coarse normals, gain 0.9 (main setting)
  ft_raw      fine-tuned NiRNE estimate as the target
  oracle      GT normals of the input view as the target (upper bound)
Evaluation: face normals rendered at 4096 from the input view and +-35 deg about the object centre vs the GT dump
normals; ang (deg) on pixels covered by both, ang_lo (8 px low-pass), hf (detail energy ratio vs GT).
Env: INPUT (input_grey_4096.png, relative to DIR) TAG (output suffix, '')
usage: whole_run.py DIR CKPT [ARMS]"""
from os import environ as _E   # install locations, set by setup/env.sh (SF3D_ROOT, SF3D_CKPT, ...)
import sys,os,json,time,numpy as np,torch,torch.nn.functional as F_
sys.path.insert(0,f'{_E["SF3D_ROOT"]}/superfine3d/detail')
from PIL import Image
from hfdetail import pipeline,heightfield as hf,estimator,raster,camera,targets
DIR,CKPT=sys.argv[1:3];INP=os.environ.get('INPUT','input_grey_4096.png');TAG=os.environ.get('TAG','');ARMS=sys.argv[3].split(',') if len(sys.argv)>3 else ['smooth','ft_resid','ft_raw','oracle'];RES=4096
def read_ply(p):
    with open(p,'rb') as fh:
        hdr=[];
        while True:
            l=fh.readline().decode().strip();hdr.append(l)
            if l=='end_header':break
        assert 'format binary_little_endian 1.0' in hdr,hdr[:3]
        nv=int([l for l in hdr if l.startswith('element vertex')][0].split()[-1]);nf=int([l for l in hdr if l.startswith('element face')][0].split()[-1])
        vi=hdr.index([l for l in hdr if l.startswith('element vertex')][0]);fi=hdr.index([l for l in hdr if l.startswith('element face')][0])
        props=[l.split() for l in hdr[vi+1:fi] if l.startswith('property')];T={'float':'<f4','double':'<f8','uchar':'u1','int':'<i4','uint':'<u4'}
        vd=np.dtype([(pp[2],T[pp[1]]) for pp in props]);V=np.fromfile(fh,dtype=vd,count=nv)
        fl=[l.split() for l in hdr[fi+1:] if l.startswith('property list')][0];assert fl[2]=='uchar' and fl[3] in ('int','uint'),fl
        Fd=np.dtype([('n','u1'),('i',T[fl[3]],(3,))]);Fa=np.fromfile(fh,dtype=Fd,count=nf);assert (Fa['n']==3).all()
    return np.stack([V['x'],V['y'],V['z']],1).astype(np.float32),Fa['i'].astype(np.int64)
t=time.time();v,Fn=read_ply(f'{DIR}/mesh_4096.ply');v=torch.from_numpy(v).cuda();Fm=torch.from_numpy(Fn).cuda();del Fn
print(f'mesh {len(v)/1e6:.1f}M vertices {len(Fm)/1e6:.1f}M faces, bbox {v.min(0).values.tolist()} {v.max(0).values.tolist()}  load {time.time()-t:.0f}s',flush=True)
cam=json.load(open(f'{DIR}/cam.json'));cams=json.load(open(f'{DIR}/cams_eval.json'))
rgba=np.array(Image.open(f'{DIR}/{INP}').convert('RGBA')).astype(np.float32)/255;img=(rgba[...,:3]*rgba[...,3:4]*255).clip(0,255).astype(np.uint8)
stats={}
ef=f'{DIR}/est_ft{TAG}.npz'
if not os.path.exists(ef):
    torch.cuda.synchronize();t=time.time();E,Em=estimator.estimate(img,cam,RES,estimator.NiRNE(CKPT),batch=8);torch.cuda.synchronize()
    stats['estimate_s']=time.time()-t;np.savez_compressed(ef,n=E.half().cpu().numpy(),m=E.new_tensor(Em.float()).bool().cpu().numpy());print('estimate %.0fs'%stats['estimate_s'],flush=True)
with np.load(ef) as a:E=torch.from_numpy(a['n']).float().cuda();Em=torch.from_numpy(a['m']).cuda()
with np.load(f'{DIR}/gt_normals_0.npz') as a:Gn=torch.from_numpy(a['n']).float().cuda();Gm=torch.from_numpy(a['m']).cuda()
frame=hf.Frame((0.,0.,0.),1.,1/4096);p=hf.HFParams(iters=800);out={'base':v}
spec=dict(smooth=dict(target=(E,Em),mode='resid',k=32,gain=0.),ft_resid=dict(target=(E,Em),mode='resid',k=32,gain=.9),
          ft_raw=dict(target=(E,Em),mode='raw'),oracle=dict(target=(Gn,Gm),mode='raw'))
for a_ in ARMS:
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.time();st={}
    out[a_]=pipeline.refine(v,Fm,frame,cam,params=p,stats=st,**spec[a_]);torch.cuda.synchronize()
    st.update(total_s=time.time()-t,peak_gib=torch.cuda.max_memory_allocated()/2**30,moved_frac=float(((out[a_]-v).abs().sum(1)>0).float().mean()))
    stats[a_]=st;print(a_,json.dumps(st),flush=True)
# ---------------- evaluation
def lp(N,m,k=8):
    Nb=F_.avg_pool2d((N*m[...,None]).permute(2,0,1)[None],k,1,k//2,count_include_pad=False)[0].permute(1,2,0)[:N.shape[0],:N.shape[1]]
    return Nb/Nb.norm(dim=-1,keepdim=True).clamp_min(1e-6)
ang=lambda a,b,m:float(torch.rad2deg(torch.acos((a*b).sum(-1).clamp(-1,1)))[m].mean())
rows={};shade={}
for va,cm in cams.items():
    with np.load(f'{DIR}/gt_normals_{va}.npz') as a:On=torch.from_numpy(a['n']).float().cuda();Om=torch.from_numpy(a['m']).cuda()
    Olp=lp(On,Om);Ohf=float((On-Olp).norm(dim=-1)[Om].mean())
    for a_,vv in out.items():
        tri=raster.tri_ids(raster.rasterize(camera.clip_coords(frame.to_object(vv),cm['P']),Fm,RES));N,m=raster.normal_map(frame.to_object(vv),Fm,tri,cm['fwd'])
        b=m&Om;Nlp=lp(N,m);rows[f'{a_}@{va}']=dict(ang=ang(N,On,b),ang_lo=ang(Nlp,Olp,b),hf=float((N-Nlp).norm(dim=-1)[b].mean())/Ohf,cover=float(b.sum()/Om.sum()))
        V=-torch.tensor(cm['fwd'],device='cuda').float();U=torch.tensor(cm['u'],device='cuda').float();R=torch.tensor(cm['r'],device='cuda').float();L=V+.5*U-.3*R;L=L/L.norm()
        shade[(a_,va)]=((.08+.9*(.65*(N@L).clamp(0,1)+.35*(N@V).clamp(0,1)))*m+(~m)).clamp(0,1).cpu().numpy()
    V=-torch.tensor(cm['fwd'],device='cuda').float();U=torch.tensor(cm['u'],device='cuda').float();R=torch.tensor(cm['r'],device='cuda').float();L=V+.5*U-.3*R;L=L/L.norm()
    shade[('GT',va)]=((.08+.9*(.65*(On@L).clamp(0,1)+.35*(On@V).clamp(0,1)))*Om+(~Om)).clamp(0,1).cpu().numpy()
print('%-10s'%'arm'+''.join('%22s'%f'view {va}' for va in cams))
for a_ in out:print('%-10s'%a_+''.join('   %6.2f (lo %5.2f hf %.2f)'%(rows[f'{a_}@{va}']['ang'],rows[f'{a_}@{va}']['ang_lo'],rows[f'{a_}@{va}']['hf']) for va in cams))
json.dump(dict(stats=stats,rows=rows),open(f'{DIR}/whole_eval{TAG}.json','w'),indent=1)
# visual: 3 zoomed windows (512 px of the 4096 view) per view, columns base / ft_resid / oracle / GT
cols=[c for c in ('base','smooth','ft_resid','oracle','GT') if c=='GT' or c in out];Wn=512;S=360
m0=np.load(f'{DIR}/gt_normals_0.npz')['m'];ys,xs=np.nonzero(m0);rng=np.random.RandomState(0);wins=[]
while len(wins)<3:
    i=rng.randint(len(ys));y0,x0=int(ys[i])-Wn//2,int(xs[i])-Wn//2
    if 0<=y0<RES-Wn and 0<=x0<RES-Wn and m0[y0:y0+Wn,x0:x0+Wn].mean()>.9:wins.append((y0,x0))
canvas=np.ones((len(wins)*len(cams)*S,len(cols)*S));r_=0
for (y0,x0) in wins:
    for va in cams:
        for j,c in enumerate(cols):
            im=Image.fromarray((shade[(c,va)][y0:y0+Wn,x0:x0+Wn]*255).astype(np.uint8)).resize((S,S),Image.Resampling.LANCZOS)
            canvas[r_*S:(r_+1)*S,j*S:(j+1)*S]=np.array(im)/255
        r_+=1
Image.fromarray((canvas*255).astype(np.uint8)).save(f'{DIR}/whole_compare{TAG}.png');print('cols',cols,'rows: windows x views',list(cams));print('WHOLE_DONE')
