"""Driver: textured crop views (ft_nirne/tex_render_bl.py) for a list of crop ids, grouped by asset, GPUs in parallel.
Cameras come from hires/{z1,train,v2}/<cid>/cam.json; GLBs from objaverse_download.py. Output ft_nirne/tex/<cid>/view_l{0,1}.png (RGBA).
usage: tex_render.py LIST.txt GPUS(e.g. 4,5,6,7) [WORKERS_PER_GPU=2]"""
from os import environ as _E   # install locations, set by setup/env.sh (SF3D_ROOT, SF3D_CKPT, ...)
import sys,os,json,subprocess,tempfile,time,itertools
from concurrent.futures import ThreadPoolExecutor
D=f'{_E["SF3D_DATA"]}/estimator';OUT=f'{D}/ft_nirne/tex';os.makedirs(OUT,exist_ok=True)
BL=f'{_E["SF3D_BLENDER"]}'
ids=open(sys.argv[1]).read().split();gpus=sys.argv[2].split(',');wpg=int(sys.argv[3]) if len(sys.argv)>3 else 2
import glob as _g
glb={os.path.basename(p)[:-4]:p for p in _g.glob(f'{D}/objaverse/raw/glbs/*/*.glb')}   # Objaverse uid -> GLB
for _s in ('z1','train'):   # z1 assets are named by the TRELLIS sha256 of the file, the others by their Objaverse uid
    for _k,_a in json.load(open(f'{_E["SF3D_ROOT"]}/training/estimator/data/{_s}.json')).items():
        if _a['uid'] in glb:glb[_k]=glb[_a['uid']]
def camdir(c):
    for s in ('z1','train','v2'):
        if os.path.exists(f'{D}/hires/{s}/{c}/cam.json'):return f'{D}/hires/{s}/{c}'
by={}
for c in ids:
    if all(os.path.exists(f'{OUT}/{c}/view_l{i}.png') for i in (0,1)):continue
    by.setdefault(c.split('_')[0],[]).append(c)
jobs=sorted(by.items());slots=[g for g in gpus for _ in range(wpg)];print('assets',len(jobs),'views',sum(len(v) for _,v in jobs),flush=True)
import queue;free=queue.Queue();[free.put(g) for g in slots]
def run(item):
    sha,cs=item;g=free.get()
    try:
        views=[dict(cid=c,cam=json.load(open(camdir(c)+'/cam.json')),out=f'{OUT}/{c}') for c in cs]
        with tempfile.NamedTemporaryFile('w',suffix='.json',delete=False,dir=f'{D}/ft_nirne') as f:json.dump(dict(glb=glb[sha],views=views,seed=int(sha[:8],16)%2**31),f);jp=f.name
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=g);t=time.time()
        p=subprocess.run([BL,'-b','-P',f'{_E["SF3D_ROOT"]}/training/estimator/tex_render_bl.py','--',jp],env=env,capture_output=True,text=True,timeout=7200)
        os.remove(jp);ok='[TEX] ALL_DONE' in p.stdout
        return sha,len(cs),ok,round(time.time()-t),'' if ok else (p.stdout[-400:]+p.stderr[-400:])
    finally:free.put(g)
with ThreadPoolExecutor(len(slots)) as ex:
    for i,(sha,n,ok,dt,err) in enumerate(ex.map(run,jobs)):print(i,sha[:12],n,'ok' if ok else 'FAIL',f'{dt}s',err,flush=True)
print('TEX_DONE',flush=True)
