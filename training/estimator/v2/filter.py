"""Keep v2 crops with enough visible surface and fine detail (target normals high-pass energy): writes ft_nirne/v2/keep.txt
usage: filter.py [HP4_MIN=0.05] [COVER_MIN=0.4] [MAX_PER_ASSET=8]"""
from os import environ as _E   # install locations, set by setup/env.sh (SF3D_ROOT, SF3D_CKPT, ...)
import os,sys,json,numpy as np,collections
from scipy.ndimage import uniform_filter
D=f'{_E["SF3D_DATA"]}/estimator';T=f'{D}/ft_nirne/targets';H,C,M=[float(a) for a in (sys.argv[1:]+['0.05','0.4','8'])[:3]]
st=json.load(open(f'{D}/ft_nirne/v2/stats.json')) if os.path.exists(f'{D}/ft_nirne/v2/stats.json') else {}
for c in sorted(os.listdir(f'{D}/hires/v2')):
    if c in st or not os.path.exists(f'{T}/{c}.npz'):continue
    with np.load(f'{T}/{c}.npz') as a:n=a['n'].astype(np.float32);m=a['m']
    if m.mean()<.05:st[c]=dict(cover=float(m.mean()),hp4=0.);continue
    mf=m.astype(np.float32);lp=np.stack([uniform_filter(n[...,i]*mf,5)/np.maximum(uniform_filter(mf,5),1e-6) for i in range(3)],-1)
    st[c]=dict(cover=float(m.mean()),hp4=float(np.sqrt(((n-lp)**2).sum(-1)[m].mean())))
json.dump(st,open(f'{D}/ft_nirne/v2/stats.json','w'))
by=collections.defaultdict(list)
for c,s in st.items():
    if s['cover']>=C and s['hp4']>=H:by[c.split('_')[0]].append((s['hp4'],c))
keep=[c for u,l in by.items() for _,c in sorted(l,reverse=True)[:int(M)]]
open(f'{D}/ft_nirne/v2/keep.txt','w').write('\n'.join(sorted(keep)));print('views',len(st),'kept',len(keep),'assets',len(by))
