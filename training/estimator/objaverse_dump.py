"""Blender mesh dump with the TRELLIS.2 toolkit script (same as data_toolkit/dump_mesh.py) of the assets in the given lists.
usage: objaverse_dump.py LIST.json [...]   (WORKERS=16)"""
from os import environ as _E   # install locations, set by setup/env.sh (SF3D_ROOT, SF3D_CKPT, ...)
import json,os,sys,subprocess,tempfile,shutil,glob,time
from concurrent.futures import ThreadPoolExecutor
R=f'{_E["SF3D_DATA"]}/estimator/objaverse';os.makedirs(R+'/mesh_dumps',exist_ok=True)
BL=f'{_E["SF3D_BLENDER"]}'
SC=f'{_E["SF3D_ROOT"]}/training/trellis2_variants/data_toolkit/blender_script/dump_mesh.py'
def load(n):   # asset list: training/estimator/data/<split>.json ({sha256: {uid, crops}}) or data/v2/v2bN.json ([{uid, sha256, ...}])
    d=json.load(open(n if os.path.exists(n) else f'{R}/{n}'))
    return [dict(sha256=k,uid=a['uid']) for k,a in d.items()] if isinstance(d,dict) else d
sel=[x for n in sys.argv[1:] for x in load(n)]
glbs={os.path.basename(p)[:-4]:p for p in glob.glob(R+'/raw/glbs/*/*.glb')}
def one(x):
    out=f"{R}/mesh_dumps/{x['sha256']}.pickle"
    if os.path.exists(out):return x['sha256'],'cached'
    g=glbs.get(x['uid'])
    if g is None:return x['sha256'],'no_glb'
    with tempfile.TemporaryDirectory() as t:
        tmp=os.path.join(t,'m.pickle')
        try:subprocess.run([BL,'-b','-P',SC,'--','--object',g,'--output_path',tmp],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=600)
        except subprocess.TimeoutExpired:return x['sha256'],'timeout'
        if not os.path.exists(tmp):return x['sha256'],'failed'
        shutil.move(tmp,out);return x['sha256'],'ok'
t=time.time();res={}
with ThreadPoolExecutor(int(os.environ.get('WORKERS','16'))) as ex:
    for i,(s,st) in enumerate(ex.map(one,sel)):
        res[s]=st
        if i%50==0:print(i,round(time.time()-t),st,flush=True)
from collections import Counter
json.dump(res,open(R+'/dump_status.json','w'),indent=1);print(Counter(res.values()),round(time.time()-t))
