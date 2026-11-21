"""Download the GLBs of the assets in the given lists from allenai/objaverse (Hugging Face) into $SF3D_DATA/estimator/objaverse/raw.
usage: objaverse_download.py LIST.json [...]"""
from os import environ as _E   # install locations, set by setup/env.sh (SF3D_ROOT, SF3D_CKPT, ...)
import json,gzip,os,sys,time
from concurrent.futures import ThreadPoolExecutor
os.environ.pop('HF_HUB_OFFLINE',None)   # setup/env.sh sets it; huggingface_hub reads it at import
from huggingface_hub import hf_hub_download
R=f'{_E["SF3D_DATA"]}/estimator/objaverse';os.makedirs(R+'/raw',exist_ok=True)
def load(n):   # asset list: training/estimator/data/<split>.json ({sha256: {uid, crops}}) or data/v2/v2bN.json ([{uid, sha256, ...}])
    d=json.load(open(n if os.path.exists(n) else f'{R}/{n}'))
    return [dict(sha256=k,uid=a['uid']) for k,a in d.items()] if isinstance(d,dict) else d
names=sys.argv[1:];sel=[x for n in names for x in load(n)]
p=hf_hub_download('allenai/objaverse','object-paths.json.gz',repo_type='dataset',local_dir=R+'/raw')
paths=json.load(gzip.open(p))
missing=[x['uid'] for x in sel if x['uid'] not in paths];print('missing uid',len(missing),flush=True)
def get(x):
    if x['uid'] not in paths:return x['sha256'],None,'no_path'
    for a in range(3):
        try:return x['sha256'],hf_hub_download('allenai/objaverse',paths[x['uid']],repo_type='dataset',local_dir=R+'/raw'),None
        except Exception as e:err=str(e);time.sleep(5)
    return x['sha256'],None,err[:300]
t=time.time();out={}
with ThreadPoolExecutor(16) as ex:
    for i,(s,path,err) in enumerate(ex.map(get,sel)):
        out[s]=dict(path=path,error=err)
        if i%50==0:print(i,round(time.time()-t),flush=True)
json.dump(out,open(f'{R}/download_{"_".join(os.path.basename(n).split(".")[0] for n in names)}.json','w'),indent=1)
print('ok',sum(v['path'] is not None for v in out.values()),'of',len(out),'seconds',round(time.time()-t))
