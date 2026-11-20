from os import environ as _E   # install locations, set by setup/env.sh (SF3D_ROOT, SF3D_CKPT, ...)
from pathlib import Path
import os,sys,json,time,fcntl,subprocess,math,hashlib
R=Path(f'{_E["SF3D_TOYS4K_EVAL"]}');D=R/'detail100'
METHODS=['TRELLIS2','Hunyuan21','Pixal3D','Ours2048','Ours4096']
POLICY=dict(CD='CD-L1: halfmean of two directional mean Euclidean point-to-triangle distances',P95='max of two directional95th percentiles of Euclidean point-to-triangle distance',samples_per_direction=2000000,seeds=[101,102],thresholds=[.0001,.001,.002,.01],units='v4 normalized sphere diameter1',sampling='area uniform; original triangle surfaces',alignment='same frozen preparation.json as existing Fscore and NC')
def stats(pd,gd):
 import numpy as np
 p,g=np.sqrt(pd),np.sqrt(gd)
 p95,g95=float(np.percentile(p,95)),float(np.percentile(g,95))
 return dict(CD_L1=float((p.mean()+g.mean())/2),P95=max(p95,g95),pred_to_GT_mean=float(p.mean()),GT_to_pred_mean=float(g.mean()),pred_to_GT_P95=p95,GT_to_pred_P95=g95,thresholds={str(t):dict(precision=float((p<t).mean()),recall=float((g<t).mean())) for t in POLICY['thresholds']})
def save(p,v):
 p=Path(p);q=p.with_name(p.name+'.'+str(os.getpid())+'.tmp');q.write_text(json.dumps(v,indent=2,allow_nan=False));q.replace(p)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def worker(key,method):
 import numpy as np
 sys.path.insert(0,str(R/'code'));import toys_metrics_v4 as v4
 guard=open(R/'detail50/cpu_geometry.lock','a');fcntl.flock(guard,fcntl.LOCK_EX)
 prep=json.loads((D/key/'preparation.json').read_text());s=prep['sources'][method]
 assert sha(s['path'])==s['sha256'];assert sha(R/key/'input_data/input.png')==prep['input_sha256']
 gp=R/key/'input_data/GT_normalized.ply';assert sha(gp)==prep['GT_sha256']
 m,_=v4.normalize(v4.load(s['path']));gt,_=v4.normalize(v4.load(gp));a=prep['alignment'][method]
 rot=np.asarray(a['rotation']);assert abs(np.linalg.det(rot)-1)<1e-6;m.vertices=np.asarray(m.vertices)@rot+np.asarray(a['translation'])
 p=v4.base.sampled(m,2000000,101);g=v4.base.sampled(gt,2000000,102)
 pd=v4.base.mesh_distances_squared(gt,p);gd=v4.base.mesh_distances_squared(m,g)
 out=stats(pd,gd)
 old=D/key/(method+'.json')
 if old.exists():
  old=json.loads(old.read_text())
  for t,v in out['thresholds'].items():
   for k in ['precision','recall']:assert abs(v[k]-old['fscore']['2000000'][t][k])<1e-12
 save(D/key/(method+'_distance.json'),dict(complete=True,asset=key,method=method,mesh_sha256=s['sha256'],input_sha256=prep['input_sha256'],GT_sha256=prep['GT_sha256'],values=out,protocol=POLICY))
def summary(failures):
 cohort=[];means={};inputs={}
 for i in range(1,101):
  key=f'TC{i:03}'
  try:
   prep=json.loads((D/key/'preparation.json').read_text())
   for method in METHODS:
    vals=[json.loads((D/key/(method+suffix)).read_text()) for suffix in ['.json','_nc_v2.json','_distance.json']]
    for v in vals:
     assert v['complete'] and v['input_sha256']==prep['input_sha256'] and v['GT_sha256']==prep['GT_sha256']
     assert v.get('mesh_sha256',v.get('mesh',{}).get('sha256'))==prep['sources'][method]['sha256']
    inputs[key,method]=vals
   cohort.append(key)
  except (OSError,KeyError,AssertionError,ValueError):continue
 for method in METHODS:
  if not cohort:continue
  means[method]={}
  for k in ['CD_L1','P95']:means[method][k]=sum(inputs[a,method][2]['values'][k] for a in cohort)/len(cohort)
  for t in POLICY['thresholds']:
   for k in ['precision','recall','fscore']:means[method][k+'@'+str(t)]=sum(inputs[a,method][0]['fscore']['2000000'][str(t)][k] for a in cohort)/len(cohort)
  means[method]['NC']=sum(inputs[a,method][1]['NC'] for a in cohort)/len(cohort)
  means[method]['DQ']=sum(inputs[a,method][0]['richness']['values']['quality_4096'] for a in cohort)/len(cohort)
 save(D/'summary_all_metrics.json',dict(requested_assets=100,complete_count=len(cohort),complete_cohort=cohort,means=means,failures=failures,distance_protocol=POLICY,updated_unix=time.time()))
 return len(cohort)
def main():
 if len(sys.argv)>1:worker(*sys.argv[1:]);return
 import numpy as np
 a=stats(np.zeros(100),np.zeros(100));b=stats(np.full(100,.01**2),np.full(100,.02**2))
 assert a['CD_L1']==a['P95']==0 and a['thresholds']['0.001']['precision']==1
 assert abs(b['CD_L1']-.015)<1e-12 and abs(b['P95']-.02)<1e-12
 own=open(D/'distance_runner.lock','a');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
 save(D/'distance_protocol.json',POLICY);save(D/'distance_statistics_gate.json',dict(passed=True,identity=a,known_distances=b))
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1');failed=set();failures=[]
 while True:
  for i in range(1,101):
   key=f'TC{i:03}';out=D/key
   if not (out/'preparation.json').exists():continue
   for method in METHODS:
    if (key,method) in failed or (out/(method+'_distance.json')).exists():continue
    save(D/'distance_status.json',dict(stage='running',pid=os.getpid(),asset=key,method=method))
    with (out/(method+'_distance.log')).open('a') as f:rc=subprocess.call([sys.executable,__file__,key,method],env=env,stdout=f,stderr=subprocess.STDOUT)
    if rc:failed.add((key,method));failures.append(dict(asset=key,method=method,returncode=rc))
    summary(failures)
  if summary(failures)==100:save(D/'distance_status.json',dict(stage='complete',pid=os.getpid()));break
  save(D/'distance_status.json',dict(stage='waiting_missing_inputs_or_repairs',pid=os.getpid(),failures=failures));time.sleep(60)
if __name__=='__main__':main()
