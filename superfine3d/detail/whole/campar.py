"""Exact parallel camera estimation: every Powell start of fit_input_camera.py / camera_refine.py runs in its own process;
the combine pass loads the partial results and applies the scripts' own selection, so the output equals the sequential run.
usage: campar.py INPUT_PNG MESH OUT_DIR   (writes OUT_DIR/cam_fit.json, OUT_DIR/cam_refined.json)"""
from os import environ as _E   # install locations, set by setup/env.sh (SF3D_ROOT, SF3D_CKPT, ...)
import os,sys,subprocess,json,time
P=f'{_E["SF3D_ROOT"]}/superfine3d/detail';INP,MESH,O=sys.argv[1:4];os.makedirs(O,exist_ok=True)
PY=sys.executable;env=dict(os.environ);env['PYTHONPATH']=':'.join([f'{_E["SF3D_ROOT"]}/third_party/site-packages']+[p for p in sys.path if p]+[env.get('PYTHONPATH','')])
def par(script,args,n,var,partdir,extra=None):
    os.makedirs(partdir,exist_ok=True);ps=[]
    for i in range(n):
        e=dict(env);e.update({var+'_ONLY':str(i),var+'_PARTDIR':partdir});e.update(extra or {})
        ps.append(subprocess.Popen([PY,script]+args,env=e,stdout=subprocess.DEVNULL,stderr=open(f'{partdir}/err{i}.log','w')))
    rc=[p.wait() for p in ps];assert all(r==0 for r in rc),(script,rc)
    e=dict(env);e[var+'_RESULTS']=partdir;r=subprocess.run([PY,script]+args,env=e,capture_output=True,text=True);assert r.returncode==0,r.stderr[-2000:]
    return r.stdout
t=time.time()
out1=par(f'{P}/fit_input_camera.py',[INP,MESH,f'{O}/cam_fit.json'],int(os.environ.get('CAMFIT_NSTART','4')),'CAMFIT',f'{O}/fitparts')
t1=time.time()
out2=par(f'{P}/whole/camera_refine.py',[INP,MESH,f'{O}/cam_fit.json',f'{O}/cam_refined.json'],12,'CAMREF',f'{O}/refparts')
print('CAMPAR fit %.1fs refine %.1fs'%(t1-t,time.time()-t1),out2.strip().split('\n')[-1],flush=True)
