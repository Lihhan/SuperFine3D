"""Pixal3D native low-resolution output: official inference, stopped right after Stage 2 (512 shape latent, pixel-aligned
projection conditioning). Saves OUT/latent_512.npz (denormalised, same VAE as TRELLIS.2), OUT/mesh_512.ply, official camera.
usage: pixal512_worker.py INPUT.png OUT"""
from os import environ as _E   # install locations, set by setup/env.sh (SF3D_ROOT, SF3D_CKPT, ...)
import os,sys,json,time,types
from pathlib import Path
IMG,O=sys.argv[1],Path(sys.argv[2]);O.mkdir(parents=True,exist_ok=True);base=Path(f'{_E["SF3D_PIXAL3D"]}');sys.path.insert(0,str(base/'repo'))
import torch,numpy as np,trimesh
import inference as api
from pixal3d.pipelines import Pixal3DImageTo3DPipeline as Pipeline
os.environ['FLEX_GEMM_AUTOTUNE_CACHE_PATH']=f'{_E["SF3D_CACHE"]}/pixal_autotune.json'
torch.set_num_threads(4);torch.set_grad_enabled(False);torch.cuda.set_per_process_memory_fraction(.45)
Pipeline.model_names_to_load=['sparse_structure_flow_model','sparse_structure_decoder','shape_slat_flow_model_512','shape_slat_flow_model_1024','shape_slat_decoder']
FAST=os.environ.get('PIXAL_FAST')=='1'   # opt-in exact (eff_opt r6): skip the unused 1024 flow model, keep models resident (no low-VRAM swapping)
if FAST:Pipeline.model_names_to_load=[m for m in Pipeline.model_names_to_load if m!='shape_slat_flow_model_1024']
if os.environ.get('PIXAL_NOINIT')=='1':   # opt-in exact: build the flow/decoder models without random init (all weights come from the
    sys.path.insert(0,f'{_E["SF3D_ROOT"]}/superfine3d/detail/bench');import noinit,pixal3d.models as _pm;noinit.wrap(_pm)   # checkpoint; coverage asserted)
oi=api.init_pipeline;cap={}
class Done(Exception):pass
def init(*a,**kw):
 p=oi(*a,**kw);cap['p']=p;p.models['tex_slat_flow_model_1024']=None
 if FAST:p.models.setdefault('shape_slat_flow_model_1024',None)
 old=p.sample_shape_slat
 def sss(self,*a,**kw):
  z=old(*a,**kw);cap['z']=z;cap['t512']=time.time();raise Done()
 p.sample_shape_slat=types.MethodType(sss,p);pre=p.preprocess_image
 def prep(self,*a,**kw):
  im=pre(*a,**kw);im.save(O/'actual_model_input.png');return im
 p.preprocess_image=types.MethodType(prep,p);return p
api.init_pipeline=init;oc=api.get_camera_params_wild_moge
def camera(*a,**kw):
 c=oc(*a,**kw);(O/'official_camera.json').write_text(json.dumps(c,default=lambda v:float(v),indent=2));return c
api.get_camera_params_wild_moge=camera;t0=time.time()
FOV=float(os.environ.get('PIXAL_FOV','-1'))   # opt-in: known input FOV in degrees (official --fov; e.g. 10 = narrowest trained FOV for orthographic renders); default MoGe-2 estimate
if FOV>0:(O/'official_camera.json').write_text(json.dumps(dict(manual_fov_deg=FOV),indent=2))
try:api.run_inference(IMG,str(O/'unused.glb'),seed=int(os.environ.get('SEED','42')),model_path=str(base/'weights'),low_vram=not FAST,resolution=1024,**(dict(manual_fov=__import__('math').radians(FOV)) if FOV>0 else {}))
except Done:pass
z=cap['z'];LEAN=os.environ.get('PIXAL_LEAN')=='1'   # opt-in (bench_t1): no latent / native-mesh copies; the Z-up mesh that the camera fit and
if not LEAN:np.savez_compressed(O/'latent_512.npz',coords=z.coords.cpu().numpy(),feats=z.feats.float().cpu().numpy())   # placement read is
m=cap['p'].decode_shape_slat(z,512)[0][0];v=m.vertices.cpu().numpy();f=m.faces.cpu().numpy()   # written from memory (byte-identical to
if not LEAN:trimesh.Trimesh(v,f,process=False).export(O/'mesh_512_native.ply')                 # bench/pixal_peak.py's conversion of the copy)
else:(O/'zup').mkdir(exist_ok=True);v32=np.asarray(v,dtype=np.float32);trimesh.Trimesh(np.stack([v32[:,0],-v32[:,2],v32[:,1]],1),f,process=False).export(O/'zup'/'mesh_512.ply')
(O/'generation.json').write_text(json.dumps(dict(method='Pixal3D native 512 (Stage 2 only)',seconds_total_incl_load=time.time()-t0,seconds_to_512=cap['t512']-t0,tokens=int(len(z.coords)),vertices=len(v),bbox=[v.min(0).tolist(),v.max(0).tolist()]),indent=1))
print('PIXAL512_DONE',len(z.coords),'tokens',round(cap['t512']-t0,1),'s')
