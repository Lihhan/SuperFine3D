"""Pixal3D multi-view mode, stopped after the 512 shape stage (two or more posed views). Weights: SF3D_PIXAL3D_MV_WEIGHTS (default
$SF3D_PIXAL3D/weights; pipeline_mv.json; only the SS and 512 multi-view flow models are needed). usage: pixal512mv_worker.py VIEWS_DIR OUT"""
from os import environ as _E   # install locations, set by setup/env.sh (SF3D_ROOT, SF3D_PIXAL3D, SF3D_MAST3R, ...)
import os,sys,json,time,types
from pathlib import Path
VD,O=sys.argv[1],Path(sys.argv[2]);O.mkdir(parents=True,exist_ok=True);base=Path(f'{_E["SF3D_PIXAL3D"]}');sys.path.insert(0,str(base/'repo'))
import torch,numpy as np,trimesh
import inference_mv as api
from pixal3d.pipelines.pixal3d_mv_image_to_3d import Pixal3DMVImageTo3DPipeline as Pipeline
os.environ['FLEX_GEMM_AUTOTUNE_CACHE_PATH']=f'{_E["SF3D_CACHE"]}/pixal_autotune.json'   # as the single-view worker
torch.set_num_threads(4);torch.set_grad_enabled(False);torch.cuda.set_per_process_memory_fraction(.45)
Pipeline.model_names_to_load=['sparse_structure_flow_model','sparse_structure_decoder','shape_slat_flow_model_512','shape_slat_decoder']
oi=api.init_pipeline;cap={}
class Done(Exception):pass
def init(*a,**kw):
 p=oi(*a,**kw);cap['p']=p
 for k in ('shape_slat_flow_model_1024','tex_slat_flow_model_1024'):p.models.setdefault(k,None)
 old=p.sample_shape_slat
 def sss(self,*a,**kw):
  z=old(*a,**kw);cap['z']=z;raise Done()
 p.sample_shape_slat=types.MethodType(sss,p);return p
api.init_pipeline=init;t0=time.time()
try:api.run_inference(VD,str(O/'unused.glb'),seed=int(os.environ.get('SEED','42')),model_path=_E.get('SF3D_PIXAL3D_MV_WEIGHTS') or str(base/'weights'),low_vram=True,resolution=1024)
except Done:pass
z=cap['z'];np.savez_compressed(O/'latent_512.npz',coords=z.coords.cpu().numpy(),feats=z.feats.float().cpu().numpy())
m=cap['p'].decode_shape_slat(z,512)[0][0];v=m.vertices.cpu().numpy();f=m.faces.cpu().numpy();trimesh.Trimesh(v,f,process=False).export(O/'mesh_512_native.ply')
(O/'generation.json').write_text(json.dumps(dict(method='Pixal3D multi-view, 512 stage only',views=VD,seconds=time.time()-t0,tokens=int(len(z.coords)),bbox=[v.min(0).tolist(),v.max(0).tolist()]),indent=1))
print('PIXAL512MV_DONE',len(z.coords),'tokens',round(time.time()-t0,1),'s')
