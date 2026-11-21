"""Versioned custom image-to-3D evaluation, NOT TRELLIS reconstruction replication."""
from os import environ as _E   # install locations, set by setup/env.sh (SF3D_ROOT, SF3D_CKPT, ...)
import os,sys,json,time,hashlib,math,gc,ast
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial import cKDTree
R=Path(f'{_E["SF3D_TOYS4K_EVAL"]}')
sys.path[:0]=[f'{_E["SF3D_ROOT"]}/evaluation/toys4k',f'{_E["SF3D_ROOT"]}/third_party/site-packages',f'{_E["SF3D_ROOT"]}/third_party/site-packages']
import toys_metrics_v4_base as base
from fair_alignment import validate,transform_mesh
VERSION='generation_sphere_normalized_v4'
METHODS=['TRELLIS2','Hunyuan21','Pixal3D','Ours2048','Ours4096']
PROTOCOL={'version':VERSION,'task':'single-image generation against original GT; not paper reconstruction replication','normalization':'exact area-weighted surface centroid; divide by twice maximum vertex distance to centroid, independently per mesh; rotation-invariant unit bounding sphere','alignment':'identical final-mesh proper rigid fit for all five methods; no fitted scale or mirror; raw geometry also retained','points':1000000,'outer_views':100,'MD_Fscore_squared_threshold':1e-8,'CD_Fscore_squared_threshold':1e-6,'distance':'symmetric halfmean squared Euclidean','normal':'camera-space flat face normals oriented toward camera, white background, four views1024px, LPIPS Alex v0.1','camera':'yaw30/120/210/300 pitch30 FOV6; one common radius per asset covers all five aligned results plus GT with5percent angular margin, minimum10','aggregation':'paired complete five-method cohorts, include/excludeTC001 separately, retain frozen200 denominator and failures','geometry_metrics_recomputed':True,'old_scores_comparable':False,'limitation':'sphere scale sensitive to outliers; geometric alignment does not guarantee semantic correspondence; normals at1024px not full4096 detail metric'}
def save(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def normalize(m):
 v=np.asarray(m.vertices);f=np.asarray(m.faces);total=0.;center=np.zeros(3)
 assert len(f) and np.isfinite(v).all()
 for i in range(0,len(f),250000):
  t=v[f[i:i+250000]];a=np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=1)*.5
  total+=a.sum();center+=(a[:,None]*t.mean(1)).sum(0)
 assert total>0;center/=total
 radius=0.
 for i in range(0,len(v),1000000):radius=max(radius,float(np.linalg.norm(v[i:i+1000000]-center,axis=1).max()))
 assert radius>0;m.vertices=(v-center)/(2*radius)
 return m,dict(center=center.tolist(),scale=2*radius,area=total)
def fit(p,g):
 # Geometry-only rotation hypotheses: world-axis and PCA-frame proper rotations.
 import itertools
 p=p[np.linspace(0,len(p)-1,min(10000,len(p)),dtype=int)];g=g[np.linspace(0,len(g)-1,min(10000,len(g)),dtype=int)]
 tree=cKDTree(g)
 def score(q):return float((np.mean(tree.query(q,workers=4)[0]**2)+np.mean(cKDTree(q).query(g,workers=4)[0]**2))/2)
 _,ep=np.linalg.eigh(np.cov(p.T));_,eg=np.linalg.eigh(np.cov(g.T));rots=[]
 for perm in itertools.permutations(range(3)):
  for signs in itertools.product((-1,1),repeat=3):
   s=np.eye(3)[list(perm)]*np.array(signs)[:,None]
   for r in (s,ep@s@eg.T):
    if np.linalg.det(r)>.999999:rots.append(r)
 starts=[]
 for idx,r in enumerate(rots):
  t=g.mean(0)-p.mean(0)@r;starts.append((score(p@r+t),idx,t))
 results=[]
 for _,idx,t in sorted(starts,key=lambda x:(x[0],x[1]))[:8]:
  r=rots[idx].copy()
  for iteration in range(80):
   q=p@r+t;_,nn=tree.query(q,workers=4);dest=g[nn];qa=q.mean(0);ga=dest.mean(0)
   u,_,vt=np.linalg.svd((q-qa).T@(dest-ga));d=np.eye(3);d[-1,-1]=np.linalg.det(u@vt);dr=u@d@vt;dt=ga-qa@dr;r,t=r@dr,t@dr+dt
   if np.linalg.norm(dr-np.eye(3))+np.linalg.norm(dt)<1e-8:break
  validate(r,t);results.append((score(p@r+t),r,t))
 results.sort(key=lambda x:x[0]);cost,r,t=results[0]
 return r,t,{'objective':cost,'candidate_costs':[x[0] for x in results],'determinant':float(np.linalg.det(r)),'scale_fit':False,'image_metrics_used':False,'global_optimum_guaranteed':False}
def load(p):return trimesh.load(p,process=False,force='mesh')
def sources(key):
 d=R/key;prep=json.loads((d/'input_data/prepare.json').read_text());inp=sha(d/'input_data/input.png');gt=sha(d/'input_data/GT_normalized.ply')
 assert inp==prep['input_sha256'] and gt==prep['GT_sha256'];out={}
 for method in METHODS:
  if method.startswith('Ours'):
   res=int(method[4:]);record=d/f'metrics_{res}/result.json';path=d/f'ours/mesh_{res}.ply';hkey='pred_mesh_sha256'
  else:
   bd=d/'baselines'/method;record=bd/'metrics/result.json';gen=json.loads((bd/'generation.json').read_text());res=gen['actual_resolution'];path=bd/f'mesh_{res}.ply';hkey='mesh_sha256'
  v=json.loads(record.read_text());assert v['complete'] and v['input_sha256']==inp and v['GT_sha256']==gt
  mh=sha(path);assert mh==v[hkey];out[method]={'path':str(path),'sha256':mh,'actual_resolution':res}
 return out,inp,gt
def frames(radius):
 for yaw in [30,120,210,300]:
  a,b=math.radians(yaw),math.radians(30)
  yield yaw,base.camera(radius*np.array([math.cos(b)*math.cos(a),math.cos(b)*math.sin(a),math.sin(b)]),fov=6)
def clipping(m,radius):
 answer={}
 for yaw,fr in frames(radius):
  pose=np.array(fr['transform_matrix']);outside=0;maximum=0.
  for i in range(0,len(m.vertices),1000000):
   c=(np.asarray(m.vertices)[i:i+1000000]-pose[:3,3])@pose[:3,:3];xy=c[:,:2]/(-c[:,2,None]*math.tan(math.radians(3)))
   outside+=int((np.abs(xy).max(1)>1).sum());maximum=max(maximum,float(np.abs(xy).max()))
  answer[str(yaw)]={'outside_vertices':outside,'max_abs_ndc':maximum}
 return answer
def prepare(key):
 dest=R/key/'metrics_v4';src,inp,gh=sources(key)
 if (dest/'preparation.json').exists():
  old=json.loads((dest/'preparation.json').read_text())
  if old['sources']==src and old['GT_sha256']==gh and old['input_sha256']==inp:return
  raise RuntimeError('Immutable v4 inputs changed')
 gt,gn=normalize(load(R/key/'input_data/GT_normalized.ply'));g=base.sampled(gt,100000,102);bounds=.5;align={}
 for method,s in src.items():
  m,n=normalize(load(s['path']));p=base.sampled(m,100000,101);r,t,a=fit(p,g);m=transform_mesh(m,r,t)
  for i in range(0,len(m.vertices),1000000):bounds=max(bounds,float(np.linalg.norm(np.asarray(m.vertices)[i:i+1000000],axis=1).max()))
  align[method]={'rotation':r.tolist(),'translation':t.tolist(),'normalization':n,'audit':a};del m,p;gc.collect()
 radius=max(10.,1.05*bounds/math.sin(math.radians(3)))
 save(dest/'preparation.json',{'version':VERSION,'sources':src,'input_sha256':inp,'GT_sha256':gh,'GT_normalization':gn,'alignment':align,'camera_radius':radius,'protocol':PROTOCOL,'complete':True})
def geometry(ctx,pred,gt):
 p,g=base.sampled(pred,1000000,101),base.sampled(gt,1000000,102)
 pd,gd=base.mesh_distances_squared(gt,p),base.mesh_distances_squared(pred,g)
 d={'MD':float((pd.mean()+gd.mean())/2),'MD_Fscore':base.fscore_squared(pd,gd,1e-8)}
 del p,g,pd,gd;gc.collect()
 p,g=base.visible_points(ctx,pred),base.visible_points(ctx,gt)
 pd=cKDTree(g).query(p,workers=4)[0]**2;gd=cKDTree(p).query(g,workers=4)[0]**2
 d.update(CD=float((pd.mean()+gd.mean())/2),CD_Fscore=base.fscore_squared(pd,gd,1e-6));return d
def score(key,method):
 import torch,nvdiffrast.torch as dr,lpips
 from PIL import Image
 torch.set_num_threads(4);torch.set_grad_enabled(False);torch.cuda.set_per_process_memory_fraction(.45)
 dest=R/key/'metrics_v4';info=json.loads((dest/'preparation.json').read_text());s=info['sources'][method]
 assert sha(s['path'])==s['sha256'] and sha(R/key/'input_data/GT_normalized.ply')==info['GT_sha256'] and sha(R/key/'input_data/input.png')==info['input_sha256']
 out=dest/method;out.mkdir(exist_ok=True);gt,_=normalize(load(R/key/'input_data/GT_normalized.ply'));m,_=normalize(load(s['path']));ctx=dr.RasterizeCudaContext()
 rawpath=out/'raw_geometry.json'
 if rawpath.exists():raw=json.loads(rawpath.read_text())
 else:raw=geometry(ctx,m,gt);save(rawpath,raw)
 a=info['alignment'][method];m=transform_mesh(m,np.array(a['rotation']),np.array(a['translation']));radius=info['camera_radius']
 clip={'GT':clipping(gt,radius),'prediction':clipping(m,radius)};assert all(v['outside_vertices']==0 for group in clip.values() for v in group.values())
 gp=out/'registered_geometry.json'
 if gp.exists():aligned=json.loads(gp.read_text())
 else:aligned=geometry(ctx,m,gt);save(gp,aligned)
 metric=lpips.LPIPS(net='alex',version='0.1').eval().cuda();psnrs=[];per=[]
 for yaw,frame in frames(radius):
  _,x=base.raster(ctx,m,frame);_,y=base.raster(ctx,gt,frame);assert np.isfinite(x).all() and np.isfinite(y).all()
  for label,img in [('pred',x),('GT',y)]:Image.fromarray((img.clip(0,1)*255).astype(np.uint8)).save(out/f'{label}_normal_{yaw}.png')
  psnrs.append(float(-10*np.log10(max(np.mean((x.astype(np.float64)-y)**2),1e-15))))
  tx=torch.from_numpy(x).permute(2,0,1)[None].cuda()*2-1;ty=torch.from_numpy(y).permute(2,0,1)[None].cuda()*2-1
  per.append(float(metric(tx,ty).item()));del tx,ty
 aligned.update(normal_PSNR=float(np.mean(psnrs)),normal_LPIPS=float(np.mean(per)))
 result={'complete':True,'version':VERSION,'asset':key,'method':method,'actual_resolution':s['actual_resolution'],'mesh_sha256':s['sha256'],'GT_sha256':info['GT_sha256'],'input_sha256':info['input_sha256'],'records':{'raw_geometry':raw,'registered':aligned},'alignment':a,'camera_radius':radius,'clipping':clip,'protocol':PROTOCOL,'geometry_metrics_recomputed':True,'time':time.time()}
 save(out/'result.json',result);print('V4_COMPLETE',key,method,flush=True)
def gates(gpu=False):
 out=R/'evaluation_protocol_v4'
 if not gpu:
  mesh=trimesh.creation.icosphere(subdivisions=2);v=np.asarray(mesh.vertices).copy();v[:,0]*=1+.3*v[:,2];v[:,1]*=.6;v[:,2]*=.4;mesh.vertices=v
  q=trimesh.transformations.rotation_matrix(.713,[.3,.7,.2])[:3,:3];changed=mesh.copy();changed.vertices=np.asarray(mesh.vertices)@q+np.array([2.,-3.,4.])
  m,_=normalize(mesh.copy());n,_=normalize(changed);exact=np.max(np.linalg.norm(np.asarray(n.vertices)@q.T-np.asarray(m.vertices),axis=1));assert exact<1e-10
  p=base.sampled(m,100000,17);g=base.sampled(n,100000,17);r,t,a=fit(p,g);err=float(np.max(np.linalg.norm(p@r+t-g,axis=1)));assert err<1e-5,(err,a)
  ds=base.mesh_distances_squared(m,p);assert ds.max()<1e-10 and base.fscore_squared(ds,ds,1e-8)['fscore']==1
  scaled=mesh.copy();scaled.vertices=np.asarray(mesh.vertices)*7.;scaled,_=normalize(scaled);assert np.max(np.abs(scaled.vertices-m.vertices))<1e-10
  save(out/'cpu_gate.json',{'passed':True,'rotation_translation_normalization_max_error':float(exact),'uniform_scale_normalization_pass':True,'rigid_fit_correspondence_max_error':err,'self_MD':float(ds.mean()),'self_Fscore':1.,'source_sha256':sha(__file__)})
 else:
  import torch,nvdiffrast.torch as dr,lpips
  torch.set_grad_enabled(False);ctx=dr.RasterizeCudaContext();m,_=normalize(trimesh.creation.icosphere(subdivisions=2));rev=m.copy();rev.faces=np.asarray(rev.faces)[:,::-1];frame=next(frames(10))[1]
  _,x=base.raster(ctx,m,frame,256);_,y=base.raster(ctx,rev,frame,256);err=float(np.max(np.abs(x-y)));assert err<1e-5,err
  metric=lpips.LPIPS(net='alex',version='0.1').eval().cuda();z=torch.from_numpy(x).permute(2,0,1)[None].cuda()*2-1;zero=float(metric(z,z).item());assert abs(zero)<1e-6
  # Deterministic visible-shell self comparison exercises the actual100-view path.
  p=base.visible_points(ctx,m);g=base.visible_points(ctx,m);assert np.array_equal(p,g)
  save(out/'gpu_gate.json',{'passed':True,'winding_image_max_error':err,'LPIPS_self':zero,'visible_shell_self_exact':True,'source_sha256':sha(__file__)})
if __name__=='__main__':
 action=sys.argv[1]
 if action=='cpu_gate':gates()
 elif action=='gpu_gate':gates(True)
 elif action=='prepare':prepare(sys.argv[2])
 elif action=='score':score(sys.argv[2],sys.argv[3])
