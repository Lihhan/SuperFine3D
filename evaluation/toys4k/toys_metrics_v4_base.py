"""TRELLIS.2 appendix D.1.1 geometric definitions, applied to generated meshes.

Implementation choices not specified by the paper: 1024px rasterization,
camera-space face normals, white background, LPIPS-Alex v0.1, deterministic
Fibonacci outer-shell cameras. Report these; do not claim exact published scores.
"""
from os import environ as _E   # install locations, set by setup/env.sh (SF3D_ROOT, SF3D_CKPT, ...)
import gc
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import torch
import trimesh

ROOT = Path(_E['SF3D_CACHE']) / 'eval'
def save_json(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)
sys.path.insert(0, str(ROOT / 'deps'))
sys.path.insert(0, f'{_E["SF3D_ROOT"]}/third_party/site-packages')
os.environ['TORCH_HOME']=str(ROOT/'torch_cache')

PROTOCOL = dict(source='https://arxiv.org/html/2512.14692#A4.SS1.SSS1',
    version='faceforward_normals_v3',
    points=1000000, outer_views=100, distance='squared Euclidean, symmetric half sum',
    md_fscore_squared_threshold=1e-8, cd_fscore_squared_threshold=1e-6,
    normal_yaw_deg=[30,120,210,300], normal_pitch_deg=30, camera_radius=10,
    fov_deg=6, raster_pixels=1024, normal_type='camera-space flat face normals, visible surface oriented toward camera',
    background='white', lpips='alex, v0.1, full image, [-1,1]',
    outer_camera_radius=2, outer_camera_fov_deg=60,
    normalization='each mesh independently centered and isotropically fit to unit cube',
    alignment='no ICP for primary paper-definition metrics',
    task='image-to-3D generated mesh vs GT; NOT a VAE reconstruction benchmark')

def normalize_mesh(mesh):
    mesh = mesh.copy()
    v = np.asarray(mesh.vertices, dtype=np.float64)
    if not len(mesh.faces) or not np.isfinite(v).all():
        raise ValueError('Empty or nonfinite mesh')
    lo, hi = v.min(0), v.max(0)
    scale = float((hi-lo).max())
    if scale <= 0:
        raise ValueError('Degenerate bounding box')
    mesh.vertices = (v-(lo+hi)/2)/scale
    return mesh

def fscore_squared(a, b, threshold):
    precision, recall = float(np.mean(a < threshold)), float(np.mean(b < threshold))
    return dict(precision=precision, recall=recall,
                fscore=2*precision*recall/max(precision+recall,1e-30))

def sampled(mesh, count, seed):
    rng = np.random.default_rng(seed)
    area = np.asarray(mesh.area_faces)
    if not np.isfinite(area).all() or area.sum() <= 0:
        raise ValueError('Invalid triangle areas')
    ids = rng.choice(len(area), count, p=area/area.sum())
    uv = rng.random((count,2)); flip = uv.sum(1)>1; uv[flip]=1-uv[flip]
    triangles = np.asarray(mesh.vertices)[np.asarray(mesh.faces)[ids]]
    return (triangles[:,0]+uv[:,0,None]*(triangles[:,1]-triangles[:,0])+
            uv[:,1,None]*(triangles[:,2]-triangles[:,0])).astype(np.float32)

def mesh_distances_squared(mesh, points):
    import open3d as o3d
    scene = o3d.t.geometry.RaycastingScene(nthreads=4)
    scene.add_triangles(o3d.core.Tensor(np.asarray(mesh.vertices,dtype=np.float32)),
                        o3d.core.Tensor(np.asarray(mesh.faces,dtype=np.uint32)))
    values = []
    for start in range(0,len(points),65536):
        distance = scene.compute_distance(o3d.core.Tensor(points[start:start+65536])).numpy()
        values.append(distance.astype(np.float64)**2)
    del scene
    return np.concatenate(values)

def camera(position, fov=6):
    position = np.asarray(position,dtype=np.float64)
    z = position/np.linalg.norm(position)
    up = np.array([0.,0.,1.])
    if abs(z@up)>.99: up=np.array([0.,1.,0.])
    x=np.cross(up,z);x/=np.linalg.norm(x); y=np.cross(z,x)
    pose=np.eye(4);pose[:3,:3]=np.stack((x,y,z),axis=1);pose[:3,3]=position
    return dict(transform_matrix=pose.tolist(),camera_angle_x=math.radians(fov))

def raster(ctx,mesh,frame,size=1024):
    """World surface points + camera normals; chunk-local face IDs avoid overflow."""
    import nvdiffrast.torch as dr
    v=torch.as_tensor(np.asarray(mesh.vertices).copy(),dtype=torch.float32,device='cuda')
    f=torch.as_tensor(np.asarray(mesh.faces).copy(),dtype=torch.int32,device='cuda')
    pose=torch.tensor(frame['transform_matrix'],dtype=torch.float32,device='cuda')
    cam=(v-pose[:3,3])@pose[:3,:3]; focal=1/math.tan(frame['camera_angle_x']/2)
    near,far=.01,100.
    clip=torch.stack((focal*cam[:,0],focal*cam[:,1],-(far+near)/(far-near)*cam[:,2]-2*far*near/(far-near),-cam[:,2]),-1)[None].contiguous()
    fn=torch.tensor(np.asarray(mesh.face_normals).copy(),dtype=torch.float32,device='cuda')@pose[:3,:3]
    depth=torch.full((size,size),float('inf'),device='cuda')
    points=torch.zeros((size,size,3),device='cuda'); normals=torch.zeros_like(points)
    for start in range(0,len(f),1000000):
        faces=f[start:start+1000000].contiguous()
        rast,_=dr.rasterize(ctx,clip,faces,resolution=[size,size])
        ids=rast[0,:,:,3].long()-1; active=(ids>=0)&(rast[0,:,:,2]<depth)
        world,_=dr.interpolate(v[None].contiguous(),rast,faces)
        normals=torch.where(active[...,None],fn[start:start+len(faces)][ids.clamp_min(0)],normals)
        points=torch.where(active[...,None],world[0],points)
        depth=torch.where(active,rast[0,:,:,2],depth)
        del rast,ids,world,faces
    mask=torch.isfinite(depth)
    pts=points[mask].cpu().numpy()
    view_rays=(points-pose[:3,3])@pose[:3,:3]
    flip=(normals*view_rays).sum(-1)>0
    normals=torch.where(flip[...,None],-normals,normals)
    normal_image=torch.where(mask[...,None],(normals+1)*.5,torch.ones_like(normals)).flip(0)
    image=normal_image.cpu().numpy()
    del v,f,clip,fn,depth,points,normals,mask,normal_image
    return pts,image

def visible_points(ctx,mesh,count=1000000):
    rng=np.random.default_rng(20261013)
    points=np.empty((0,3),np.float32); keys=np.empty(0,np.float64)
    # Random-priority reservoir is exactly uniform over all unprojected depth pixels.
    for i in range(100):
        z=1-2*(i+.5)/100; phi=i*math.pi*(3-math.sqrt(5)); radius=math.sqrt(1-z*z)
        p,_=raster(ctx,mesh,camera(2*np.array([radius*math.cos(phi),radius*math.sin(phi),z]),fov=60))
        points=np.concatenate((points,p));keys=np.concatenate((keys,rng.random(len(p))))
        if len(points)>count:
            keep=np.argpartition(keys,count-1)[:count]
            points,keys=points[keep],keys[keep]
    if not len(points): raise ValueError('No visible points')
    if len(points)<count:
        raise ValueError('Insufficient visible samples for the paper protocol')
    return points

def _evaluate_normalized(pred,gt,out):
    import nvdiffrast.torch as dr
    import lpips
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    p,g=sampled(pred,1000000,101),sampled(gt,1000000,102)
    pd=mesh_distances_squared(gt,p);gd=mesh_distances_squared(pred,g)
    result={'protocol':PROTOCOL,'MD':float((pd.mean()+gd.mean())/2),
            'MD_Fscore':fscore_squared(pd,gd,1e-8)}
    del p,g,pd,gd;gc.collect()
    ctx=dr.RasterizeCudaContext()
    p,g=visible_points(ctx,pred),visible_points(ctx,gt)
    pd=cKDTree(g).query(p,workers=4)[0]**2;gd=cKDTree(p).query(g,workers=4)[0]**2
    result.update(CD=float((pd.mean()+gd.mean())/2),CD_Fscore=fscore_squared(pd,gd,1e-6))
    del p,g,pd,gd;gc.collect()
    metric=lpips.LPIPS(net='alex',version='0.1').eval().cuda()
    psnrs=[];perceptual=[]
    from PIL import Image
    for yaw in [30,120,210,300]:
        angle,pitch=math.radians(yaw),math.radians(30)
        frame=camera(10*np.array([math.cos(pitch)*math.cos(angle),math.cos(pitch)*math.sin(angle),math.sin(pitch)]))
        _,a=raster(ctx,pred,frame);_,b=raster(ctx,gt,frame)
        Image.fromarray((a.clip(0,1)*255).astype(np.uint8)).save(out/f'pred_normal_{yaw}.png')
        Image.fromarray((b.clip(0,1)*255).astype(np.uint8)).save(out/f'GT_normal_{yaw}.png')
        psnrs.append(float(-10*np.log10(max(np.mean((a.astype(np.float64)-b)**2),1e-15))))
        ta=torch.from_numpy(a).permute(2,0,1)[None].cuda()*2-1
        tb=torch.from_numpy(b).permute(2,0,1)[None].cuda()*2-1
        with torch.no_grad(): perceptual.append(float(metric(ta,tb).item()))
        del ta,tb
    result['normal_PSNR']=float(np.mean(psnrs));result['normal_LPIPS']=float(np.mean(perceptual))
    del metric,ctx;torch.cuda.empty_cache()
    save_json(out/'paper_metrics.json',result)
    return result

def evaluate(pred,gt,out):
    """Normalize once; fit SE(3) once from geometry; re-render every aligned view."""
    from fair_alignment import fit,transform_mesh
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    pred,gt=normalize_mesh(pred),normalize_mesh(gt)
    raw=_evaluate_normalized(pred,gt,out/'raw')
    p,g=sampled(pred,100000,101),sampled(gt,100000,102)
    r,t,audit=fit(p,g)
    aligned_mesh=transform_mesh(pred,r,t)
    # No second normalization after rotation/translation: it would alter scale.
    aligned=_evaluate_normalized(aligned_mesh,gt,out/'aligned')
    aligned['protocol']=dict(PROTOCOL,alignment='single geometry-only SE(3); no scaling or mirror')
    save_json(out/'aligned'/'paper_metrics.json',aligned)
    result=dict(raw)
    result['raw']=raw;result['aligned']=aligned
    result['rigid_transform']=dict(rotation=r.tolist(),translation=t.tolist(),audit=audit)
    result['protocol']=dict(PROTOCOL,alignment='raw and aligned reported separately',
        render_after_alignment=True,per_metric_refitting=False)
    save_json(out/'paper_metrics.json',result)
    return result

def smoke_cpu():
    triangle=trimesh.Trimesh(vertices=[[0,0,0],[1,0,0],[0,1,0]],faces=[[0,1,2]],process=False)
    d=mesh_distances_squared(triangle,np.array([[.2,.2,0],[.2,.2,.001]],np.float32))
    assert abs(d[0])<1e-14 and np.isclose(d[1],1e-6,rtol=1e-5)
    assert fscore_squared(np.array([0.,1e-8]),np.array([0.,1e-8]),1e-8)['fscore']==.5
    assert normalize_mesh(triangle).extents.max()==1
    (ROOT/'metric_cpu_gate.json').write_text(json.dumps(dict(passed=True,tests=['exact point-to-triangle','squared strict threshold','unit-cube normalization'])))
    print('METRIC_CPU_GATE_PASS')

def smoke_gpu():
    import nvdiffrast.torch as dr
    import lpips
    mesh=trimesh.creation.box()
    context=dr.RasterizeCudaContext()
    points,image=raster(context,mesh,camera([5,5,5]),size=128)
    assert len(points)>0 and np.isfinite(image).all()
    assert mesh_distances_squared(mesh,points).max()<1e-10
    metric=lpips.LPIPS(net='alex',version='0.1').eval().cuda()
    x=torch.from_numpy(image).permute(2,0,1)[None].cuda()*2-1
    with torch.no_grad():score=float(metric(x,x).item())
    assert abs(score)<1e-6
    (ROOT/'metric_gpu_gate.json').write_text(json.dumps(dict(passed=True,tests=['raster world-coordinate interpolation','identical normal LPIPS zero'],lpips_self=score)))
    print('METRIC_GPU_GATE_PASS')

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--smoke-cpu',action='store_true');p.add_argument('--smoke-gpu',action='store_true');a=p.parse_args()
    if a.smoke_cpu:smoke_cpu()
    if a.smoke_gpu:smoke_gpu()
