"""Exact camera for the 4096 chain of a true-focal MV start: input.png = photo0 rectified (rotation homography, true focal)
to the virtual view0 camera of make_views_tf.py at RES px, cam_fit.json / cam_refined.json in the chain's format
(canonical front camera at (0,-d0',0), fov0'). usage: chain_cam.py CHAIN_DIR VIEWS_DIR F_TRUE PHOTO0 CUT0 [RES]"""
from os import environ as _E   # install locations, set by setup/env.sh (SF3D_ROOT, SF3D_PIXAL3D, SF3D_MAST3R, ...)
import sys,json,math,numpy as np,torch,cv2,trimesh
from PIL import Image
sys.path.insert(0,f'{_E["SF3D_ROOT"]}/superfine3d/detail')
from hfdetail import raster,camera
D,VD,FT,PH0,CU0=sys.argv[1],sys.argv[2],float(sys.argv[3]),sys.argv[4],sys.argv[5];RES=int(sys.argv[6]) if len(sys.argv)>6 else 2048
T=json.load(open(f'{VD}/transforms.json'))['frames'][0];fov=T['camera_angle_x'];d=-T['transform_matrix'][1][3]
ph=np.asarray(Image.open(PH0).convert('RGB'));al=np.asarray(Image.open(CU0).convert('RGBA'))[...,3];H,W=ph.shape[:2];Kt=np.array([[FT,0,W/2],[0,FT,H/2],[0,0,1.]])
ys,xs=np.nonzero(al>127);z=np.linalg.inv(Kt)@np.array([(xs.min()+xs.max())/2,(ys.min()+ys.max())/2,1.]);z/=np.linalg.norm(z)
x=np.cross([0,1.,0],z);x/=np.linalg.norm(x);y=np.cross(z,x);Rv0=np.stack([x,y,z])                       # same virtual camera as make_views_tf.py
fv=RES/2/math.tan(fov/2);Kv=np.array([[fv,0,RES/2],[0,fv,RES/2],[0,0,1.]]);Hm=Kv@Rv0@np.linalg.inv(Kt)
rgb=cv2.warpPerspective(ph,Hm,(RES,RES),flags=cv2.INTER_LANCZOS4,borderValue=(0,0,0));a=cv2.warpPerspective(al,Hm,(RES,RES),flags=cv2.INTER_LINEAR,borderValue=0)
Image.fromarray(np.dstack([rgb,a])).save(f'{D}/input.png')
# check against the rectified MV view0 (same camera, lower res)
v0=np.asarray(Image.open(f'{VD}/view0.png').convert('RGBA').resize((RES,RES),Image.Resampling.BILINEAR))[...,3]>127
eye=np.array([0,-d,0.]);r=np.array([1,0,0.]);u=np.array([0,0,1.]);fw=np.array([0,1,0.]);f=.5/math.tan(fov/2)
P=camera.make_P(eye,r,u,fw,f)
m=trimesh.load(f'{D}/mesh_512.ply',process=False);vt=torch.tensor(np.asarray(m.vertices),dtype=torch.float32,device='cuda');Ft=torch.tensor(np.asarray(m.faces),dtype=torch.long,device='cuda')
sil=(raster.rasterize(camera.clip_coords(vt,P.tolist()),Ft,RES)[...,3]>0).cpu().numpy();ma=a>127
iou=float((sil&ma).sum()/(sil|ma).sum())
json.dump(dict(box=[0.,0.,float(RES),float(RES)],size=[RES,RES],note='exact camera of the MV world (chain_cam.py), no fit'),open(f'{D}/cam_fit.json','w'),indent=1)
json.dump(dict(P=P.tolist(),r=r.tolist(),u=u.tolist(),fwd=fw.tolist(),yaw=270.,pitch=0.,roll=0.,fov=math.degrees(fov),iou=iou,ncc=0.,box=[0.,0.,float(RES),float(RES)],size=[RES,RES],
               note='exact camera of the MV world (chain_cam.py), no fit'),open(f'{D}/cam_refined.json','w'),indent=1)
print('%s: input %dpx fov %.1f d %.3f | mesh silhouette IoU %.3f | alpha agreement with MV view0 %.3f'%(D.split('/')[-1],RES,math.degrees(fov),d,iou,(v0&ma).sum()/(v0|ma).sum()));print('CHAIN_CAM_DONE')
