"""Pose of a raw photo in the Pixal3D world of a single-view run (view0): SIFT matches between view0's model input and the
photo, 3D points from the view0 mesh rendered at view0's camera, PnP-RANSAC with unknown focal (principal point at the photo
centre), refined by least squares. Optionally writes a rectified view: the photo warped (pure rotation homography) to a camera
at the same centre looking at the world origin, Z up, framed like view0 -> exact input for Pixal3D MV.
usage: pnp_view.py VIEW0_DIR PHOTO OUT_PREFIX [OUT_RES]"""
from os import environ as _E   # install locations, set by setup/env.sh (SF3D_ROOT, SF3D_PIXAL3D, SF3D_MAST3R, ...)
import sys,json,math,numpy as np,torch,trimesh,cv2
from PIL import Image
sys.path.insert(0,f'{_E["SF3D_ROOT"]}/superfine3d/detail')
from hfdetail import raster,camera
from scipy.optimize import least_squares
V0,PH,OUT=sys.argv[1:4];ORES=int(sys.argv[4]) if len(sys.argv)>4 else 1110
cam0=json.load(open(f'{V0}/official_camera.json'));fov0,d0=cam0['camera_angle_x'],cam0['distance']
I0=np.asarray(Image.open(f'{V0}/actual_model_input.png').convert('RGB'));R0=I0.shape[0]
m=trimesh.load(f'{V0}/mesh_512_native.ply',process=False);v=np.asarray(m.vertices,np.float64);v=np.stack([v[:,0],-v[:,2],v[:,1]],1)
vt=torch.tensor(v,dtype=torch.float32,device='cuda');Ft=torch.tensor(np.asarray(m.faces),dtype=torch.long,device='cuda')
C0=np.array([0,-d0,0.]);r0=np.array([1,0,0.]);u0=np.array([0,0,1.]);f0=np.array([0,1,0.]);fn0=.5/math.tan(fov0/2)
P0=np.c_[np.stack([fn0*r0+.5*f0,-fn0*u0+.5*f0,f0]),-np.stack([fn0*r0+.5*f0,-fn0*u0+.5*f0,f0])@C0]
rast=raster.rasterize(camera.clip_coords(vt,P0),Ft,R0);tri=rast[...,3].long()-1;hit=tri>=0
t=Ft[tri.clamp_min(0)];b0=rast[...,0:1].float();b1=rast[...,1:2].float()
X=(b0*vt[t[...,0]]+b1*vt[t[...,1]]+(1-b0-b1)*vt[t[...,2]]);X=X.cpu().numpy();hit=hit.cpu().numpy()
dep=np.where(hit,(X-C0)@f0,np.nan);gy,gx=np.gradient(np.nan_to_num(dep,nan=9));edge=(np.hypot(gx,gy)>0.01)   # depth discontinuities (object units / px)
ok3=hit&~cv2.dilate(edge.astype(np.uint8),np.ones((5,5),np.uint8)).astype(bool)
ph=np.asarray(Image.open(PH).convert('RGB'));H,W=ph.shape[:2];sc=1600/max(H,W);phs=cv2.resize(ph,(int(W*sc),int(H*sc)),interpolation=cv2.INTER_AREA)
sift=cv2.SIFT_create(nfeatures=40000,contrastThreshold=0.01)
k0,d0_=sift.detectAndCompute(cv2.cvtColor(I0,cv2.COLOR_RGB2GRAY),(I0.max(-1)>12).astype(np.uint8)*255);k1,d1_=sift.detectAndCompute(cv2.cvtColor(phs,cv2.COLOR_RGB2GRAY),None)
bf=cv2.BFMatcher();mm=bf.knnMatch(d0_,d1_,k=2);mr=bf.knnMatch(d1_,d0_,k=1)
good=[a for a,b in mm if a.distance<.8*b.distance and mr[a.trainIdx][0].trainIdx==a.queryIdx]
p0=np.float32([k0[g.queryIdx].pt for g in good]);p1=np.float32([k1[g.trainIdx].pt for g in good])/sc
Fm,inl=cv2.findFundamentalMat(p0,p1,cv2.FM_RANSAC,3.0/sc*1.0,0.999);inl=inl.ravel().astype(bool);p0,p1=p0[inl],p1[inl]
ij=np.round(p0).astype(int).clip(0,R0-1);sel=ok3[ij[:,1],ij[:,0]];P3=X[ij[sel,1],ij[sel,0]].astype(np.float64);p2=p1[sel].astype(np.float64)
print('matches %d  F-inliers %d  with 3D %d'%(len(good),inl.sum(),len(P3)))
best=None
for fpx in np.geomspace(.5*max(H,W),6*max(H,W),28):
    K=np.array([[fpx,0,W/2],[0,fpx,H/2],[0,0,1.]]);ok,rv,tv,il=cv2.solvePnPRansac(P3,p2,K,None,reprojectionError=6.0,iterationsCount=3000,confidence=.999,flags=cv2.SOLVEPNP_EPNP)
    if ok and il is not None and (best is None or len(il)>best[0]):best=(len(il),fpx,rv,tv,il.ravel())
n,fpx,rv,tv,il=best
def resid(x,P,p):
    K=np.array([[x[6],0,W/2],[0,x[6],H/2],[0,0,1.]]);q,_=cv2.projectPoints(P,x[:3],x[3:6],K,None);return (q[:,0]-p).ravel()
x=np.r_[rv.ravel(),tv.ravel(),fpx];P3i,p2i=P3[il],p2[il]
for _ in range(3):
    r=least_squares(resid,x,args=(P3i,p2i),loss='huber',f_scale=3.0);x=r.x
    e=np.linalg.norm(resid(x,P3,p2).reshape(-1,2),axis=1);keep=e<8;P3i,p2i=P3[keep],p2[keep]
e=np.linalg.norm(resid(x,P3i,p2i).reshape(-1,2),axis=1)
Rcv,_=cv2.Rodrigues(x[:3]);C=-Rcv.T@x[3:6];fpx=x[6]
az=math.degrees(math.atan2(C[0],-C[1]));el=math.degrees(math.asin(C[2]/np.linalg.norm(C)))
print('PnP inliers %d (of %d)  median reproj %.2f px (photo %dx%d)  f %.0f px (hfov %.1f deg)  centre %s |C| %.3f  az %+.1f el %+.1f'%(len(P3i),len(P3),np.median(e),W,H,fpx,math.degrees(2*math.atan(W/2/fpx)),np.round(C,3),np.linalg.norm(C),az,el))
# rectified view: camera at C looking at the origin, world Z up, fov chosen so the object is framed as in view0
dC=np.linalg.norm(C);fov=2*math.atan(math.tan(fov0/2)*d0/dC);z=C/dC;xa=np.cross([0,0,1.],z);xa/=np.linalg.norm(xa);ya=np.cross(z,xa)
c2w=np.eye(4);c2w[:3,0]=xa;c2w[:3,1]=ya;c2w[:3,2]=z;c2w[:3,3]=C
Rv=np.stack([xa,-ya,-z]);fv=ORES/2/math.tan(fov/2);Kv=np.array([[fv,0,ORES/2],[0,fv,ORES/2],[0,0,1.]]);K1=np.array([[fpx,0,W/2],[0,fpx,H/2],[0,0,1.]])
Hm=Kv@Rv@Rcv.T@np.linalg.inv(K1);rect=cv2.warpPerspective(ph,Hm,(ORES,ORES),flags=cv2.INTER_LANCZOS4,borderValue=(0,0,0))
Image.fromarray(rect).save(OUT+'_rect.png')
json.dump(dict(photo=PH,f_px=fpx,W=W,H=H,R_cv=Rcv.tolist(),t_cv=x[3:6].tolist(),C=C.tolist(),az=az,el=el,inliers=int(len(P3i)),median_px=float(np.median(e)),
               rect=dict(camera_angle_x=fov,transform_matrix=c2w.tolist(),res=ORES)),open(OUT+'_pose.json','w'),indent=1)
# overlay check: view0 mesh silhouette at the rectified camera over the rectified photo
fn=.5/math.tan(fov/2);fw=-z;Pv=np.c_[np.stack([fn*xa+.5*fw,-fn*ya+.5*fw,fw]),-np.stack([fn*xa+.5*fw,-fn*ya+.5*fw,fw])@C]
s=(raster.rasterize(camera.clip_coords(vt,Pv),Ft,ORES)[...,3]>0).cpu().numpy();mk=rect.max(-1)>15
print('view0-mesh silhouette vs rectified photo IoU %.3f'%((s&mk).sum()/(s|mk).sum()))
ov=rect.copy();edge=cv2.morphologyEx(s.astype(np.uint8),cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8)).astype(bool);ov[edge]=(0,255,0);Image.fromarray(ov).save(OUT+'_overlay.jpg',quality=85)
print('PNP_DONE')
