"""Pixal3D MV input with the TRUE focal (EXIF) for every photo, all views rectified from the raw photos.
view0: photo0 rotated about its centre to look at the object (cutout bbox centre), framed like Pixal3D's single-view crop
(same object fraction), fov from the true focal, distance d0' = d0 tan(fov0/2)/tan(fov0'/2) (same world size as Pixal3D's
normalisation). Other photos: relative pose from the essential matrix on MASt3R matches (true focal), scale from the
single-view mesh depth offsets, rectified to look-at-origin cameras. usage: make_views_tf.py OUTDIR F_TRUE PHOTO0 CUT0
MATCH_k.npz:PHOTO_k:CUT_k ...   env V0 (single-view Pixal3D dir), P0POSE (photo0 PnP pose in the single-view world)"""
from os import environ as _E   # install locations, set by setup/env.sh (SF3D_ROOT, SF3D_PIXAL3D, SF3D_MAST3R, ...)
import sys,os,json,math,numpy as np,torch,trimesh,cv2
from PIL import Image
sys.path.insert(0,f'{_E["SF3D_ROOT"]}/superfine3d/detail')
from hfdetail import raster,camera
OD,FT,PH0,CU0=sys.argv[1],float(sys.argv[2]),sys.argv[3],sys.argv[4];os.makedirs(OD,exist_ok=True)
V0=os.environ['V0'];p0=json.load(open(os.environ['P0POSE']));c0=json.load(open(f'{V0}/official_camera.json'));fov0,d0=c0['camera_angle_x'],c0['distance']
R0=np.array(p0['R_cv']);C0=np.array(p0['C']);f0=p0['f_px']
ph0=np.asarray(Image.open(PH0).convert('RGB'));al0=np.asarray(Image.open(CU0).convert('RGBA'))[...,3];H,W=ph0.shape[:2];Kt=np.array([[FT,0,W/2],[0,FT,H/2],[0,0,1.]])
mi=np.asarray(Image.open(f'{V0}/actual_model_input.png').convert('RGB'));ORES=mi.shape[0];mm=mi.max(-1)>12;ys,xs=np.nonzero(mm)
frac0=max(xs.max()-xs.min(),ys.max()-ys.min())/ORES                     # object extent / image in Pixal3D's crop
def basis(z,yh=np.array([0,1.,0])):
    z=z/np.linalg.norm(z);x=np.cross(yh,z);x/=np.linalg.norm(x);y=np.cross(z,x);return np.stack([x,y,z])   # rows: cam axes (OpenCV)
ys,xs=np.nonzero(al0>127);Rv0=basis(np.linalg.inv(Kt)@np.array([(xs.min()+xs.max())/2,(ys.min()+ys.max())/2,1.]))
pts=np.stack([xs,ys,np.ones_like(xs)],0)[:,::40].astype(np.float64);q=Rv0@np.linalg.inv(Kt)@pts;q=q[:2]/q[2]
ext=max(q[0].max()-q[0].min(),q[1].max()-q[1].min())                    # object extent in normalised virtual coords
tan_half=ext/frac0/2;fov_v=2*math.atan(tan_half);d0v=d0*math.tan(fov0/2)/tan_half
def warp(ph,Kp,Rrel,fov,res,al=None):
    fv=res/2/math.tan(fov/2);Kv=np.array([[fv,0,res/2],[0,fv,res/2],[0,0,1.]]);Hm=Kv@Rrel@np.linalg.inv(Kp)
    im=cv2.warpPerspective(ph,Hm,(res,res),flags=cv2.INTER_LANCZOS4,borderValue=(0,0,0))
    if al is not None:im=np.dstack([im,cv2.warpPerspective(al,Hm,(res,res),flags=cv2.INTER_LINEAR,borderValue=0)])
    return im
Image.fromarray(warp(ph0,Kt,Rv0,fov_v,ORES,al0)).save(f'{OD}/view0.png')
A=np.array([[1,0,0],[0,0,1.],[0,-1.,0]]);b=np.array([0,-d0v,0]);PHYS=[dict(view='view0',photo=PH0,R_cv=(Rv0.T@A.T).tolist(),C=b.tolist(),f_px=FT,W=W,H=H)]           # world X = A x_v + b  (x_v: virtual view0 cam, OpenCV)
fr=[dict(file_path='view0.png',name='view0',camera_angle_x=fov_v,transform_matrix=[[1,0,0,0],[0,0,-1,-d0v],[0,1,0,0],[0,0,0,1.]])]
print('true f %.0f px (Pixal3D single-view %.0f, x%.3f) | view0 fov %.1f deg (single-view %.1f) d %.3f (single-view %.3f)'%(FT,f0,FT/f0,math.degrees(fov_v),math.degrees(fov0),d0v,d0))
# single-view mesh depth (its own world) at photo0 pixels -> per-point offsets from the object centre depth
m=trimesh.load(f'{V0}/mesh_512_native.ply',process=False);v=np.asarray(m.vertices,np.float64);v=np.stack([v[:,0],-v[:,2],v[:,1]],1)
vt=torch.tensor(v,dtype=torch.float32,device='cuda');Ft=torch.tensor(np.asarray(m.faces),dtype=torch.long,device='cuda')
K0=np.array([[f0,0,W/2],[0,f0,H/2],[0,0,1.]]);rs=1024;sc=rs/max(W,H);M=np.diag([sc/rs,sc/rs,1.])@K0@R0;Pp=np.c_[M,-M@C0]
rast=raster.rasterize(camera.clip_coords(vt,Pp),Ft,rs);tri=rast[...,3].long()-1;hit=(tri>=0).cpu().numpy()
tq=Ft[tri.clamp_min(0)];b0=rast[...,0:1].float();b1=rast[...,1:2].float();Xm=(b0*vt[tq[...,0]]+b1*vt[tq[...,1]]+(1-b0-b1)*vt[tq[...,2]]).cpu().numpy()
off=np.einsum('hwc,c->hw',Xm,R0[2])                                      # depth offset from the world origin along the photo0 axis
for i,spec in enumerate(sys.argv[5:],1):
    MZ,PH,CU=spec.split(':');Z=np.load(MZ);g=(Z['c0']>1.5)&(Z['ck']>1.5);m0,mk=Z['m0'][g],Z['mk'][g]
    ph=np.asarray(Image.open(PH).convert('RGB'));al=np.asarray(Image.open(CU).convert('RGBA'))[...,3];Hk,Wk=ph.shape[:2];Kk=np.array([[FT,0,Wk/2],[0,FT,Hk/2],[0,0,1.]])
    a=cv2.undistortPoints(m0[:,None].astype(np.float64),Kt,None)[:,0];bb=cv2.undistortPoints(mk[:,None].astype(np.float64),Kk,None)[:,0]
    E,inl=cv2.findEssentialMat(a,bb,np.eye(3),cv2.RANSAC,0.9999,1.5/FT,maxIters=20000);n,Rr,t,inl2=cv2.recoverPose(E,a,bb,np.eye(3),mask=inl.copy());kp=inl2.ravel()>0
    Xt=cv2.triangulatePoints(np.c_[np.eye(3),np.zeros(3)],np.c_[Rr,t],a[kp].T,bb[kp].T);Xt=(Xt[:3]/Xt[3]).T     # cam0 coords, |t|=1
    zv=(Rv0@Xt.T)[2]                                                     # depth along the virtual view0 axis
    ij=np.floor(m0[kp]*sc).astype(int).clip(0,rs-1);ok=hit[ij[:,1],ij[:,0]]&(zv>0)
    s=float(np.median((d0v+off[ij[ok,1],ij[ok,0]])/zv[ok]))               # object centre at depth d0v, surface offsets from the single-view mesh
    c2=-Rr.T@t.ravel()*s;R2v=Rr@Rv0.T;C2=A@(Rv0@c2)+b;R2w=R2v@A.T            # camera k: centre and world->cam rotation (OpenCV)
    dC=np.linalg.norm(C2);az=math.degrees(math.atan2(C2[0],-C2[1]));el=math.degrees(math.asin(C2[2]/dC))
    z=C2/dC;xa=np.cross([0,0,1.],z);xa/=np.linalg.norm(xa);ya=np.cross(z,xa);Rv=np.stack([xa,-ya,-z])
    fg=al>127;yy,xx=np.nonzero(fg);pp=np.stack([xx,yy,np.ones_like(xx)],0)[:,::40].astype(np.float64);ray=Rv@R2w.T@np.linalg.inv(Kk)@pp;tx=np.abs(ray[:2]/ray[2]).max()
    fov=max(2*math.atan(math.tan(fov_v/2)*d0v/dC),2*math.atan(tx/0.92))
    Image.fromarray(warp(ph,Kk,Rv@R2w.T,fov,ORES,al)).save(f'{OD}/view{i}.png')
    c2w=np.eye(4);c2w[:3,0]=xa;c2w[:3,1]=ya;c2w[:3,2]=z;c2w[:3,3]=C2
    fr.append(dict(file_path=f'view{i}.png',name=f'view{i}',camera_angle_x=fov,transform_matrix=c2w.tolist()))
    PHYS.append(dict(view=f'view{i}',photo=PH,R_cv=R2w.tolist(),C=C2.tolist(),f_px=FT,W=Wk,H=Hk))
    print('view%d (%s): E inliers %d/%d  scale %.3f  az %+.1f el %+.1f |C| %.3f (view0 %.3f) fov %.1f'%(i,os.path.basename(PH),kp.sum(),len(m0),s,az,el,dC,d0v,math.degrees(fov)))
json.dump(dict(mesh_scale=1.0,frames=fr),open(f'{OD}/transforms.json','w'),indent=1);json.dump(PHYS,open(f'{OD}/physical_poses.json','w'),indent=1);print('VIEWS_TF_DONE')
