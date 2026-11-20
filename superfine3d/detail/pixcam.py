"""Camera helpers shared by training and inference.
P maps homogeneous object-frame points [x,y,z,1] (TRELLIS voxel frame, [-0.5,0.5]^3) to homogeneous normalized
coordinates of the *conditioning image as the model sees it* (alpha-bbox square crop, then resize): (u*w, v*w, w),
u,v in [0,1], u right, v down. Blender camera: c2w transform_matrix, looks along -Z, +Y up, square image, fov=camera_angle_x."""
import numpy as np
def bbox_crop(alpha):
    ys,xs=np.nonzero(alpha);x0,y0,x1,y1=xs.min(),ys.min(),xs.max(),ys.max()
    cx,cy=(x0+x1)/2,(y0+y1)/2;h=max(x1-x0,y1-y0)/2
    return int(cx-h),int(cy-h),int(cx+h),int(cy+h)   # identical integer box to ImageConditionedMixin
def frame_P(frame,image,out=None):
    """frame: transforms.json entry; image: PIL RGBA (full render). Returns P (3x4) and (optionally) the cropped alpha at `out` px."""
    W=image.size[0];alpha=np.array(image.getchannel(3))
    box=bbox_crop(alpha);c2w=np.array(frame['transform_matrix'],dtype=np.float64);w2c=np.linalg.inv(c2w)
    f=0.5/np.tan(frame['camera_angle_x']/2)          # focal in units of image width
    # camera (x right, y up, looking -z) -> full-image normalized (u right, v down)
    K=np.array([[f,0,0.5],[0,-f,0.5],[0,0,1.]])@np.diag([1,1,-1.])
    full=K@w2c[:3]
    s=W/(box[2]-box[0]);A=np.array([[s,0,-box[0]/(box[2]-box[0])],[0,s,-box[1]/(box[3]-box[1])],[0,0,1.]])
    P=A@full
    if out is None:return P
    a=image.crop(box).resize((out,out)).getchannel(3);return P,np.array(a)
def project(P,pts):
    h=np.c_[pts,np.ones(len(pts))]@P.T;w=h[:,2:3];uv=h[:,:2]/w
    return uv,(w[:,0]>0)&(uv>=0).all(1)&(uv<1).all(1)

def preprocess_box(alpha):
    """Crop box of Trellis2ImageTo3DPipeline.preprocess_image (alpha > 0.8, centre +- size//2)."""
    yx=np.argwhere(alpha>0.8*255);x0,y0=yx[:,1].min(),yx[:,0].min();x1,y1=yx[:,1].max(),yx[:,0].max()
    cx,cy=(x0+x1)/2,(y0+y1)/2;size=int(max(x1-x0,y1-y0));return (cx-size//2,cy-size//2,cx+size//2,cy+size//2)
def box_P(frame,W,box):
    """Camera P for an arbitrary crop box (float pixels) of a W x W Blender render."""
    c2w=np.array(frame['transform_matrix'],dtype=np.float64);w2c=np.linalg.inv(c2w);f=0.5/np.tan(frame['camera_angle_x']/2)
    K=np.array([[f,0,0.5],[0,-f,0.5],[0,0,1.]])@np.diag([1,1,-1.]);bw=box[2]-box[0];bh=box[3]-box[1]
    A=np.array([[W/bw,0,-box[0]/bw],[0,W/bh,-box[1]/bh],[0,0,1.]]);return A@K@w2c[:3]
