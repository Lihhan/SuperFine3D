"""MASt3R dense matches between photo0 and photoK (object crops, 512 long side) + photo0 pointmap at the matches.
Writes OUT.npz: m0, mk (original photo pixels, x,y), X0 (MASt3R points of the photo0 matches, photo0 camera frame, OpenCV),
c0, ck (confidences), f0_m (focal of the photo0 pointmap, original px).
usage: mast3r_match.py PHOTO0 CUT0 PHOTOK CUTK OUT.npz"""
from os import environ as _E   # install locations, set by setup/env.sh (SF3D_ROOT, SF3D_PIXAL3D, SF3D_MAST3R, ...)
import sys,numpy as np,torch
sys.path.insert(0,f'{_E["SF3D_MAST3R"]}');import mast3r.utils.path_to_dust3r  # noqa   (github.com/naver/mast3r, with dust3r / croco submodules)
from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs
from dust3r.inference import inference
from dust3r.utils.image import ImgNorm
from PIL import Image
P0,C0,PK,CK,OUT=sys.argv[1:6];torch.set_num_threads(4);dev='cuda'
def prep(ph,cut,i):
    im=Image.open(ph).convert('RGB');a=np.asarray(Image.open(cut).convert('RGBA'))[...,3]>127;W,H=im.size
    ys,xs=np.nonzero(a);x0,x1,y0,y1=xs.min(),xs.max(),ys.min(),ys.max();cx,cy=(x0+x1)/2,(y0+y1)/2;w,h=(x1-x0)*1.12,(y1-y0)*1.12
    if w>=h:w=max(w,h*4/3);h=w*3/4
    else:h=max(h,w*4/3);w=h*3/4
    w,h=min(w,W),min(h,H);cx=min(max(cx,w/2),W-w/2);cy=min(max(cy,h/2),H-h/2)
    box=(cx-w/2,cy-h/2,cx+w/2,cy+h/2);s=512/max(w,h);tw,th=int(round(w*s))//16*16,int(round(h*s))//16*16
    cr=im.crop(tuple(int(round(b)) for b in box)).resize((tw,th),Image.Resampling.LANCZOS)
    sx=(int(round(box[2]))-int(round(box[0])))/tw;sy=(int(round(box[3]))-int(round(box[1])))/th
    al=np.asarray(Image.fromarray(a.astype(np.uint8)*255).crop(tuple(int(round(b)) for b in box)).resize((tw,th),Image.Resampling.BILINEAR))>127
    v=dict(img=ImgNorm(cr)[None],true_shape=np.int32([[th,tw]]),idx=i,instance=str(i))
    return v,(int(round(box[0])),int(round(box[1])),sx,sy),al
v0,t0,a0=prep(P0,C0,0);vk,tk,ak=prep(PK,CK,1)
model=AsymmetricMASt3R.from_pretrained('naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric').to(dev).eval()
with torch.no_grad():out=inference([(v0,vk)],model,dev,batch_size=1,verbose=False)
p1,p2=out['pred1'],out['pred2'];d1,d2=p1['desc'][0].detach(),p2['desc'][0].detach()
xy0,xyk=fast_reciprocal_NNs(d1,d2,subsample_or_initxy1=4,device=dev,dist='dot',block_size=2**13)
H0,W0=v0['true_shape'][0];Hk,Wk=vk['true_shape'][0]
ok=(xy0[:,0]>=3)&(xy0[:,0]<W0-3)&(xy0[:,1]>=3)&(xy0[:,1]<H0-3)&(xyk[:,0]>=3)&(xyk[:,0]<Wk-3)&(xyk[:,1]>=3)&(xyk[:,1]<Hk-3)
xy0,xyk=xy0[ok],xyk[ok];fg=a0[xy0[:,1],xy0[:,0]]&ak[xyk[:,1],xyk[:,0]];xy0,xyk=xy0[fg],xyk[fg]
pts=p1['pts3d'][0].cpu().numpy();c1=p1['conf'][0].cpu().numpy();c2=p2['conf'][0].cpu().numpy()
X0=pts[xy0[:,1],xy0[:,0]];cf0=c1[xy0[:,1],xy0[:,0]];cfk=c2[xyk[:,1],xyk[:,0]]
to=lambda xy,t:np.stack([t[0]+(xy[:,0]+.5)*t[2],t[1]+(xy[:,1]+.5)*t[3]],1)
m0=to(xy0,t0);mk=to(xyk,tk)
# focal of the photo0 pointmap (pinhole through the original photo centre): u-cx = f x/z over the whole fg pointmap
Wp,Hp=Image.open(P0).size;yy,xx=np.nonzero(a0);Pz=pts[yy,xx];u=np.stack([t0[0]+(xx+.5)*t0[2]-Wp/2,t0[1]+(yy+.5)*t0[3]-Hp/2],1);q=Pz[:,:2]/Pz[:,2:3]
f0m=float((u*q).sum()/(q*q).sum())
np.savez(OUT,m0=m0,mk=mk,X0=X0,c0=cf0,ck=cfk,f0_m=f0m)
print('matches %d (fg) conf median %.1f/%.1f  pointmap focal(photo0) %.0f px'%(len(m0),np.median(cf0),np.median(cfk),f0m));print('MATCH_DONE')
