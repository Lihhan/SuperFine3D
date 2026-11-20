"""Candidate 512 start from another generator's mesh: place it in the TRELLIS-512 frame of the same input via the two fitted
input cameras (rotation = camera bases, scale = vertical extent, translation = silhouette box + near surface), keep it inside
the unit cube, encode at 512 -> OUT/latent_512.npz. usage: cand_start.py MESH CAM REF_MESH REF_CAM OUT"""
from os import environ as _E   # install locations, set by setup/env.sh (SF3D_ROOT, SF3D_CKPT, ...)
import sys,json,numpy as np,torch
torch.set_num_threads(4)
sys.path[:0]=[f'{_E["SF3D_ROOT"]}/superfine3d/detail/whole',f'{_E["SF3D_ROOT"]}/superfine3d/detail']
import inject_lib as L
from hfdetail import latent_io
M,C,RM,RC,O=sys.argv[1:6];nrm=lambda a:a/np.linalg.norm(a)
B=lambda c:np.stack([nrm(np.array(c['r'])),nrm(np.array(c['u'])),nrm(np.array(c['fwd']))])
gv,_=L.read_mesh(RM);gv=gv.astype(np.float64);Bg=B(json.load(open(RC)));v,f=L.read_mesh(M);v=v.astype(np.float64);Bm=B(json.load(open(C)))
y=(v-v.mean(0))@Bm.T;g=gv@Bg.T;s=np.ptp(g[:,1])/np.ptp(y[:,1]);y=y*s
t=np.array([(g[:,0].min()+g[:,0].max()-y[:,0].min()-y[:,0].max())/2,(g[:,1].min()+g[:,1].max()-y[:,1].min()-y[:,1].max())/2,np.percentile(g[:,2],2)-np.percentile(y[:,2],2)])
x=(y+t)@Bg;c=(x.min(0)+x.max(0))/2;h=np.abs(x-c).max()
if np.abs(x).max()>.495:x=(x-c)*min(1,.495/h)+np.clip(c,-.495+min(h,.495),.495-min(h,.495));print('refit into cube, scale',min(1,.495/h))
co,fe=latent_io.encode_direct(torch.from_numpy(x).float(),torch.from_numpy(f).long(),512)
np.savez_compressed(f'{O}/latent_512.npz',coords=np.asarray(co),feats=np.asarray(fe,dtype=np.float32));print('encoded',len(co),'tokens, scale',round(float(s),3))
