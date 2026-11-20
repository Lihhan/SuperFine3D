"""Input crop for the hfdetail pipeline (as demo_real.py, CAP 4096): fitted preprocess box of the input image, masked by
the dilated silhouette of the generated 1024 mesh under the refined camera. usage: crop_input.py DIR"""
from os import environ as _E   # install locations, set by setup/env.sh (SF3D_ROOT, SF3D_CKPT, ...)
import sys,json,numpy as np,torch
sys.path.insert(0,f'{_E["SF3D_ROOT"]}/superfine3d/detail/whole');import inject_lib as L
from hfdetail import raster,camera
from PIL import Image
D=sys.argv[1];fit=json.load(open(f'{D}/cam_fit.json'));cam=json.load(open(f'{D}/cam_refined.json'))
src=Image.open(f'{D}/input.png').convert('RGBA');bg=Image.new('RGBA',src.size,(0,0,0,255));bg.alpha_composite(src);src=bg.convert('RGB')
sc=max(src.size)/max(fit['size']);box=[b*sc for b in fit['box']];side=box[2]-box[0];W0=int(min(4096,round(side)));W0-=W0%8
crop=src.crop(tuple(int(round(b)) for b in box)).resize((W0,W0),Image.Resampling.LANCZOS)
v,F=L.read_mesh(f'{D}/mesh_1024.ply');v=torch.from_numpy(v).cuda();F=torch.from_numpy(F).cuda()
tri=raster.tri_ids(raster.rasterize(camera.clip_coords(v,cam['P']),F,W0));sil=torch.nn.functional.max_pool2d((tri>=0).float()[None,None],9,1,4)[0,0]>0
img=(np.array(crop).astype(np.float32)*sil.cpu().numpy()[...,None]).astype(np.uint8);Image.fromarray(img).save(f'{D}/input_crop.png')
print('crop',W0,'IoU %.3f NCC %.3f'%(cam['iou'],cam['ncc']),flush=True)
