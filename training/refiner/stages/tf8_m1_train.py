from os import environ as _E   # install locations, set by setup/env.sh (SF3D_ROOT, SF3D_CKPT, ...)
import sys,json,runpy,argparse,hashlib
from pathlib import Path
import torch,numpy as np
R=Path(__file__).parent
p=argparse.ArgumentParser(add_help=False)
for n in ['trellis-root','tf1-root','config']:p.add_argument('--'+n,required=True)
a,rest=p.parse_known_args();sys.path[:0]=[a.tf1_root,a.trellis_root]
from trellis2.datasets.paired_lr import PairedLRImageConditionedSLatShape as Base
from tf6.trainer import TF6Trainer
from tf8.model import ElasticTF8SLatFlowModel
import trellis2.datasets as datasets,trellis2.models as models,trellis2.trainers as trainers
class MultiDataset(Base):
 def get_instance(self,root,instance):
  pack=super().get_instance(root,instance)
  lo={}
  for res,prefix,mean,std in [(512,'low_hr',self.mean,self.std),(256,'low_lr',self.lr_mean,self.lr_std)]:
   path=Path(root['shape_latent_lr']) if res==512 else Path(f'{_E["SF3D_DATA"]}/data256/shape_latents/shape_enc_next_dc_f16c32_fp16_256')
   with np.load(path/(instance+'.npz')) as z:
    lo[prefix+'_coords']=torch.from_numpy(z['coords'].copy()).int()
    lo[prefix+'_feats']=(torch.from_numpy(z['feats'].copy()).float()-mean)/std
  pack.update(lo);return pack
 @staticmethod
 def collate_fn(batch,split_size=None):
  high=[];low=[]
  for b in batch:
   h={k:v for k,v in b.items() if not k.startswith('low_')};l=dict(h)
   l.update(coords=b['low_hr_coords'],feats=b['low_hr_feats'],lr_coords=b['low_lr_coords'],lr_feats=b['low_lr_feats']);high.append(h);low.append(l)
  # Collate separately to preserve each scale's own load-balanced grouping.
  h=Base.collate_fn(high,split_size);l=Base.collate_fn(low,split_size)
  if split_size is None:h['low_pack']=l;return h
  for x,y in zip(h,l):x['low_pack']=y
  return h
class M1Trainer(TF6Trainer):
 def finetune_from(self,ckpt):
  super().finetune_from(ckpt);m=self.models['denoiser'];path=Path(f'{_E["SF3D_CKPT"]}/stages/tf8_B/denoiser_step0003000.pt');s=torch.load(path,map_location='cpu',weights_only=True)
  assert set(s)=={n for n,p in m.named_parameters() if p.requires_grad}
  m.load_state_dict(s,strict=False);states={n:x.state_dict() for n,x in self.models.items()};self._state_dicts_to_master_params(self.master_params,states)
  for e in self.ema_params:self._state_dicts_to_master_params(e,states)
  assert all(not p.requires_grad for n,p in m.named_parameters() if not n.startswith('lr_'))
  masters=self._master_params_to_state_dicts(self.master_params)['denoiser']
  assert all(torch.equal(masters[n].float(),p.detach().float()) for n,p in m.named_parameters() if p.requires_grad)
  Path(self.output_dir,'m1_initialization.json').write_text(json.dumps(dict(checkpoint=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),hr_frozen=True,masters_synced=True)))
 def training_losses(self,x_0,cond=None,low_pack=None,**kwargs):
  low=self.step%4==0
  if low:
   assert low_pack is not None
   x_0=low_pack['x_0'];cond=low_pack.get('cond');kwargs={k:v for k,v in low_pack.items() if k not in ['x_0','cond']}
  hr,lr=(32,16) if low else (64,32)
  self.models['denoiser'].set_resolution_pair(hr,lr)
  assert x_0.coords[:,1:].max()<hr and kwargs['lr_cond'].coords[:,1:].max()<lr
  terms,extra=super().training_losses(x_0,cond,**kwargs)
  terms['scale_256_512' if low else 'scale_512_1024']={'mse':float(terms['mse'].detach()),'hr_tokens':len(x_0.coords),'lr_tokens':len(kwargs['lr_cond'].coords)}
  if self.step<4:print('M1_SCALE',self.step,hr,lr,'loss',float(terms['loss'].detach()),flush=True)
  return terms,extra
models.ElasticTF8SLatFlowModel=ElasticTF8SLatFlowModel;datasets.MultiDataset=MultiDataset;trainers.M1Trainer=M1Trainer
sys.argv=[sys.argv[0],'--config',a.config,*rest];runpy.run_path(str(Path(a.trellis_root)/'train.py'),run_name='__main__')
