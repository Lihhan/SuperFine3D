from os import environ as _E   # install locations, set by setup/env.sh (SF3D_ROOT, SF3D_CKPT, ...)
import sys, json, runpy, argparse, hashlib, ast
from pathlib import Path
import torch
R=Path(f'{_E["SF3D_ROOT"]}/training/refiner/stages')
sys.path[:0]=[str(R),f'{_E["SF3D_ROOT"]}/superfine3d/refiner/code',f'{_E["SF3D_ROOT"]}/training/refiner']
from sc1_stable_pool import install
install()
from structure import StructureFlow
from tf6.trainer import TF6Trainer
from trellis2.datasets.paired_lr import PairedLRImageConditionedSLatShape as Base
import numpy as np
# Reuse only the audited dataset class, without executing the old training entry.
src=Path(f'{_E["SF3D_ROOT"]}/training/refiner/stages/tf8_m1_train.py').read_text()
node=next(n for n in ast.parse(src).body if isinstance(n,ast.ClassDef) and n.name=='MultiDataset')
exec(compile(ast.Module(body=[node],type_ignores=[]),'<paired-three-scale>','exec'))
BASE=Path(f'{_E["SF3D_CKPT"]}/stages/tf8_B/denoiser_step0003000.pt')
def is_hr(n):return n.startswith('blocks.') and '.self_attn.' in n and 'lora_' in n
class AblationFlow(StructureFlow):
 def __init__(self,*args,joint_hr=False,**kw):
  self._ablation_ready=False
  super().__init__(*args,**kw)
  self.joint_hr=joint_hr
  for n,p in self.named_parameters():
   if is_hr(n):p.requires_grad_(joint_hr)
  self._ablation_ready=True
  self.assert_trainable_contract()
 def assert_trainable_contract(self):
  if not getattr(self,'_ablation_ready',False):return super().assert_trainable_contract()
  actual={n for n,p in self.named_parameters() if p.requires_grad}
  expected={n for n,p in self.named_parameters() if (n.startswith('lr_') and n!='lr_null_embedding') or (self.joint_hr and is_hr(n))}
  assert actual==expected,(actual^expected)
  return dict(joint_hr=self.joint_hr,tensors=len(actual),parameters=sum(p.numel() for p in self.parameters() if p.requires_grad))
class SplitMasterAdamW(torch.optim.AdamW):
 """Leaf views retain the trainer's flat master while giving real per-group LRs."""
 def __init__(self,master,named,hr_ratio,**kw):
  self.full_master=master;self.spans=[];groups={False:[],True:[]};offset=0
  for n,p in named:
   end=offset+p.numel();v=torch.nn.Parameter(master.data[offset:end]);groups[is_hr(n)].append(v);self.spans.append((v,offset,end));offset=end
  assert offset==master.numel()
  lr=kw['lr'];super().__init__([dict(params=ps,lr=lr*(hr_ratio if hr else 1),group_name='hr' if hr else 'lr') for hr,ps in groups.items() if ps],**kw)
 def step(self,closure=None):
  assert closure is None and self.full_master.grad is not None
  for p,a,b in self.spans:p.grad=self.full_master.grad[a:b]
  return super().step()
class AblationTrainer(TF6Trainer):
 def __init__(self,*args,mixed_scales=False,hr_lr_ratio=.1,**kw):
  self.mixed_scales=mixed_scales;self.hr_lr_ratio=hr_lr_ratio;self.scale_counts={'256_512':0,'512_1024':0}
  super().__init__(*args,**kw)
 def init_models_and_more(self,**kw):
  super().init_models_and_more(**kw)
  assert self.mix_precision_mode=='inflat_all' and self.lr_scheduler_config is None
  named=[(n,p) for n,p in self.models['denoiser'].named_parameters() if p.requires_grad]
  self.optimizer=SplitMasterAdamW(self.master_params[0],named,self.hr_lr_ratio,**self.optimizer_config['args'])
 def finetune_from(self,ckpt):
  super().finetune_from(ckpt);m=self.models['denoiser'];s=torch.load(BASE,map_location='cpu',weights_only=True)
  expected={n for n,p in m.named_parameters() if n.startswith('lr_') and p.requires_grad and not n.startswith('lr_structure')}
  assert set(s)==expected
  m.load_state_dict(s,strict=False)
  states={n:x.state_dict() for n,x in self.models.items()};self._state_dicts_to_master_params(self.master_params,states)
  for e in self.ema_params:self._state_dicts_to_master_params(e,states)
  masters=self._master_params_to_state_dicts(self.master_params)['denoiser']
  assert all(torch.equal(masters[n].float(),p.detach().float()) for n,p in m.named_parameters() if p.requires_grad)
  digest=hashlib.sha256();hrhash=hashlib.sha256()
  for n,p in m.named_parameters():
   raw=p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes();digest.update(n.encode());digest.update(raw)
   if is_hr(n):hrhash.update(raw)
  report=dict(base=str(BASE),base_sha256=hashlib.sha256(BASE.read_bytes()).hexdigest(),all_parameter_sha256=digest.hexdigest(),hr_sha256=hrhash.hexdigest(),contract=m.assert_trainable_contract(),masters_synced=True,optimizer_groups=[dict(name=g['group_name'],lr=g['lr'],numel=sum(p.numel() for p in g['params'])) for g in self.optimizer.param_groups])
  Path(self.output_dir,'ablation_initialization.json').write_text(json.dumps(report,indent=2))
  print('ABLATION_INIT',json.dumps(report),flush=True)
 def training_losses(self,x_0,cond=None,low_pack=None,**kw):
  low=self.mixed_scales and self.step%2==0
  if low:
   assert low_pack is not None
   x_0=low_pack['x_0'];cond=low_pack.get('cond');kw={k:v for k,v in low_pack.items() if k not in ['x_0','cond']}
  hr,lr=(32,16) if low else (64,32);m=self.models['denoiser'];m.set_resolution_pair(hr,lr)
  assert x_0.coords[:,1:].max()<hr and kw['lr_cond'].coords[:,1:].max()<lr
  # Shape-dependent RNG draws must not shift subsequent steps in the mixed arm.
  with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
   torch.manual_seed(730000+self.step*2+self._micro_count%2)
   terms,extra=super().training_losses(x_0,cond,**kw)
  key='256_512' if low else '512_1024';self.scale_counts[key]+=1
  terms['scale_'+key]={'mse':float(terms['mse'].detach()),'hr_tokens':len(x_0.coords),'lr_tokens':len(kw['lr_cond'].coords)}
  if self._micro_count%2==0:
   Path(self.output_dir,'progress.json').write_text(json.dumps(dict(completed_forward_steps=self.step+1,scale_microbatches=self.scale_counts,peak_gib=torch.cuda.max_memory_allocated()/2**30)))
  return terms,extra

def main():
 ap=argparse.ArgumentParser(add_help=False)
 for n in ['trellis-root','tf1-root','config']:ap.add_argument('--'+n,required=True)
 a,rest=ap.parse_known_args();sys.path[:0]=[a.tf1_root,a.trellis_root]
 import trellis2.models as models,trellis2.trainers as trainers,trellis2.datasets as datasets
 models.AblationFlow=AblationFlow;trainers.AblationTrainer=AblationTrainer;datasets.MultiDataset=MultiDataset
 sys.argv=[sys.argv[0],'--config',a.config,*rest];runpy.run_path(str(Path(a.trellis_root)/'train.py'),run_name='__main__')
if __name__=='__main__':main()
