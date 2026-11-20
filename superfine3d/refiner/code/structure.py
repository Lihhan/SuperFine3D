"""SC1: fixed object-frame LR structure, bounded group modulation before QKV."""
import inspect,textwrap
import torch
from torch import nn
from torch.nn import functional as F
from tf8.model import ElasticTF8SLatFlowModel
import tf7.model as base

class StructureField(nn.Module):
 def __init__(self,channels=32,evidence_width=128,width=64):
  super().__init__();self.project=nn.Sequential(nn.Linear(channels+evidence_width,width),nn.SiLU(),nn.Linear(width,width));self.width=width
 def forward(self,lr,evidence,hr_coords,lr_edge,hr_edge,edge=8):
  # Small dense object-frame field; no HR-sized neighborhood materialization.
  dtype=self.project[0].weight.dtype;f=self.project(torch.cat((lr.feats.to(dtype),evidence.to(dtype)),1));out=[]
  for b,sl in enumerate(lr.layout):
   xyz=(lr.coords[sl,1:].long()*edge//lr_edge).clamp(0,edge-1);key=(xyz[:,0]*edge+xyz[:,1])*edge+xyz[:,2]
   sums=torch.zeros(edge**3,self.width,device=f.device,dtype=torch.float32).index_add(0,key,f[sl].float());count=torch.zeros(edge**3,1,device=f.device).index_add(0,key,torch.ones(len(key),1,device=f.device))
   valid=(count>0).float();field=(sums/count.clamp_min(1))*valid
   field=field.T.reshape(1,self.width,edge,edge,edge);valid=valid.T.reshape(1,1,edge,edge,edge)
   # Average occupied coarse cells equally; a 3-cell footprint is fixed in world units.
   numer=F.avg_pool3d(field,3,1,1,count_include_pad=True);denom=F.avg_pool3d(valid,3,1,1,count_include_pad=True)
   hc=hr_coords[hr_coords[:,0]==b,1:].float();grid=((hc+.5)/hr_edge*2-1)[:,[2,1,0]].reshape(1,-1,1,1,3)
   num=F.grid_sample(numer,grid,mode='bilinear',padding_mode='zeros',align_corners=False)[0,:,:,0,0].T
   den=F.grid_sample(denom,grid,mode='bilinear',padding_mode='zeros',align_corners=False)[0,:,:,0,0].T
   out.append(torch.cat((num/den.clamp_min(1e-6),den),1))
  return torch.cat(out).to(dtype)

class GroupMod(nn.Module):
 def __init__(self,width,channels,groups=12,cap=.1):
  super().__init__();self.out=nn.Linear(width+1,2*groups);nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias);self.groups=groups;self.cap=cap
 def forward(self,h,c):
  pieces=[]
  for first in range(0,len(h),65536):
   f=h[first:first+65536];s=c[first:first+65536];ab=self.cap*torch.tanh(self.out(s.to(self.out.weight.dtype)));a,b=ab.chunk(2,1)
   # Confidence reaches zero outside all occupied coarse neighborhoods.
   confidence=(s[:,-1:]>1e-6).to(a.dtype);a=a*confidence;b=b*confidence
   z=f.reshape(-1,self.groups,f.shape[-1]//self.groups)
   pieces.append((z*(1+a[:,:,None].to(z.dtype))+b[:,:,None].to(z.dtype)).reshape_as(f))
  return torch.cat(pieces)

class StructureFlow(ElasticTF8SLatFlowModel):
 def __init__(self,*args,structure_edge=16,structure_cap=.1,**kwargs):
  super().__init__(*args,**kwargs)
  # Preserve random streams relative to control, including subsequent flow noise.
  with torch.random.fork_rng(devices=[]):
   self.lr_structure=StructureField(kwargs['in_channels'],kwargs.get('tf5_width',128),64)
   self.lr_structure_mod=nn.ModuleList([GroupMod(64,self.model_channels,self.num_heads,structure_cap) for _ in self.blocks])
  self.structure_edge=structure_edge;self.structure_mode='fixed';self.structure_enabled=True;self.assert_trainable_contract()
 def structure(self,lr,contexts,coords,resolution):
  edge=self.structure_edge if self.structure_mode=='fixed' else max(2,round(self.structure_edge*self.lr_resolution/32))
  return self.lr_structure(lr,contexts,coords,self.lr_resolution,resolution,edge)
 def modulate(self,i,hidden,structure):
  return hidden if not self.structure_enabled else hidden.replace(self.lr_structure_mod[i](hidden.feats,structure))

# Explicitly pass structure through checkpoint inputs; no mutable cross-batch state.
source=textwrap.dedent(inspect.getsource(base.ElasticTF7SLatFlowModel.forward))
source=source.replace('lr=lp=lm=evidence=coverage=times=plans=anchor=None','lr=lp=lm=evidence=coverage=times=plans=anchor=structure=None')
source=source.replace('            additions.append(delta);contexts.append(ctx)','            additions.append(delta);contexts.append(ctx)')
source=source.replace('        if len(lr.feats)!=len(lr_cond.feats):', '        structure=self.structure(lr_cond,torch.cat(contexts),x.coords,resolution)\n        if len(lr.feats)!=len(lr_cond.feats):')
source=source.replace('ev,cv,ts,an,i=i,block=block','ev,cv,ts,an,st,i=i,block=block')
source=source.replace('q,k,v=self._qkv(block.self_attn,hidden,positions)','q,k,v=self._qkv(block.self_attn,self.modulate(i,hidden,st) if enabled else hidden,positions)')
source=source.replace('coverage,times,anchor,use_reentrant=False','coverage,times,anchor,structure,use_reentrant=False').replace('coverage,times,anchor)','coverage,times,anchor,structure)')
ns=dict(base.__dict__);exec(compile(source,'<SC1-forward>','exec'),ns);StructureFlow.forward=ns['forward']
FORWARD_SOURCE=source
