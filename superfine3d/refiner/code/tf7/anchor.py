"""Geometry-only cached trilinear plans; feature gathers remain differentiable."""
from dataclasses import dataclass
import itertools
import torch
from torch import nn
from torch.nn import functional as F
from tf5.conditioning import GatedResidual

@dataclass
class Plan:
    indices: torch.Tensor
    weights: torch.Tensor
    mass: torch.Tensor
    metadata: torch.Tensor

@torch.no_grad()
def build_plan(lr, hr, le, he):
    if le <= 0 or he < le or he % le: raise ValueError('Invalid resolution pair')
    if lr.device != hr.device: raise ValueError('Coordinate device mismatch')
    for c,e in ((lr,le),(hr,he)):
        if c.ndim!=2 or c.shape[1]!=4 or c.dtype not in (torch.int32,torch.int64): raise ValueError('Invalid coordinates')
        if c.numel() and ((c<0).any() or (c[:,1:]>=e).any()): raise ValueError('Coordinates out of bounds')
    def key(c):
        c=c.long()
        return ((c[:,0]*le+c[:,1])*le+c[:,2])*le+c[:,3]
    keys,order=key(lr).sort()
    if len(keys)>1 and (keys[1:]==keys[:-1]).any(): raise ValueError('Duplicate LR coordinates')
    pos=(hr[:,1:].float()+.5)*(le/he)-.5
    lower=pos.floor().long(); frac=pos-lower
    ids=[]; weights=[]
    for bits in itertools.product((0,1),repeat=3):
        bit=torch.tensor(bits,device=hr.device)
        xyz=lower+bit
        valid=((xyz>=0)&(xyz<le)).all(1)
        q=key(torch.cat((hr[:,:1].long(),xyz.clamp(0,le-1)),1))
        if len(keys):
            i=torch.searchsorted(keys,q); safe=i.clamp(max=len(keys)-1)
            valid=valid & (i<len(keys)) & (keys[safe]==q)
            idx=order[safe]
        else:
            valid=torch.zeros_like(valid); idx=torch.zeros(len(hr),device=hr.device,dtype=torch.long)
        w=torch.where(bit.bool(),frac,1-frac).prod(1)*valid
        ids.append(idx); weights.append(w)
    weights=torch.stack(weights,1); mass=weights.sum(1,keepdim=True)
    # Count positive-weight neighbors; normalized count and ratio are explicit.
    meta=torch.cat((frac,mass,(weights>0).sum(1,keepdim=True)/8,
                    torch.full_like(mass,le/he)),1)
    return Plan(torch.stack(ids,1),weights/mass.clamp_min(1e-8),mass,meta)

class PlanCache:
    """One entry, coordinate identity AND mutation version; never caches features."""
    def __init__(self): self.entry=None
    def get(self,lr,hr,le,he):
        # inference_mode tensors have no version counter: rebuild rather than risk staleness.
        try: versions=(lr._version,hr._version)
        except RuntimeError: return build_plan(lr,hr,le,he)
        if self.entry is not None:
            l,h,v,scale,p=self.entry
            if l is lr and h is hr and v==versions and scale==(le,he): return p
        p=build_plan(lr,hr,le,he)
        self.entry=(lr,hr,versions,(le,he),p)
        return p

def apply_plan(plan,features):
    if not len(features): return features.new_zeros((len(plan.indices),features.shape[-1]),dtype=torch.float32)+features.sum()*0
    # Eight narrow gathers; no [N,8,C] materialization, sort or CPU copy here.
    out=torch.zeros((len(plan.indices),features.shape[-1]),device=features.device,dtype=torch.float32)
    for j in range(8):
        out=out+features[plan.indices[:,j]].float()*plan.weights[:,j,None]
    return out

class AnchorEncoder(nn.Module):
    def __init__(self,channels,width):
        super().__init__()
        self.project=nn.Linear(channels,width)
        self.mix=nn.Sequential(nn.Linear(width+6,width),nn.SiLU(),nn.Linear(width,width))
    def encode(self,sampled,plan):
        # Zero contribution for empty neighborhoods even with learned biases.
        return self.mix(torch.cat((sampled,plan.metadata),-1).to(self.project.weight.dtype))*plan.mass.to(self.project.weight.dtype)

class AnchoredResidual(GatedResidual):
    def __init__(self,channels,width):
        super().__init__(channels,width)
        self.hr_gate=nn.Linear(channels,1,bias=False)
        nn.init.zeros_(self.hr_gate.weight)
        self.anchor_enabled=True
    def forward(self,read,evidence,coverage,time,anchor,hr):
        gate=torch.sigmoid(self.gate(torch.cat((evidence,coverage,time),-1))+self.hr_gate(hr))
        low=self.down(read)
        if self.anchor_enabled: low=low+anchor
        return gate*self.up(F.silu(low))
