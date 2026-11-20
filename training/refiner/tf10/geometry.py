"""Training-only, tolerant decoded coarse-surface consistency.

Native predicted subdivision is retained. No target topology is copied.
This differentiates decoded offsets/intersection probabilities, not the hard
subdivision choices; missing input support remains outside its capabilities.
"""
import torch
from trellis2.modules.sparse import SparseTensor
from tf3.geometry import FrozenGeometryConsistency, surface_density, _BackwardScale

class NoValidPatch(RuntimeError):
    """No full-halo patch fits the explicit decoder budget."""

class CoarseStructure:
    def __init__(self,decoder_path,hr_statistics,lr_statistics,core_lr=4,
                 halo=20,max_tokens=8192,grid_edges=(8,16),fallback_max_tokens=16384):
        self.base=FrozenGeometryConsistency(decoder_path,hr_statistics,lr_statistics)
        self.core_lr=core_lr;self.halo=halo;self.max_tokens=max_tokens
        self.grid_edges=grid_edges
        self.fallback_max_tokens=max(max_tokens,fallback_max_tokens)
        if halo<20: raise ValueError('Decoder receptive field requires full halo')

    def select(self,hr,lr,seed):
        gen=torch.Generator(device=lr.device).manual_seed(seed)
        best=None
        for k in torch.randperm(len(lr.coords),device=lr.device,generator=gen)[:64]:
            lo=lr.coords[k,1:]//self.core_lr*self.core_lr
            hi=lo+self.core_lr
            ml=((lr.coords[:,1:]>=lo-self.halo)&(lr.coords[:,1:]<hi+self.halo)).all(-1)
            mh=((hr.coords[:,1:]>=2*lo-self.halo)&(hr.coords[:,1:]<2*hi+self.halo)).all(-1)
            nh,nl=int(mh.sum()),int(ml.sum())
            if nh>0 and nl>0 and (best is None or max(nh,nl)<best[0]):
                best=(max(nh,nl),lo,ml,mh)
            if 0<nh<=self.max_tokens and 0<nl<=self.max_tokens:
                return lo,ml,mh
        if best is not None and best[0]<=self.fallback_max_tokens:
            return best[1:]
        raise NoValidPatch('No full-halo patch fits even the dense-patch budget')

    def fields(self,raw,lo,edge):
        keep=((raw.coords[:,1:]>=lo)&(raw.coords[:,1:]<lo+edge)).all(-1)
        if not keep.any(): return [raw.feats.sum()*0+raw.feats.new_zeros((1,g,g,g)) for g in self.grid_edges]
        coords=raw.coords[keep].clone();coords[:,1:]-=lo
        core=SparseTensor(coords=coords,feats=raw.feats[keep])
        return [surface_density(core,edge,g,self.base.decoder.voxel_margin) for g in self.grid_edges]

    def __call__(self,pred,gt,lr,seed=0):
        if pred.shape[0]!=1: raise ValueError('Geometry microbatch must be one')
        lo,ml,mh=self.select(gt,lr,seed)
        def crop(x,m,grad=False):
            f=x.feats[m]
            if grad:f=_BackwardScale.apply(f,1/1024.)
            else:f=f.detach()
            return SparseTensor(coords=x.coords[m].clone(),feats=f)
        with torch.no_grad():
            raw=self.base._raw(crop(lr,ml),self.base.lr_statistics,512)
            low=self.fields(raw,lo*16,self.core_lr*16);del raw
            raw=self.base._raw(crop(gt,mh),self.base.hr_statistics,1024)
            high=self.fields(raw,lo*32,self.core_lr*32);del raw
        raw=self.base._raw(crop(pred,mh,True),self.base.hr_statistics,1024)
        estimate=self.fields(raw,lo*32,self.core_lr*32)
        losses=[];excesses=[];trusts=[]
        for p,h,l in zip(estimate,high,low):
            # GT/GT discrepancy is an irreducible reconstruction/sampling floor.
            # No penalty for fine structure that already differs across GT scales.
            floor=(h-l).abs().detach()
            trust=torch.exp(-floor/.1).detach()
            weights=(1+4*torch.maximum(h,l))*trust
            excess=((p-l).abs()-floor-.01).clamp_min(0)
            losses.append((excess*weights).sum()/weights.sum().clamp_min(1))
            excesses.append(float(excess.detach().mean()));trusts.append(float(trust.mean()))
        loss=torch.stack(losses).mean()
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite structure loss')
        return _BackwardScale.apply(loss,1024.),dict(crop_hr=int(mh.sum()),crop_lr=int(ml.sum()),
            excess=sum(excesses)/len(excesses),trust=sum(trusts)/len(trusts))
