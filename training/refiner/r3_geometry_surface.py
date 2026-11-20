"""One full teacher and student decode, shared by all four supervision terms."""
import math
import torch
import torch.nn.functional as F
from refine_surface import RefinementSurface
from refine_objective import refinement_objective
from r3_geometry_objective import teacher_planes,raw_losses
from trellis2.modules.sparse import SparseTensor
from tf3.geometry import _BackwardScale
from tf10.geometry import NoValidPatch

class GeometrySurface(RefinementSurface):
    def __call__(self,pred,target,lr,seed=0):
        if target.shape[0]!=1 or not torch.equal(pred.coords,target.coords):raise ValueError('Single paired asset required')
        if max(len(target.coords),len(lr.coords))>self.full_decode_token_limit:raise NoValidPatch('Full-context decode token budget')
        lo,_=self.select(target,seed)
        def fresh(x,grad=False):
            f=_BackwardScale.apply(x.feats,1/1024.) if grad else x.feats.detach()
            return SparseTensor(coords=x.coords.clone(),feats=f)
        with torch.no_grad():
            high=self.base._raw(fresh(target),self.stats,1024)
            vh,fh=self.local_mesh(high,lo,1024)
            # Retain only the local teacher outputs after full-context decoding.
            core=((high.coords[:,1:]>=lo*16)&(high.coords[:,1:]<(lo+self.core_edge)*16)).all(1)
            if not core.any():raise NoValidPatch('Empty teacher decoder core')
            high=SparseTensor(coords=high.coords[core].clone(),feats=high.feats[core].clone())
            normals,plane_valid=teacher_planes(high.coords,high.feats,vh,fh,lo,self.core_edge,self.base.decoder.voxel_margin)
            low=self.base._raw(fresh(lr),self.base.lr_statistics,512)
            vl,fl=self.local_mesh(low,lo,512);del low
        raw=self.base._raw(fresh(pred,True),self.stats,1024)
        aux,report=raw_losses(raw,high,normals,plane_valid,lo,self.core_edge,self.supervision_inset,self.base.decoder.voxel_margin)
        vp,fp=self.local_mesh(raw,lo,1024);del raw,high
        angle=(seed%360)*math.pi/180
        axis=F.normalize(vp.new_tensor([math.cos(angle),math.sin(angle),.7]),dim=0)
        right=F.normalize(torch.cross(vp.new_tensor([0.,0.,1.]),axis,dim=0),dim=0);up=torch.cross(axis,right,dim=0)
        results=[];reports=[];h_lr=(1/512.)/(self.core_edge/128.)
        for rot in [torch.stack([right,up,axis]),torch.stack([-right,up,-axis])]:
            try:
                with torch.no_grad():
                    hd,hn,hm=self.render(vh,fh,rot);ld,ln,lm=self.render(vl,fl,rot)
                    reliable=hm*lm*((hn*ln).sum(1,keepdim=True)>.8)*((hd-ld).abs()<4*h_lr)
                pd,pn,pm=self.render(vp,fp,rot)
                losses,detail=refinement_objective(pd,hd,ld,reliable,pm,h_lr)
                visible=(reliable*pm).detach()
                normal=((1-(pn*hn).sum(1,keepdim=True).clamp(-1,1))*visible).sum()/visible.sum().clamp_min(1)
                losses['detail']=losses['detail']+.1*normal
                results.append(losses);reports.append(detail)
            except NoValidPatch:
                pass  # Edge supervision remains valid even if prediction has no faces.
        render_valid=sum(x['valid_pixels'] for x in reports)>=64
        losses={k:torch.stack([r[k] for r in results]).mean() if render_valid else aux['edge']*0 for k in ['keep','detail']}
        losses.update(aux)
        if not all(torch.isfinite(v) for v in losses.values()):raise FloatingPointError('Nonfinite geometry objective')
        losses={k:_BackwardScale.apply(v,1024.) for k,v in losses.items()}
        return losses,dict(views=reports,raw_supervision=report,render_valid=render_valid,
                          hr_tokens=len(target.coords),lr_tokens=len(lr.coords),lo=lo.tolist(),
                          full_context_decode=True,target='GT HR latent frozen-decoder reconstruction')
