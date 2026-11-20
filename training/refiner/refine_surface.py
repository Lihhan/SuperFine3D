"""Full latent decode, local surface supervision: no cropped decoder context.

Budget exclusions are explicit. The first pilot uses 512->1024 only.
"""
import math
import torch
import torch.nn.functional as F
from tf11.native_surface import NativeSurface
from tf10.geometry import NoValidPatch
from tf3.geometry import _BackwardScale
from refine_objective import refinement_objective
from trellis2.modules.sparse import SparseTensor

class RefinementSurface(NativeSurface):
    def __init__(self,**kw):
        super().__init__(**kw)
        self.full_decode_token_limit=12000

    def local_mesh(self,raw,lo,res):
        from o_voxel.convert import flexible_dual_grid_to_mesh
        # lo/core are expressed in HR latent units for both scales.
        factor=res/64
        keep=((raw.coords[:,1:]>=lo*factor-2)&(raw.coords[:,1:]<(lo+self.core_edge)*factor+2)).all(1)
        f=raw.feats[keep].float();c=raw.coords[keep,1:]
        off=(1+2*self.base.decoder.voxel_margin)*f[:,:3].sigmoid()-self.base.decoder.voxel_margin
        v,faces=flexible_dual_grid_to_mesh(c,off,f[:,3:6]>0,F.softplus(f[:,6:7]),
            aabb=[[-.5]*3,[.5]*3],grid_size=res,train=False)
        v=((v+.5)*64-(lo.float()+self.core_edge/2))/(self.core_edge/2)
        return v,faces

    def __call__(self,pred,target,lr,seed=0):
        if target.shape[0]!=1 or not torch.equal(pred.coords,target.coords):raise ValueError('Single paired asset required')
        if max(len(target.coords),len(lr.coords))>self.full_decode_token_limit:
            raise NoValidPatch('Full-context decode token budget')
        lo,_=self.select(target,seed)
        def fresh(x,grad=False):
            # SparseTensor.replace shares caches. Never reuse teacher no-grad
            # convolution maps in the student's differentiable decoder pass.
            f=_BackwardScale.apply(x.feats,1/1024.) if grad else x.feats.detach()
            return SparseTensor(coords=x.coords.clone(),feats=f)
        with torch.no_grad():
            high=self.base._raw(fresh(target),self.stats,1024)
            vh,fh=self.local_mesh(high,lo,1024);del high
            low=self.base._raw(fresh(lr),self.base.lr_statistics,512)
            vl,fl=self.local_mesh(low,lo,512);del low
        raw=self.base._raw(fresh(pred,True),self.stats,1024)
        vp,fp=self.local_mesh(raw,lo,1024);del raw
        angle=(seed%360)*math.pi/180
        axis=F.normalize(vp.new_tensor([math.cos(angle),math.sin(angle),.7]),dim=0)
        right=F.normalize(torch.cross(vp.new_tensor([0.,0.,1.]),axis,dim=0),dim=0);up=torch.cross(axis,right,dim=0)
        results=[];reports=[]
        # One LR geometry voxel measured in the local render coordinate frame.
        h_lr=(1/512.)/(self.core_edge/128.)
        for rot in [torch.stack([right,up,axis]),torch.stack([-right,up,-axis])]:
            with torch.no_grad():
                hd,hn,hm=self.render(vh,fh,rot);ld,ln,lm=self.render(vl,fl,rot)
                reliable=hm*lm*((hn*ln).sum(1,keepdim=True)>.8)*((hd-ld).abs()<4*h_lr)
            pd,pn,pm=self.render(vp,fp,rot)
            losses,report=refinement_objective(pd,hd,ld,reliable,pm,h_lr)
            visible=(reliable*pm).detach()
            normal=((1-(pn*hn).sum(1,keepdim=True).clamp(-1,1))*visible).sum()/visible.sum().clamp_min(1)
            losses['detail']=losses['detail']+.1*normal
            results.append(losses);reports.append(report)
        if sum(x['valid_pixels'] for x in reports)<64:raise NoValidPatch('Too few reliable paired surface pixels')
        losses={k:_BackwardScale.apply(torch.stack([r[k] for r in results]).mean(),1024.) for k in results[0]}
        return losses,dict(views=reports,hr_tokens=len(target.coords),lr_tokens=len(lr.coords),
            full_context_decode=True,lo=lo.tolist(),target='GT latent reconstructed surfaces')
