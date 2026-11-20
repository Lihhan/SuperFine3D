"""Prototype: native subdivision, coordinate-matched face supervision and local views.

GT is the frozen decoder reconstruction, not the original mesh. Missing native
subdivision is reported explicitly; no target topology is imposed on prediction.
Training integration remains gated on ROI diagnosis and real-latent gradients.
"""
import math
import json,hashlib
import torch
import torch.nn.functional as F
from trellis2.modules.sparse import SparseTensor
from tf3.geometry import FrozenGeometryConsistency, _BackwardScale
from tf10.geometry import NoValidPatch
from .surface_objective import surface_objective, face_objective


def correspondence(pc,qc,resolution):
    def key(c):
        c=c.long();return ((c[:,0]*resolution+c[:,1])*resolution+c[:,2])*resolution+c[:,3]
    pk,qk=key(pc),key(qc);ordered,perm=qk.sort()
    where=torch.searchsorted(ordered,pk)
    safe=where.clamp_max(len(ordered)-1)
    valid=(where<len(ordered)) & (ordered[safe]==pk)
    return valid,perm[safe]


class NativeSurface:
    def __init__(self,decoder_path,hr_statistics,lr_statistics,core_edge=8,halo=20,max_tokens=16384,render_size=128,supervision_inset=2,verified_regions=None):
        if halo<20:raise ValueError('Full decoder receptive field required')
        self.base=FrozenGeometryConsistency(decoder_path,hr_statistics,lr_statistics)
        self.stats=hr_statistics;self.core_edge=core_edge;self.halo=halo
        self.max_tokens=max_tokens;self.render_size=render_size;self.ctx=None
        self.supervision_inset=supervision_inset
        if not 0<=supervision_inset<core_edge/2:raise ValueError('Invalid supervision inset')
        self.verified=json.load(open(verified_regions)) if verified_regions else None

    def select(self,target,seed):
        xyz=target.coords[:,1:];g=torch.Generator(device=xyz.device).manual_seed(seed)
        if self.verified is not None:
            key=hashlib.sha256(target.coords.detach().cpu().numpy().astype('int32').tobytes()).hexdigest()
            if key not in self.verified:raise NoValidPatch('No full-decode-verified region')
            regions=self.verified[key]['regions'];lo=xyz.new_tensor(regions[seed%len(regions)])
            mask=((xyz>=lo-self.halo)&(xyz<lo+self.core_edge+self.halo)).all(1)
            if int(mask.sum())>self.max_tokens:raise NoValidPatch('Verified region exceeds budget')
            return lo,mask
        for index in torch.randperm(len(xyz),generator=g,device=xyz.device)[:64]:
            lo=xyz[index]//self.core_edge*self.core_edge;hi=lo+self.core_edge
            mask=((xyz>=lo-self.halo)&(xyz<hi+self.halo)).all(1)
            inside=((xyz>=lo+self.supervision_inset)&(xyz<hi-self.supervision_inset)).all(1)
            if int(inside.sum())>=8 and 0<int(mask.sum())<=self.max_tokens:return lo,mask
        raise NoValidPatch('Native-surface full halo exceeds budget')

    def mesh(self,raw,lo):
        from o_voxel.convert import flexible_dual_grid_to_mesh
        c=raw.coords;keep=((c[:,1:]>=lo*16-2)&(c[:,1:]<(lo+self.core_edge)*16+2)).all(1)
        f=raw.feats[keep].float();c=c[keep,1:]
        offsets=(1+2*self.base.decoder.voxel_margin)*f[:,:3].sigmoid()-self.base.decoder.voxel_margin
        vertices,faces=flexible_dual_grid_to_mesh(c,offsets,f[:,3:6]>0,F.softplus(f[:,6:7]),
                         aabb=[[-.5]*3,[.5]*3],grid_size=1024,train=False)
        # Normalize to fixed LR patch units. No per-mesh bounding-box rescaling.
        vertices=((vertices+.5)*1024-(lo.float()+self.core_edge/2)*16)/(self.core_edge*8)
        return vertices,faces

    def render(self,vertices,faces,rotation):
        import nvdiffrast.torch as dr
        if self.ctx is None:self.ctx=dr.RasterizeCudaContext(device=vertices.device)
        if len(faces)==0:raise NoValidPatch('Empty native surface')
        v=vertices@rotation.T
        clip=torch.cat((v[:,:2]*.7,v[:,2:3]*.25,torch.ones_like(v[:,:1])),1)[None].contiguous()
        f=faces.int().contiguous();rast,_=dr.rasterize(self.ctx,clip,f,resolution=[self.render_size]*2)
        depth=dr.interpolate(v[None,:,2:3].contiguous(),rast,f)[0]
        tri=v[f.long()];n=F.normalize(torch.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0],dim=-1),dim=-1,eps=1e-8)
        n=n*torch.where(n[:,2:3]<0,-1.,1.)
        nf=torch.arange(len(f),device=v.device,dtype=torch.int32)[:,None].expand(-1,3).contiguous()
        normal=dr.interpolate(n[None].contiguous(),rast,nf)[0]
        mask=(rast[...,3:4]>0).float()
        # Exclude the crop boundary in object space, for both opposing views.
        world=dr.interpolate(vertices[None].contiguous(),rast,f)[0]
        limit=1.-2.*self.supervision_inset/self.core_edge
        mask=mask*(world.abs()<limit).all(-1,keepdim=True).float()
        return tuple(a.permute(0,3,1,2) for a in (depth,normal,mask))

    def __call__(self,pred,target,seed=0):
        if pred.shape[0]!=1 or not torch.equal(pred.coords,target.coords):
            raise ValueError('Paired training needs identical input HR support')
        lo,mask=self.select(target,seed)
        gt=SparseTensor(coords=target.coords[mask].clone(),feats=target.feats[mask].detach())
        pp=SparseTensor(coords=pred.coords[mask].clone(),feats=_BackwardScale.apply(pred.feats[mask],1/1024.))
        with torch.no_grad():q=self.base._raw(gt,self.stats,1024)
        p=self.base._raw(pp,self.stats,1024)
        core=((p.coords[:,1:] >= (lo+self.supervision_inset)*16)&(p.coords[:,1:]<(lo+self.core_edge-self.supervision_inset)*16)).all(1)
        pc=p.coords[core];pf=p.feats[core].float()
        valid,ids=correspondence(pc,q.coords,1024)
        qf=q.feats[ids].float().detach()
        # A missing GT output cell has no reconstructed edge intersections.
        labels=(qf[:,3:6]>0).float()*valid[:,None]
        intersection=face_objective(pf[:,3:6],labels,torch.ones_like(labels,dtype=torch.bool))
        active=valid & (labels>0).any(1)
        margin=self.base.decoder.voxel_margin
        po=(1+2*margin)*pf[:,:3].sigmoid()-margin
        qo=(1+2*margin)*qf[:,:3].sigmoid()-margin
        offset=F.smooth_l1_loss(po[active],qo[active],beta=.1) if active.any() else pf.sum()*0
        vp,fp=self.mesh(p,lo)
        with torch.no_grad():vq,fq=self.mesh(q,lo)
        angle=(seed%360)*math.pi/180
        axis=F.normalize(vp.new_tensor([math.cos(angle),math.sin(angle),.7]),dim=0)
        right=F.normalize(torch.cross(vp.new_tensor([0.,0.,1.]),axis,dim=0),dim=0)
        up=torch.cross(axis,right,dim=0)
        observations=[];coverage=[]
        for rot in [torch.stack((right,up,axis)),torch.stack((-right,up,-axis))]:
            with torch.no_grad():td,tn,tm=self.render(vq,fq,rot)
            pd,pn,pm=self.render(vp,fp,rot)
            valid_pixel=(tm*pm).detach()
            observations.append(surface_objective(pd,td,pn,tn,valid_pixel))
            coverage.append(float(valid_pixel.sum()/tm.sum().clamp_min(1)))
        result={k:torch.stack([o[k] for o in observations]).mean() for k in observations[0]}
        result.update(offset=offset,intersection=intersection)
        if not all(torch.isfinite(v) for v in result.values()):raise FloatingPointError('Invalid native surface loss')
        result={k:_BackwardScale.apply(v,1024.) for k,v in result.items()}
        detail={'crop_tokens':int(mask.sum()),'matched_pred_fraction':float(valid.float().mean()),
                'native_faces_pred':len(fp),'native_faces_target':len(fq),'render_overlap':sum(coverage)/len(coverage),
                'target':'decoded GT HR; native predicted subdivision, no topology substitution'}
        return result,detail
