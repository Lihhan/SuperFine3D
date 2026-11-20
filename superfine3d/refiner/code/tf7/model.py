"""TF7: TF6-2 HR routing, native LR anchors plus existing LR context."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from trellis2.modules import sparse as sp
from tf3.baseline.model import ElasticTF2SLatFlowModel
from tf6.near_attention import near_attention
from tf6_2.boundary_attention import build_boundary_plan
from .anchor import PlanCache, apply_plan, AnchorEncoder, AnchoredResidual
from tf5.conditioning import ImageLRFusion, GatedResidual
from tf5.routing import lr_plan, lr_attention


class ElasticTF7SLatFlowModel(ElasticTF2SLatFlowModel):
    def __init__(self,*args,tf5_width=128,tf6_stage="hr_local",near_budget=128,near_chunk_cells=64,boundary_budget=96,**kwargs):
        kwargs['tf2_stage']=tf6_stage
        self.boundary_budget=int(boundary_budget)
        if not 0 <= self.boundary_budget <= int(near_budget): raise ValueError("Invalid boundary K/V budget")
        self.near_budget=int(near_budget)
        self.near_chunk_cells=int(near_chunk_cells)
        super().__init__(*args,**kwargs)
        self.lr_image_fusion=ImageLRFusion(kwargs['in_channels'],kwargs['cond_channels'],self.model_channels,tf5_width)
        self.lr_gated_residual=nn.ModuleList([AnchoredResidual(self.model_channels,tf5_width) for _ in self.blocks])
        self.lr_anchor_encoder=AnchorEncoder(kwargs['in_channels'],tf5_width)
        self._anchor_cache=PlanCache()
        if tf6_stage == 'hr_local':
            for n,p in self.named_parameters():
                p.requires_grad_(n.startswith('blocks.') and 'lora_' in n)
        self.assert_trainable_contract()

    def forward(self,x,t,cond,lr_cond=None,lr_enabled=None,hr_resolution=None,
                lr_drop_mask=None,lr_cache=None,**kwargs):
        if kwargs: raise ValueError(f'Unsupported TF5 arguments: {tuple(kwargs)}')
        if lr_cache is not None:
            raise ValueError('TF2 LR caches are invalid for image-dependent TF5; uncached path required')
        if lr_drop_mask is not None and lr_drop_mask.any(): raise ValueError('TF5 does not use whole-LR dropout')
        resolution=self.resolution if hr_resolution is None else int(hr_resolution)
        if ((x.coords[:,1:]<0)|(x.coords[:,1:]>=resolution)).any(): raise ValueError('HR coordinates outside grid')
        enabled=self.tf2_stage=='lr_match' if lr_enabled is None else bool(lr_enabled)
        if enabled and self.tf2_stage != 'lr_match': raise ValueError('HR-only stage cannot enable LR')
        if isinstance(cond,list): cond=sp.VarLenTensor.from_tensor_list(cond)
        cond=cond.to(self.dtype)
        h=self.input_layer(x).to(self.dtype)
        mod=self.t_embedder(t)
        if self.share_mod: mod=self.adaLN_modulation(mod)
        mod=mod.to(self.dtype)
        prep=self._grid(x); positions=x.coords[:,1:].float()
        cache_key=f'tf6_2_boundary_v1_{self.cell_size}_{self.near_budget}_{self.boundary_budget}_{self.near_chunk_cells}'
        near=x.get_spatial_cache(cache_key)
        if near is None:
            near=build_boundary_plan(x.coords,prep,self.cell_size,self.near_budget,self.near_chunk_cells,self.boundary_budget)
            x.register_spatial_cache(cache_key,near)
        self.near_stats=near.stats
        lr=lp=lm=evidence=coverage=times=plans=anchor=None
        if enabled:
            if lr_cond is None: raise ValueError('TF5 needs native LR, never an implicit GT fallback')
            lr,lp=self._prepare_lr_input(lr_cond,x.shape[0],lr_drop_mask,resolution)
            additions,contexts=[],[]
            dtype=self.lr_image_fusion.lr.weight.dtype
            for b,sl in enumerate(lr_cond.layout):
                image=cond.feats[cond.layout[b]] if hasattr(cond,'feats') else cond[b]
                delta,ctx=self.lr_image_fusion(lr_cond.feats[sl].to(dtype),image.to(dtype))
                additions.append(delta);contexts.append(ctx)
            if len(lr.feats)!=len(lr_cond.feats): raise ValueError('TF5 requires native, unpooled LR tokens')
            lr=lr.replace(lr.feats+torch.cat(additions).to(lr.feats.dtype))
            plan=self._anchor_cache.get(lr_cond.coords,x.coords,self.lr_resolution,resolution)
            native=self.lr_anchor_encoder.project(lr_cond.feats.to(dtype))
            sampled=apply_plan(plan,torch.cat((torch.cat(contexts),native),-1))
            evidence,raw_anchor=sampled.chunk(2,-1)
            coverage=plan.mass
            anchor=self.lr_anchor_encoder.encode(raw_anchor,plan).to(dtype)
            times=torch.cat([(t[b]/1000).expand(sl.stop-sl.start,1) for b,sl in enumerate(x.layout)])
            plans,self.routing_stats=lr_plan(x.coords,lr_cond.coords,prep,resolution,self.lr_resolution,self.lr_context_edge,self.cell_size)
            lm=self.t_embedder(torch.zeros_like(t))
            if self.share_mod: lm=self.adaLN_modulation(lm)
            lm=lm.to(self.dtype)
        for i,block in enumerate(self.blocks):
            def run(source,m,c,l,lm_,lp_,ev,cv,ts,an,i=i,block=block):
                hidden,gate,shift,scale,mg=self._pre(block,source,m)
                q,k,v=self._qkv(block.self_attn,hidden,positions)
                local=near_attention(q,k,v,prep,near)
                out=block.self_attn.to_out(local.flatten(1));next_lr=None
                if enabled:
                    next_lr,lk,lv=self._lr_step(i,l,lm_,c,lp_)
                    read=lr_attention(q,lk,lv,prep,plans).flatten(1)
                    adapter=self.lr_gated_residual[i];dt=adapter.down.weight.dtype
                    out=out+adapter(read.to(dt),ev.to(dt),cv.to(dt),ts.to(dt),an.to(dt),hidden.feats.to(dt)).to(out.dtype)
                hidden=hidden.replace(out)
                return self._post(block,source,hidden,gate,shift,scale,mg,c),next_lr
            if self.use_checkpoint and torch.is_grad_enabled():
                h,lr=checkpoint(run,h,mod,cond,lr,lm,lp,evidence,coverage,times,anchor,use_reentrant=False)
            else:
                h,lr=run(h,mod,cond,lr,lm,lp,evidence,coverage,times,anchor)
        h=h.to(x.dtype)
        return self.out_layer(h.replace(F.layer_norm(h.feats,h.feats.shape[-1:])))
