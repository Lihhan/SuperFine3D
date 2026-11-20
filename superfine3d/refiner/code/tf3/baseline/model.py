"""TF2: frozen HR local interface, spatial LR readout, independent LR FFN LoRA."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from grid_flow import ElasticTF1GridSLatFlowModel
from fixed_grid import cell_attention
from trellis2.models.m2_flow import LRDelta
from trellis2.models.dual_resolution_flow import attend
from trellis2.modules import sparse as sp
from .routing import context_plan, packed_joint


class ElasticTF2SLatFlowModel(ElasticTF1GridSLatFlowModel):
    def __init__(self,*args,tf2_stage='hr_local',lr_context_edge=16,**kwargs):
        if tf2_stage not in ('hr_local','lr_match'):
            raise ValueError('Unknown TF2 stage')
        kwargs['experiment_arm']='m0' if tf2_stage=='hr_local' else 'm2_frozen'
        kwargs['grid_use_lr']=tf2_stage=='lr_match'
        super().__init__(*args,**kwargs)
        self.tf2_stage=tf2_stage
        self.lr_context_edge=int(lr_context_edge)
        rank=kwargs.get('lr_adapter_rank',16)
        with torch.random.fork_rng(devices=[]):
            self.lr_ffn_in_delta=nn.ModuleList([LRDelta(self.model_channels,b.mlp.mlp[0].out_features,rank) for b in self.blocks[:-1]])
            self.lr_ffn_out_delta=nn.ModuleList([LRDelta(b.mlp.mlp[2].in_features,self.model_channels,rank) for b in self.blocks[:-1]])
        for name,p in self.named_parameters():
            lr=name.startswith(('lr_input_delta.','lr_qkv_delta.','lr_out_delta.','lr_ffn_'))
            hr=name.startswith('blocks.') and 'lora_' in name
            p.requires_grad_(lr if tf2_stage=='lr_match' else hr)
        self.assert_trainable_contract()

    def assert_trainable_contract(self):
        if not hasattr(self,'tf2_stage'):
            return super().assert_trainable_contract()
        names=[n for n,p in self.named_parameters() if p.requires_grad]
        expected=('lr_',) if self.tf2_stage=='lr_match' else ('blocks.',)
        assert names and all(n.startswith(expected) for n in names)
        assert not self.lr_null_embedding.requires_grad
        return {'stage':self.tf2_stage,'tensors':len(names),
                'parameters':sum(p.numel() for p in self.parameters() if p.requires_grad)}

    def _signature(self):
        return ('TF2_v1',self.tf2_stage,self.lr_context_edge,super()._signature())

    def set_resolution_pair(self,hr_latent_edge,lr_latent_edge):
        """Explicit inference API: inputs are SLat grid edges, NOT O-Voxel resolution."""
        if hr_latent_edge <= 0 or lr_latent_edge <= 0 or hr_latent_edge % lr_latent_edge:
            raise ValueError('Positive integer HR/LR ratio required')
        if (hr_latent_edge//lr_latent_edge)*self.lr_context_edge % self.cell_size:
            raise ValueError('Context boundaries must align with HR cells')
        self.resolution=int(hr_latent_edge)
        self.lr_resolution=int(lr_latent_edge)

    def _lr_qkv(self,i,hidden,positions):
        # Bypass HR LoRA completely: LR sees W_base + Delta_LR.
        attn=self.blocks[i].self_attn
        layer=attn.to_qkv
        base=F.linear(hidden.feats,layer.weight,layer.bias)
        raw=base+self.lr_qkv_delta[i](hidden.feats).to(base.dtype)
        q,k,v=raw.reshape(-1,3,self.num_heads,self.model_channels//self.num_heads).unbind(1)
        if attn.qk_rms_norm:
            q,k=attn.q_rms_norm(q),attn.k_rms_norm(k)
        return self.physical_rope(q,positions),self.physical_rope(k,positions),v

    def _lr_step(self,i,lr,lm,cond,positions):
        block=self.blocks[i]
        hidden,gate,shift,scale,mlp_gate=self._pre(block,lr,lm)
        q,k,v=self._lr_qkv(i,hidden,positions)
        if i==self.num_blocks-1:
            return lr,k,v
        # Preserve TF1's global LR encoder; HR only reads the spatial context.
        flat=torch.cat([attend(q[s],k[s],v[s]) for s in lr.layout]).flatten(1)
        proj=block.self_attn.to_out
        base=F.linear(flat,proj.weight,proj.bias)
        out=base+self.lr_out_delta[i](flat).to(base.dtype)
        result=lr+hidden.replace(out)*gate
        ff=result.replace(block.norm3(result.feats))*(1+scale)+shift
        first,act,last=block.mlp.mlp
        z=first(ff)
        z=z.replace(z.feats+self.lr_ffn_in_delta[i](ff.feats).to(z.feats.dtype))
        z=act(z)
        y=last(z)
        y=y.replace(y.feats+self.lr_ffn_out_delta[i](z.feats).to(y.feats.dtype))
        return result+y*mlp_gate,k,v

    def forward(self,x,t,cond,lr_cond=None,lr_enabled=None,hr_resolution=None,
                lr_drop_mask=None,lr_cache=None,**kwargs):
        if kwargs:
            raise ValueError(f'Unsupported arguments: {tuple(kwargs)}')
        enabled=self.grid_use_lr if lr_enabled is None else bool(lr_enabled)
        if enabled and not self.grid_use_lr:
            raise ValueError('HR-only stage cannot enable LR')
        if lr_drop_mask is not None and lr_drop_mask.any():
            raise ValueError('TF2 requires lr_dropout=0')
        resolution=self.resolution if hr_resolution is None else int(hr_resolution)
        if ((x.coords[:,1:]<0)|(x.coords[:,1:]>=resolution)).any():
            raise ValueError('HR coordinates out of bounds')
        if isinstance(cond,list): cond=sp.VarLenTensor.from_tensor_list(cond)
        cond=cond.to(self.dtype)
        h=self.input_layer(x).to(self.dtype)
        mod=self.t_embedder(t)
        if self.share_mod: mod=self.adaLN_modulation(mod)
        mod=mod.to(self.dtype)
        prep=self._grid(x)
        positions=x.coords[:,1:].float()
        lr=lp=lm=plans=None
        if enabled:
            if lr_cond is None: raise ValueError('Native LR required even with cache')
            if lr_cache is None:
                lr,lp=self._prepare_lr_input(lr_cond,x.shape[0],lr_drop_mask,resolution)
                lm=self.t_embedder(torch.zeros_like(t))
                if self.share_mod: lm=self.adaLN_modulation(lm)
                lm=lm.to(self.dtype)
            else:
                self._validate_cache(lr_cache,lr_cond,cond,resolution,lr_drop_mask,x.shape[0])
            key=f'tf2_context_{resolution}_{self.lr_resolution}_{self.lr_context_edge}_{self.cell_size}'
            cached=x.get_spatial_cache(key)
            try: version=lr_cond.coords._version
            except RuntimeError: version=None
            if version is not None and cached is not None and cached[0] is lr_cond.coords and cached[1]==version:
                plans,self.routing_stats=cached[2:]
            else:
                plans,self.routing_stats=context_plan(x.coords,lr_cond.coords,prep,resolution,
                    self.lr_resolution,self.lr_context_edge,self.cell_size)
                if version is not None:
                    x.register_spatial_cache(key,(lr_cond.coords,version,plans,self.routing_stats))
        elif lr_cache is not None:
            raise ValueError('LR cache supplied to disabled branch')
        for i,block in enumerate(self.blocks):
            def run(source,m,c,l,lm_,lp_,i=i,block=block):
                hidden,gate,shift,scale,mg=self._pre(block,source,m)
                q,k,v=self._qkv(block.self_attn,hidden,positions)
                next_lr=None
                if enabled:
                    if lr_cache is None: next_lr,lk,lv=self._lr_step(i,l,lm_,c,lp_)
                    else: lk,lv=lr_cache.kv[i]
                    out=packed_joint(q,k,v,lk,lv,prep,plans,self.lr_checkpoint_chunks)
                else: out=cell_attention(q,k,v,x.coords,prep)
                hidden=hidden.replace(block.self_attn.to_out(out.flatten(1)))
                return self._post(block,source,hidden,gate,shift,scale,mg,c),next_lr
            if self.use_checkpoint and torch.is_grad_enabled():
                h,lr=checkpoint(run,h,mod,cond,lr,lm,lp,use_reentrant=False)
            else: h,lr=run(h,mod,cond,lr,lm,lp)
        h=h.to(x.dtype)
        return self.out_layer(h.replace(F.layer_norm(h.feats,h.feats.shape[-1:])))
