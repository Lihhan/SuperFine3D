"""Shared DiT: HR local attention + bounded clean LR global memory."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .structured_latent_flow import ElasticSLatFlowModel
from ..modules import sparse as sp
from ..modules.sparse.attention.lr_guidance import PhysicalRotaryEmbedding, LRHRLatentAligner

class LoRALinear(nn.Linear):
    def __init__(self, original, rank=16):
        super().__init__(original.in_features, original.out_features, bias=original.bias is not None, device=original.weight.device, dtype=original.weight.dtype)
        self.weight=original.weight; self.bias=original.bias
        self.lora_A=nn.Parameter(torch.empty(rank,self.in_features,device=self.weight.device,dtype=self.weight.dtype))
        self.lora_B=nn.Parameter(torch.zeros(self.out_features,rank,device=self.weight.device,dtype=self.weight.dtype))
        nn.init.kaiming_uniform_(self.lora_A,a=math.sqrt(5))
    def forward(self,x):
        return F.linear(x,self.weight,self.bias)+F.linear(F.linear(x,self.lora_A),self.lora_B)

def attend(q,k,v):
    if q.is_cuda and q.dtype in (torch.float16,torch.bfloat16):
        from flash_attn import flash_attn_func
        return flash_attn_func(q[None],k[None],v[None],dropout_p=0.0,causal=False)[0]
    return F.scaled_dot_product_attention(q.transpose(0,1)[None],k.transpose(0,1)[None],v.transpose(0,1)[None],dropout_p=0.0)[0].transpose(0,1)

def joint_cells(q,k,v,lr_k,lr_v,routes,lr_layout,chunk_cells=8):
    # Explicit union K/V gives one differentiable softmax. Bound LR duplication.
    import itertools
    outputs=[];indices=[]
    for start in range(0,len(routes),chunk_cells):
        qs=[];ks=[];vs=[];nq=[];nk=[]
        for route in routes[start:start+chunk_cells]:
            batch,idx=route[:2]; kv_idx=route[2] if len(route)==3 else idx
            qs.append(q[idx]);kk=k[kv_idx];vv=v[kv_idx]
            if lr_k is not None:
                sl=lr_layout[batch];kk=torch.cat((kk,lr_k[sl]),0);vv=torch.cat((vv,lr_v[sl]),0)
            ks.append(kk);vs.append(vv);nq.append(len(idx));nk.append(len(kk));indices.append(idx)
        if q.is_cuda and q.dtype in (torch.float16,torch.bfloat16):
            from flash_attn import flash_attn_varlen_func
            cq=torch.tensor([0]+list(itertools.accumulate(nq)),device=q.device,dtype=torch.int32)
            ck=torch.tensor([0]+list(itertools.accumulate(nk)),device=q.device,dtype=torch.int32)
            outputs.append(flash_attn_varlen_func(torch.cat(qs),torch.cat(ks),torch.cat(vs),cq,ck,max(nq),max(nk),dropout_p=0.0,causal=False))
        else:outputs.extend(attend(a,b,c) for a,b,c in zip(qs,ks,vs))
    idx=torch.cat(indices)
    return torch.cat(outputs)[torch.argsort(idx)]

class ElasticDualResolutionSLatFlowModel(ElasticSLatFlowModel):
    def __init__(self,*args,use_lr=True,cell_size=12,lr_grid_size=8,lr_resolution=32,lr_coord_scale=2.0,lr_coord_offset=0.5,lora_rank=16,lr_alignment_ckpt=None,chunk_cells=8,msga=False,msga_ring_mult=(1.0,0.5,0.25),msga_samples_per_ring=16,**kwargs):
        super().__init__(*args,**kwargs)
        if self.pe_mode!='rope':raise ValueError('Requires original 3D RoPE')
        self.msga=bool(msga);self.msga_ring_mult=tuple(float(v) for v in msga_ring_mult);self.msga_samples_per_ring=int(msga_samples_per_ring)
        if any(v<0 for v in self.msga_ring_mult):raise ValueError("negative ring budget")
        self.use_lr=bool(use_lr);self.cell_size=int(cell_size);self.lr_grid_size=int(lr_grid_size)
        self.lr_resolution=int(lr_resolution);self.lr_coord_scale=float(lr_coord_scale);self.lr_coord_offset=float(lr_coord_offset);self.chunk_cells=int(chunk_cells)
        if min(self.cell_size,self.lr_grid_size,self.chunk_cells)<1:raise ValueError('invalid grid/chunk')
        self.physical_rope=PhysicalRotaryEmbedding(self.model_channels//self.num_heads,kwargs.get('rope_freq',(1.0,10000.0)))
        if self.use_lr:
            # LR-only initialization must not change S0/S1 LoRA or loader RNG.
            with torch.random.fork_rng(devices=[]):
                self.lr_aligner=LRHRLatentAligner(self.in_channels,256,3)
            if lr_alignment_ckpt:
                state=torch.load(lr_alignment_ckpt,map_location='cpu',weights_only=True)
                if 'state_dict' in state:state=state['state_dict']
                self.lr_aligner.load_state_dict(state,strict=True)
            self.lr_null=nn.Parameter(torch.zeros(1,self.model_channels,dtype=torch.float32))
        for block in self.blocks:
            for name in ('to_qkv','to_out'):setattr(block.self_attn,name,LoRALinear(getattr(block.self_attn,name),lora_rank))
        for name,p in self.named_parameters():p.requires_grad_('lora_' in name or name=='lr_null')
        for block in self.blocks:
            for attn in (block.self_attn,block.cross_attn):
                if attn.qk_rms_norm:attn.q_rms_norm.float();attn.k_rms_norm.float()

    def _routes(self,x,shift):
        if self.msga:return self._msga_routes(x,shift)
        cells=torch.div(x.coords[:,1:].long()+shift,self.cell_size,rounding_mode='floor')
        keys=torch.cat((x.coords[:,0:1].long(),cells),1)
        unique,inverse=torch.unique(keys,dim=0,sorted=True,return_inverse=True)
        order=torch.argsort(inverse,stable=True);counts=torch.bincount(inverse).tolist()
        return [(int(batch),idx) for batch,idx in zip(unique[:,0].tolist(),order.split(counts))]

    def _msga_routes(self,x,shift):
        # Use Pointelligence's own partition, graph, Hamilton budget and strided
        # representative builder. Cache indices only; K/V remain current-layer.
        from trellis2.third_party.pointelligence.layers.grid_shift_builder import GridShiftBuilder
        from trellis2.third_party.pointelligence.sparse_engines.multi_scale_grid_attention import build_cell_multiscale_idx,_build_fused_kv
        key=f'm0m1_routes_{self.cell_size}_{shift}_{self.msga_ring_mult}_{self.msga_samples_per_ring}'
        cached=x.get_spatial_cache(key)
        if cached is not None:return cached
        prep=GridShiftBuilder(cell_scaler=float(self.cell_size),num_partitions=1).build(
            x.coords[:,1:].float(),x.coords[:,0].long(),
            torch.bincount(x.coords[:,0].long(),minlength=x.shape[0]),grid_size=1.0,
            shift_fractions=[(shift/self.cell_size,)*3])[0]
        graph=build_cell_multiscale_idx(prep,num_octaves=len(self.msga_ring_mult),
            samples_per_ring=self.msga_samples_per_ring,deterministic_coverage=True,full_ring0=True)
        dummy=torch.empty(len(x.coords),1,1,device=x.device)
        gather,_,cu_k,_,_=_build_fused_kv(*graph,
            torch.tensor(self.msga_ring_mult,device=x.device,dtype=torch.float32),
            len(self.msga_ring_mult),prep.cu_seqlens,dummy,dummy,len(prep.cell_batch),x.device,materialize=False)
        cu_q=prep.cu_seqlens.tolist();cu_k=cu_k.tolist()
        routes=[(int(prep.cell_batch[c]),prep.perm[cu_q[c]:cu_q[c+1]].long(),
                 prep.perm[gather[cu_k[c]:cu_k[c+1]].long()].long()) for c in range(len(prep.cell_batch))]
        x.register_spatial_cache(key,routes)
        return routes

    def _pool_lr(self,lr):
        with torch.no_grad():aligned=lr.replace(self.lr_aligner(lr.feats.float()))
        coords=[];feats=[];positions=[]
        for batch,sl in enumerate(lr.layout):
            xyz=lr.coords[sl,1:].float();cells=torch.floor(xyz*self.lr_grid_size/self.lr_resolution).long()
            if (cells<0).any() or (cells>=self.lr_grid_size).any():raise ValueError('LR coordinate bounds')
            unique,inverse=torch.unique(cells,dim=0,sorted=True,return_inverse=True)
            count=torch.bincount(inverse).float()[:,None]
            f=torch.zeros(len(unique),self.in_channels,device=lr.device,dtype=torch.float32)
            pos=torch.zeros(len(unique),3,device=lr.device,dtype=torch.float32)
            f.index_add_(0,inverse,aligned.feats[sl].float());pos.index_add_(0,inverse,xyz)
            coords.append(torch.cat((torch.full((len(unique),1),batch,device=lr.device,dtype=torch.int64),unique),1).int())
            feats.append(f/count);positions.append(pos/count*self.lr_coord_scale+self.lr_coord_offset)
        pooled=sp.SparseTensor(feats=torch.cat(feats),coords=torch.cat(coords))
        return pooled,torch.cat(positions)

    def _qkv(self,attn,x,positions):
        q,k,v=attn.to_qkv(x.feats).reshape(-1,3,self.num_heads,self.model_channels//self.num_heads).unbind(1)
        if attn.qk_rms_norm:q=attn.q_rms_norm(q);k=attn.k_rms_norm(k)
        # Retain the source tensor and its mutation version: pooled continuous
        # positions are not determined by SparseTensor integer coordinates alone.
        # Tensor identity avoids GPU equality/hash synchronizations on cache hits.
        key=f'joint_rope_hd{self.model_channels//self.num_heads}_freq{attn.rope.rope_freq}_scale{self.lr_coord_scale}_offset{self.lr_coord_offset}'
        cached=x.get_spatial_cache(key)
        try:
            position_version=positions._version
        except RuntimeError:
            # Inference tensors have no mutation counter. Recompute rather than
            # trusting an unverifiable cache if an inference tensor is modified.
            position_version=None
        valid=(position_version is not None and isinstance(cached,tuple)
               and len(cached)==3 and cached[0] is positions
               and cached[1]==position_version)
        phase=cached[2] if valid else None
        if phase is None:
            angle=torch.outer(positions.reshape(-1).float(),self.physical_rope.freqs)
            phase=torch.polar(torch.ones_like(angle),angle).reshape(len(positions),-1)
            padding=self.model_channels//self.num_heads//2-phase.shape[-1]
            if padding:phase=torch.cat((phase,torch.ones(len(positions),padding,device=phase.device,dtype=phase.dtype)),1)
            if position_version is not None:
                x.register_spatial_cache(key,(positions,position_version,phase))
        return attn.rope._rotary_embedding(q,phase),attn.rope._rotary_embedding(k,phase),v

    @staticmethod
    def _pre(block,x,mod):
        m=(block.modulation+mod).to(mod.dtype) if block.share_mod else block.adaLN_modulation(mod)
        shift,scale,gate,shift_mlp,scale_mlp,gate_mlp=m.chunk(6,dim=1)
        h=x.replace(block.norm1(x.feats))*(1+scale)+shift
        return h,gate,shift_mlp,scale_mlp,gate_mlp

    @staticmethod
    def _post(block,x,h,gate,shift_mlp,scale_mlp,gate_mlp,cond):
        x=x+h*gate
        x=x+block.cross_attn(x.replace(block.norm2(x.feats)),cond)
        h=x.replace(block.norm3(x.feats))*(1+scale_mlp)+shift_mlp
        return x+block.mlp(h)*gate_mlp

    def _block(self,block,hr,lr,hr_mod,lr_mod,cond,lr_pos,routes,hr_pos=None):
        hh,gate,sh,sc,gm=self._pre(block,hr,hr_mod)
        q,k,v=self._qkv(block.self_attn,hh,hr.coords[:,1:].float() if hr_pos is None else hr_pos)
        lk=lv=None
        if lr is not None:
            lh,lg,lsh,lsc,lgm=self._pre(block,lr,lr_mod)
            lq,lk,lv=self._qkv(block.self_attn,lh,lr_pos)
            lo=torch.cat([attend(lq[sl],lk[sl],lv[sl]) for sl in lr.layout])
            lh=lh.replace(block.self_attn.to_out(lo.reshape(-1,self.model_channels)))
            lr_next=self._post(block,lr,lh,lg,lsh,lsc,lgm,cond)
        else:lr_next=None
        ho=joint_cells(q,k,v,lk,lv,routes,None if lr is None else lr.layout,self.chunk_cells)
        hh=hh.replace(block.self_attn.to_out(ho.reshape(-1,self.model_channels)))
        return self._post(block,hr,hh,gate,sh,sc,gm,cond),lr_next

    def forward(self,x,t,cond,lr_cond=None,lr_drop_mask=None,**kwargs):
        if isinstance(cond,list):cond=sp.VarLenTensor.from_tensor_list(cond)
        hr=self.input_layer(x).to(self.dtype);cond=cond.to(self.dtype)
        hr_mod=self.t_embedder(t)
        if self.share_mod:hr_mod=self.adaLN_modulation(hr_mod)
        hr_mod=hr_mod.to(self.dtype);lr=lr_pos=lr_mod=None
        if self.use_lr:
            if lr_cond is None:raise ValueError('LR latent is required')
            pooled,lr_pos=self._pool_lr(lr_cond);lr=self.input_layer(pooled).to(self.dtype)
            if lr_drop_mask is None:lr_drop_mask=torch.zeros(x.shape[0],device=x.device,dtype=torch.bool)
            drop=lr_drop_mask[lr.coords[:,0].long(),None]
            lr=lr.replace(torch.where(drop,self.lr_null.to(self.dtype).expand_as(lr.feats),lr.feats))
            lr_mod=self.t_embedder(torch.zeros_like(t))
            if self.share_mod:lr_mod=self.adaLN_modulation(lr_mod)
            lr_mod=lr_mod.to(self.dtype)
        hr_pos=x.coords[:,1:].float()
        routes=[self._routes(x,s) for s in (0,self.cell_size//2)]
        for i,block in enumerate(self.blocks):
            route=routes[i%2]
            # Defaults capture immutable values for checkpoint recomputation.
            def run(h,l,hm,lm,c,p,block=block,route=route,hr_pos=hr_pos):return self._block(block,h,l,hm,lm,c,p,route,hr_pos=hr_pos)
            if self.use_checkpoint and torch.is_grad_enabled():
                hr,lr=checkpoint(run,hr,lr,hr_mod,lr_mod,cond,lr_pos,use_reentrant=False)
            else:hr,lr=run(hr,lr,hr_mod,lr_mod,cond,lr_pos)
        hr=hr.to(x.dtype)
        return self.out_layer(hr.replace(F.layer_norm(hr.feats,hr.feats.shape[-1:])))
