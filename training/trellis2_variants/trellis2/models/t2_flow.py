"""Ultra-main-structure transfer: frozen M0 HR, clean LR stream with own LoRA.

    3D adaptations: MSGA/halo HR communication, native sparse LR spatial
    partitions, TRELLIS image cross-attention and voxel-center 3D RoPE.
    This is not bitwise equivalent to the 2D FLUX architecture.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .m2_flow import ElasticM2SLatFlowModel, LRDelta
from .t2_routing import spatial_routes, routed_joint_attention, partition_attention
from ..modules import sparse as sp


def base_linear(module, features):
    """Bypass M0 HR LoRA without mutating its modules or checkpoint state."""
    return F.linear(features, module.weight, module.bias)


def base_plus_delta(module, features, delta):
    """Keep attention activations in base dtype under BF16 autocast."""
    base = base_linear(module, features)
    return base + delta(features).to(base.dtype)


class ElasticT2SLatFlowModel(ElasticM2SLatFlowModel):
    def __init__(self, *args, lr_partition_size=16, lr_adapter_rank=16, fast_inference=True, fast_training=True, **kwargs):
        if kwargs.get('lr_pooling', 'none') != 'none':
            raise ValueError('T2 preserves all native LR tokens')
        kwargs['lr_pooling'] = 'none'
        super().__init__(*args, lr_adapter_rank=lr_adapter_rank, **kwargs)
        self.fast_inference_enabled = bool(fast_inference)
        self.fast_training_enabled = bool(fast_training)
        self._fast_active = False
        self.lr_partition_size = int(lr_partition_size)
        if self.lr_partition_size < 1:
            raise ValueError('Positive LR partition edge required')
        with torch.random.fork_rng(devices=[]):
            self.lr_mlp_in_delta = nn.ModuleList([
                LRDelta(b.mlp.mlp[0].in_features, b.mlp.mlp[0].out_features, lr_adapter_rank)
                for b in self.blocks[:-1]])
            self.lr_mlp_out_delta = nn.ModuleList([
                LRDelta(b.mlp.mlp[2].in_features, b.mlp.mlp[2].out_features, lr_adapter_rank)
                for b in self.blocks[:-1]])
            # Ultra PEFT gaussian initialization, alpha/r = 1.
            for name, module in self.named_modules():
                if name.startswith('lr_') and isinstance(module, LRDelta):
                    nn.init.normal_(module.A, std=1/lr_adapter_rank)
        # Last LR output/FFN is not consumed by HR; omit unused parameters.
        for name, p in self.named_parameters():
            p.requires_grad_(name.startswith(('lr_input_delta.', 'lr_qkv_delta.',
                'lr_out_delta.', 'lr_mlp_in_delta.', 'lr_mlp_out_delta.')))
        self.assert_trainable_contract()

    def _apply(self, fn, recurse=True):
        # Device/dtype transfers must release geometry caches as well as weights.
        if hasattr(self, '_fast_backend'):
            del self._fast_backend
        self._fast_active = False
        return super()._apply(fn, recurse=recurse)

    def assert_trainable_contract(self):
        names = [n for n,p in self.named_parameters() if p.requires_grad]
        if not names or any(not n.startswith('lr_') or 'lora_' in n for n in names):
            raise ValueError('T2 must train only fresh LR deltas')
        expected = 2 + 2*self.num_blocks + 6*(self.num_blocks-1)
        if len(names) != expected:
            raise ValueError(f'T2 trainable tensor mismatch: {len(names)} != {expected}')
        return {'arm': 't2', 'hr_tensors': 0, 'lr_tensors': len(names),
                'lr_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)}

    def _lr_qkv(self, i, hidden, positions):
        attn = self.blocks[i].self_attn
        raw = base_plus_delta(attn.to_qkv, hidden.feats, self.lr_qkv_delta[i])
        q,k,v = raw.reshape(-1,3,self.num_heads,self.model_channels//self.num_heads).unbind(1)
        if attn.qk_rms_norm:
            q,k = attn.q_rms_norm(q),attn.k_rms_norm(k)
        return self.physical_rope(q, positions),self.physical_rope(k, positions),v

    def _lr_step(self, i, lr, lr_mod, cond, positions, partition_fn=partition_attention):
        b = self.blocks[i]
        h,gate,sh,sc,gm = self._pre(b,lr,lr_mod)
        q,k,v = self._lr_qkv(i,h,positions)
        if i == self.num_blocks-1:
            return lr,k,v
        out = partition_fn(q,k,v,lr.coords,self.lr_resolution,self.lr_partition_size)
        flat = out.reshape(-1,self.model_channels)
        h = h.replace(base_plus_delta(b.self_attn.to_out,flat,self.lr_out_delta[i]))
        x = lr+h*gate
        x = x+b.cross_attn(x.replace(b.norm2(x.feats)),cond)
        h = x.replace(b.norm3(x.feats))*(1+sc)+sh
        inner = base_plus_delta(b.mlp.mlp[0],h.feats,self.lr_mlp_in_delta[i])
        # Preserve the original activation rather than assuming its GELU settings.
        activated = b.mlp.mlp[1](h.replace(inner)).feats
        output = base_plus_delta(b.mlp.mlp[2],activated,self.lr_mlp_out_delta[i])
        return x+h.replace(output)*gm,k,v

    def prepare_lr_cache(self, *args, **kwargs):
        raise ValueError('T2 first validation disables LR caching')

    def forward(self,x,t,cond,lr_cond=None,lr_drop_mask=None,lr_enabled=True,
                hr_resolution=None,lr_cache=None,**kwargs):
        if lr_cache is not None or kwargs.get('lr_noise') is not None or kwargs.get('concat_cond') is not None:
            raise ValueError('T2 requires clean native LR, no cached/noisy/concatenated LR')
        if not lr_enabled:
            return super().forward(x,t,cond,lr_enabled=False,**kwargs)
        if lr_cond is None:
            raise ValueError('T2 requires a clean LR latent')
        resolution = self.resolution if hr_resolution is None else int(hr_resolution)
        if isinstance(cond,list):
            cond = sp.VarLenTensor.from_tensor_list(cond)
        cond = cond.to(self.dtype)
        hr = self.input_layer(x).to(self.dtype)
        hm = self.t_embedder(t)
        lm = self.t_embedder(torch.zeros_like(t))
        if self.share_mod:
            hm,lm = self.adaLN_modulation(hm),self.adaLN_modulation(lm)
        hm,lm = hm.to(self.dtype),lm.to(self.dtype)
        lr,positions = self._prepare_lr_input(lr_cond,x.shape[0],lr_drop_mask,resolution)
        self._fast_active = self.fast_inference_enabled and not self.training and not torch.is_grad_enabled()
        route_fn = spatial_routes
        attention_fn = routed_joint_attention
        partition_fn = partition_attention
        if self._fast_active:
            if not hasattr(self,'_fast_backend'):
                from .t2_fast_inference import FastT2Inference
                self._fast_backend = FastT2Inference()
            self._fast_backend.begin(x.coords,lr.coords,resolution,self.lr_resolution)
            route_fn = self._fast_backend.routes
            attention_fn = self._fast_backend.attention
            partition_fn = self._fast_backend.partition_attention
        elif self.fast_training_enabled and torch.is_grad_enabled():
            from .t2_training_attention import TrainingGeometry
            # Owned by this graph: never overwrite indices needed by an older
            # microbatch's activation-checkpoint recomputation.
            geometry = TrainingGeometry(lr.coords,self.lr_resolution,self.lr_partition_size)
            route_fn = geometry.routes
            attention_fn = geometry.attention
            partition_fn = geometry.partition_attention
        routes = [route_fn(x.coords,lr.coords,self._routes(x,s),resolution,
                  self.lr_resolution,self.lr_partition_size) for s in (0,self.cell_size//2)]
        hp = x.coords[:,1:].float()
        for i,b in enumerate(self.blocks):
            def run(h,l,hm_,lm_,c,p,i=i,b=b,route=routes[i%2],
                    attention_fn=attention_fn,partition_fn=partition_fn):
                hh,gate,sh,sc,gm = self._pre(b,h,hm_)
                q,k,v = self._qkv(b.self_attn,hh,hp)
                next_lr,lk,lv = self._lr_step(i,l,lm_,c,p,partition_fn)
                out = attention_fn(q,k,v,lk,lv,route,
                    self.lr_joint_chunk_cells,self.lr_checkpoint_chunks)
                hh = hh.replace(b.self_attn.to_out(out.reshape(-1,self.model_channels)))
                return self._post(b,h,hh,gate,sh,sc,gm,c),next_lr
            if self.use_checkpoint and torch.is_grad_enabled():
                hr,lr = checkpoint(run,hr,lr,hm,lm,cond,positions,use_reentrant=False)
            else:
                hr,lr = run(hr,lr,hm,lm,cond,positions)
        hr = hr.to(x.dtype)
        return self.out_layer(hr.replace(F.layer_norm(hr.feats,hr.feats.shape[-1:])))
