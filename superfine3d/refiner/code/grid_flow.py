"""HR fixed-cell GridAttention. Complete image-conditioned DiT, no HR mixing."""
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from trellis2.models.tf1_flow import ElasticTF1SLatFlowModel
from trellis2.modules import sparse as sp
from fixed_grid import build_fixed_grid, cell_attention, build_joint_plan, grid_joint_attention


class ElasticTF1GridSLatFlowModel(ElasticTF1SLatFlowModel):
    def __init__(self,*args,grid_use_lr=False,**kwargs):
        kwargs["msga"]=True
        super().__init__(*args,**kwargs)
        self.msga=False
        self.grid_use_lr=bool(grid_use_lr)
        if self.grid_use_lr != (self.experiment_arm != 'm0'):
            raise ValueError('TF0 must use m0 arm; TF1 must train HR and LR adapters')

    def _routes(self,*args,**kwargs):
        raise RuntimeError("Fixed-grid model must never construct MSGA routes")

    def _signature(self):
        return ('fixed_grid_v1',self.cell_size,self.grid_use_lr,super()._signature())

    def _grid(self, x):
        key = f'tf1_fixed_grid_v1_{self.cell_size}'
        prep = x.get_spatial_cache(key)
        if prep is None:
            prep = build_fixed_grid(x.coords, self.cell_size)
            x.register_spatial_cache(key, prep)
        return prep

    def forward(self, x, t, cond, lr_cond=None, lr_enabled=None,
                hr_resolution=None,lr_drop_mask=None,lr_cache=None, **kwargs):
        enabled=self.grid_use_lr if lr_enabled is None else bool(lr_enabled)
        if enabled and not self.grid_use_lr:
            raise ValueError('TF0 has no LR path')
        if kwargs:
            raise ValueError(f'Unsupported arguments: {tuple(kwargs)}')
        resolution = self.resolution if hr_resolution is None else int(hr_resolution)
        if (x.coords[:, 1:] < 0).any() or (x.coords[:, 1:] >= resolution).any():
            raise ValueError('Support outside latent grid')
        if isinstance(cond, list):
            cond = sp.VarLenTensor.from_tensor_list(cond)
        cond = cond.to(self.dtype)
        h = self.input_layer(x).to(self.dtype)
        mod = self.t_embedder(t)
        if self.share_mod:
            mod = self.adaLN_modulation(mod)
        mod = mod.to(self.dtype)
        prep = self._grid(x)
        positions = x.coords[:, 1:].float()
        lr=lp=lm=None; plans=None
        if enabled:
            if lr_cache is not None:
                self._validate_cache(lr_cache,lr_cond,cond,resolution,lr_drop_mask,x.shape[0])
                layout=lr_cache.layout
            else:
                if lr_cond is None: raise ValueError('TF1 requires LR condition')
                lr,lp=self._prepare_lr_input(lr_cond,x.shape[0],lr_drop_mask,resolution)
                lm=self.t_embedder(torch.zeros_like(t))
                if self.share_mod: lm=self.adaLN_modulation(lm)
                lm=lm.to(self.dtype); layout=tuple(lr.layout)
            cachekey=f'fixed_joint_v1_{self.cell_size}_{tuple((s.start,s.stop) for s in layout)}'
            plans=x.get_spatial_cache(cachekey)
            if plans is None:
                plans=build_joint_plan(prep,layout,len(x.coords))
                x.register_spatial_cache(cachekey,plans)
        elif lr_cache is not None:
            raise ValueError('Cannot use LR cache when LR is disabled')
        for i,block in enumerate(self.blocks):
            def run(source,mod_,cond_,lr_,lm_,lp_,block=block,i=i):
                hidden, gate, shift, scale, mlp_gate = self._pre(block, source, mod_)
                q, k, v = self._qkv(block.self_attn, hidden, positions)
                next_lr=None
                if enabled:
                    if lr_cache is None: next_lr,lk,lv=self._lr_step(i,lr_,lm_,cond_,lp_)
                    else: lk,lv=lr_cache.kv[i]
                    out=grid_joint_attention(q,k,v,lk,lv,prep,plans,self.lr_checkpoint_chunks)
                else: out = cell_attention(q, k, v, x.coords, prep)
                hidden = hidden.replace(block.self_attn.to_out(out.flatten(1)))
                return self._post(block, source, hidden, gate, shift, scale, mlp_gate, cond_),next_lr
            if self.use_checkpoint and torch.is_grad_enabled():
                h,lr = checkpoint(run,h,mod,cond,lr,lm,lp,use_reentrant=False)
            else:
                h,lr = run(h,mod,cond,lr,lm,lp)
        h = h.to(x.dtype)
        return self.out_layer(h.replace(F.layer_norm(h.feats, h.feats.shape[-1:])))
