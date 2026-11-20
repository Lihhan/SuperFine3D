"""M2: frozen M0 MSGA with a layerwise, clean LR global stream.

    HR <- HR MSGA + LR in a single softmax; LR <- LR global + image condition.
    Only LR-specific adapters are trainable. LR-off exactly follows M0.
"""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .dual_resolution_flow import ElasticDualResolutionSLatFlowModel, attend
from .m2_attention import joint_attention, map_voxel_centers, unique_stable
from ..modules import sparse as sp


class LRDelta(nn.Module):
    """FP32 low-rank delta; the base layer stays registered only in the HR model."""
    def __init__(self, in_features, out_features, rank=16):
        super().__init__()
        if rank < 1:
            raise ValueError('LoRA rank must be positive')
        self.A = nn.Parameter(torch.empty(rank, in_features, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(out_features, rank, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)

    def forward(self, x):
        return F.linear(F.linear(x.float(), self.A), self.B).to(x.dtype)


def _version(t):
    try:
        return t._version
    except RuntimeError:
        return None


def _cond_tensor(cond):
    return cond.feats if isinstance(cond, sp.VarLenTensor) else cond


@dataclass(frozen=True)
class M2LRCache:
    kv: tuple
    layout: tuple
    resolution: int
    condition: torch.Tensor
    condition_layout: tuple
    model_signature: tuple
    source: object
    source_feats: torch.Tensor
    source_coords: torch.Tensor
    drop_mask: torch.Tensor


class ElasticM2SLatFlowModel(ElasticDualResolutionSLatFlowModel):
    def __init__(self, *args, lr_adapter_rank=16, lr_checkpoint_chunks=True, lr_joint_chunk_cells=2,
                 halo_voxels=0, lr_pooling='mean', **kwargs):
        if kwargs.get('use_lr', False):
            raise ValueError('M2 uses its own LR stream; legacy use_lr must be false')
        kwargs['use_lr'] = False
        if not kwargs.get('msga', True):
            raise ValueError('M2 requires MSGA')
        kwargs['msga'] = True
        super().__init__(*args, **kwargs)
        if lr_pooling not in ('mean', 'none'):
            raise ValueError('lr_pooling must be mean or none')
        self.lr_pooling = lr_pooling
        self.lr_checkpoint_chunks = bool(lr_checkpoint_chunks)
        self.lr_joint_chunk_cells = int(lr_joint_chunk_cells)
        if self.lr_joint_chunk_cells < 1:
            raise ValueError('Positive LR joint-attention chunk size required')
        self.halo_voxels = int(halo_voxels)
        if self.halo_voxels not in (0, 1):
            raise ValueError('First M2 implementation supports halo 0 or 1')
        # Fork RNG so adding LR adapters cannot change M0 initialization.
        with torch.random.fork_rng(devices=[]):
            self.lr_input_delta = LRDelta(self.in_channels, self.model_channels, lr_adapter_rank)
            self.lr_qkv_delta = nn.ModuleList([
                LRDelta(self.model_channels, 3 * self.model_channels, lr_adapter_rank)
                for _ in self.blocks])
            # The final LR output is never consumed. Do not register unused parameters.
            self.lr_out_delta = nn.ModuleList([
                LRDelta(self.model_channels, self.model_channels, lr_adapter_rank)
                for _ in self.blocks[:-1]])
        self.lr_null_embedding = nn.Parameter(torch.zeros(1, self.model_channels, dtype=torch.float32))
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith(('lr_input_delta.', 'lr_qkv_delta.',
                                                      'lr_out_delta.', 'lr_null_embedding')))

    def load_m0_adapter(self, path):
        state = torch.load(path, map_location='cpu', weights_only=True)
        expected = {n for n, _ in self.named_parameters() if 'lora_' in n and n.startswith('blocks.')}
        if set(state) != expected:
            raise ValueError(f'M0 adapter mismatch: missing={sorted(expected-set(state))[:8]}, extra={sorted(set(state)-expected)[:8]}')
        self.load_state_dict(state, strict=False)

    def _prepare_lr_input(self, lr, batch_size, drop_mask, hr_resolution):
        if lr.shape[0] != batch_size or len(lr.coords) == 0:
            raise ValueError('LR/HR batch mismatch or empty LR support')
        coords, features, positions = [], [], []
        for batch, sl in enumerate(lr.layout):
            xyz = lr.coords[sl, 1:].float()
            if len(xyz) == 0 or (xyz < 0).any() or (xyz >= self.lr_resolution).any():
                raise ValueError('Each sample requires non-empty, in-bounds LR support')
            if getattr(self, 'lr_pooling', 'mean') == 'none':
                # M3 preserves native feature/coordinate pairs; no centroid tokens.
                coords.append(lr.coords[sl])
                features.append(lr.feats[sl].float())
                positions.append(map_voxel_centers(xyz, self.lr_resolution, hr_resolution))
                continue
            cells = torch.floor(xyz * self.lr_grid_size / self.lr_resolution).long()
            unique, inverse = torch.unique(cells, dim=0, sorted=True, return_inverse=True)
            count = torch.bincount(inverse).float()[:, None]
            pooled = torch.zeros(len(unique), self.in_channels, dtype=torch.float32, device=lr.device)
            center = torch.zeros(len(unique), 3, dtype=torch.float32, device=lr.device)
            pooled.index_add_(0, inverse, lr.feats[sl].float())
            center.index_add_(0, inverse, xyz)
            coords.append(torch.cat((torch.full((len(unique), 1), batch, device=lr.device), unique), dim=1).int())
            features.append(pooled / count)
            positions.append(map_voxel_centers(center / count, self.lr_resolution, hr_resolution))
        pooled = sp.SparseTensor(torch.cat(features), torch.cat(coords))
        embedded = self.input_layer(pooled).to(self.dtype)
        embedded = embedded.replace(embedded.feats + self.lr_input_delta(pooled.feats.to(self.dtype)))
        if drop_mask is None:
            drop_mask = torch.zeros(batch_size, dtype=torch.bool, device=lr.device)
        if drop_mask.shape != (batch_size,):
            raise ValueError('LR drop mask must have shape [batch_size]')
        mask = drop_mask.to(device=lr.device, dtype=torch.bool)[embedded.coords[:, 0].long(), None]
        embedded = embedded.replace(torch.where(mask, self.lr_null_embedding.to(self.dtype), embedded.feats))
        return embedded, torch.cat(positions)

    def _lr_qkv(self, i, hidden, positions):
        attn = self.blocks[i].self_attn
        base = attn.to_qkv(hidden.feats)
        raw = base + self.lr_qkv_delta[i](hidden.feats).to(base.dtype)
        q, k, v = raw.reshape(-1, 3, self.num_heads, self.model_channels // self.num_heads).unbind(1)
        if attn.qk_rms_norm:
            q, k = attn.q_rms_norm(q), attn.k_rms_norm(k)
        return self.physical_rope(q, positions), self.physical_rope(k, positions), v

    def _lr_step(self, i, lr, lr_mod, cond, positions):
        block = self.blocks[i]
        lh, gate, sh, sc, gm = self._pre(block, lr, lr_mod)
        lq, lk, lv = self._lr_qkv(i, lh, positions)
        if i == self.num_blocks - 1:
            return lr, lk, lv
        lo = torch.cat([attend(lq[sl], lk[sl], lv[sl]) for sl in lr.layout])
        flat = lo.reshape(-1, self.model_channels)
        base_out = block.self_attn.to_out(flat)
        lh = lh.replace(base_out + self.lr_out_delta[i](flat).to(base_out.dtype))
        return self._post(block, lr, lh, gate, sh, sc, gm, cond), lk, lv

    def _routes(self, x, shift):
        base = super()._routes(x, shift)
        if self.halo_voxels == 0:
            return base
        cache_key = f'm2_halo1_{self.cell_size}_{shift}_{self.msga}_{self.msga_ring_mult}_{self.msga_samples_per_ring}'
        cached = x.get_spatial_cache(cache_key)
        if cached is not None:
            return cached
        # Only visit the 27 spatially adjacent cells, never scan all HR tokens per cell.
        cells = torch.div(x.coords[:, 1:].long() + shift, self.cell_size, rounding_mode='floor')
        lookup = {}
        for batch, qidx, _ in base:
            key = (batch, *cells[qidx[0]].tolist())
            lookup[key] = qidx
        import itertools
        offsets = tuple(itertools.product((-1, 0, 1), repeat=3))
        result = []
        for batch, qidx, kidx in base:
            cell = cells[qidx[0]]
            key = tuple(cell.tolist())
            neighbors = [lookup[(batch, *(key[j]+offset[j] for j in range(3)))]
                         for offset in offsets if (batch, *(key[j]+offset[j] for j in range(3))) in lookup]
            candidates = torch.cat(neighbors)
            xyz = x.coords[candidates, 1:].long() + shift
            inside = ((xyz >= cell*self.cell_size-1) & (xyz < (cell+1)*self.cell_size+1)).all(1)
            result.append((batch, qidx, unique_stable(torch.cat((kidx, candidates[inside])))))
        x.register_spatial_cache(cache_key, result)
        return result

    def _signature(self):
        tensors = list(self.named_parameters()) + list(self.named_buffers())
        # Inference-created tensors do not have mutation counters. A signature
        # containing None cannot detect parameter updates, so refuse such caches.
        if any(_version(p) is None for _, p in tensors):
            raise ValueError('Model tensors without mutation counters cannot be safely cached')
        return (self.lr_resolution, self.lr_grid_size, self.share_mod, self.lr_pooling,
                tuple((name, id(p), _version(p), str(p.device), str(p.dtype)) for name, p in tensors))

    @staticmethod
    def _condition_layout(cond):
        if isinstance(cond, sp.VarLenTensor):
            return tuple((sl.start, sl.stop) for sl in cond.layout)
        return tuple(cond.shape)

    @torch.no_grad()
    def prepare_lr_cache(self, lr_cond, cond, hr_resolution=None, lr_drop_mask=None):
        if self.training:
            raise ValueError('LR KV cache is inference-only; call eval() first')
        resolution = self.resolution if hr_resolution is None else int(hr_resolution)
        if isinstance(cond, list):
            cond = sp.VarLenTensor.from_tensor_list(cond)
        cond = cond.to(self.dtype)
        lr, positions = self._prepare_lr_input(lr_cond, lr_cond.shape[0], lr_drop_mask, resolution)
        mod = self.t_embedder(torch.zeros(lr.shape[0], device=lr.device))
        if self.share_mod:
            mod = self.adaLN_modulation(mod)
        mod = mod.to(self.dtype)
        layout = tuple(lr.layout)
        entries = []
        for i in range(self.num_blocks):
            lr, lk, lv = self._lr_step(i, lr, mod, cond, positions)
            entries.append((lk, lv))
        mask = (torch.zeros(lr_cond.shape[0], device=lr_cond.device, dtype=torch.bool)
                if lr_drop_mask is None else lr_drop_mask.bool().clone())
        return M2LRCache(tuple(entries), layout, resolution, _cond_tensor(cond).clone(), self._condition_layout(cond), self._signature(),
                         lr_cond, lr_cond.feats.clone(), lr_cond.coords.clone(), mask)

    def _validate_cache(self, cache, lr_cond, cond, resolution, drop_mask, batch_size):
        if self.training or torch.is_grad_enabled():
            raise ValueError('Cannot use inference LR cache with training/autograd')
        if cache.model_signature != self._signature() or cache.resolution != resolution:
            raise ValueError('Stale LR cache: model or resolution changed')
        source = cache.source if lr_cond is None else lr_cond
        mask = (torch.zeros(batch_size, device=source.device, dtype=torch.bool)
                if drop_mask is None else drop_mask.bool())
        if (len(cache.layout) != batch_size or cache.condition_layout != self._condition_layout(cond)
                or not torch.equal(cache.condition, _cond_tensor(cond))
                or not torch.equal(cache.source_feats, source.feats)
                or not torch.equal(cache.source_coords, source.coords)
                or not torch.equal(cache.drop_mask, mask)):
            raise ValueError('LR cache condition/support/dropout mismatch (including CFG branch)')

    def forward(self, x, t, cond, lr_cond=None, lr_drop_mask=None,
                lr_enabled=True, hr_resolution=None, lr_cache=None, **kwargs):
        if kwargs.get('lr_noise') is not None or kwargs.get('concat_cond') is not None:
            raise ValueError('M2 shape stream expects clean LR and no concatenated condition')
        if not lr_enabled:
            return super().forward(x, t, cond, **kwargs)
        resolution = self.resolution if hr_resolution is None else int(hr_resolution)
        if resolution < 1 or (x.coords[:, 1:] < 0).any() or (x.coords[:, 1:] >= resolution).any():
            raise ValueError('HR coordinates outside explicit latent-grid resolution')
        if isinstance(cond, list):
            cond = sp.VarLenTensor.from_tensor_list(cond)
        cond = cond.to(self.dtype)
        hr = self.input_layer(x).to(self.dtype)
        hm = self.t_embedder(t)
        if self.share_mod:
            hm = self.adaLN_modulation(hm)
        hm = hm.to(self.dtype)
        if lr_cache is not None:
            self._validate_cache(lr_cache, lr_cond, cond, resolution, lr_drop_mask, x.shape[0])
            lr, positions, lm = None, None, None
            layout = lr_cache.layout
        else:
            if lr_cond is None:
                raise ValueError('M2 requires clean LR latent or a validated LR cache')
            lr, positions = self._prepare_lr_input(lr_cond, x.shape[0], lr_drop_mask, resolution)
            lm = self.t_embedder(torch.zeros_like(t))
            if self.share_mod:
                lm = self.adaLN_modulation(lm)
            lm = lm.to(self.dtype)
            layout = tuple(lr.layout)
        routes = [self._routes(x, shift) for shift in (0, self.cell_size//2)]
        hp = x.coords[:, 1:].float()
        for i, block in enumerate(self.blocks):
            route = routes[i % 2]

            def run(h, l, hm_, lm_, c, p, i=i, block=block, route=route, hp=hp):
                hh, gate, sh, sc, gm = self._pre(block, h, hm_)
                q, k, v = self._qkv(block.self_attn, hh, hp)
                if lr_cache is None:
                    next_lr, lk, lv = self._lr_step(i, l, lm_, c, p)
                else:
                    next_lr, (lk, lv) = None, lr_cache.kv[i]
                out = joint_attention(q, k, v, lk, lv, route, layout,
                                      self.lr_joint_chunk_cells, self.lr_checkpoint_chunks)
                hh = hh.replace(block.self_attn.to_out(out.reshape(-1, self.model_channels)))
                return self._post(block, h, hh, gate, sh, sc, gm, c), next_lr

            if self.use_checkpoint and torch.is_grad_enabled():
                hr, lr = checkpoint(run, hr, lr, hm, lm, cond, positions, use_reentrant=False)
            else:
                hr, lr = run(hr, lr, hm, lm, cond, positions)
        hr = hr.to(x.dtype)
        return self.out_layer(hr.replace(F.layer_norm(hr.feats, hr.feats.shape[-1:])))
