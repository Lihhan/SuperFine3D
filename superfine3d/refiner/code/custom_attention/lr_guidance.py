"""Efficient LR latent guidance for sparse Shape SLat generation."""

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .. import SparseTensor, VarLenTensor, SparseConv3d, SparseDownsample
from .full_attn import sparse_scaled_dot_product_attention
from .modules import SparseMultiHeadRMSNorm


def _fp32_layer_norm(module, x):
    return F.layer_norm(
        x.float(), module.normalized_shape,
        module.weight.float() if module.weight is not None else None,
        module.bias.float() if module.bias is not None else None,
        module.eps,
    ).to(x.dtype)


def smoothstep_schedule(t, timestep_scale, low, high, minimum):
    normalized = float(t.detach().float().mean().item() / timestep_scale)
    u = min(1.0, max(0.0, (normalized - low) / (high - low)))
    return minimum + (1.0 - minimum) * (u * u * (3.0 - 2.0 * u))


def pool_sparse_lr_memory(x, coords, grid_size, lr_resolution):
    """Deterministically mean-pool LR tokens into at most grid_size**3 bins."""
    feats_out, coords_out, lengths = [], [], []
    for layout in x.layout:
        feat = x.feats[layout]
        xyz = coords[layout, 1:].float()
        cell = torch.floor(xyz * grid_size / lr_resolution).long().clamp_(0, grid_size - 1)
        key = cell[:, 0] * grid_size * grid_size + cell[:, 1] * grid_size + cell[:, 2]
        unique, inverse = torch.unique(key, sorted=True, return_inverse=True)
        count = torch.bincount(inverse, minlength=unique.numel()).to(feat.dtype).unsqueeze(-1)
        pooled_feat = torch.zeros(unique.numel(), feat.shape[-1], device=feat.device, dtype=feat.dtype)
        pooled_feat.index_add_(0, inverse, feat)
        pooled_xyz = torch.zeros(unique.numel(), 3, device=xyz.device, dtype=xyz.dtype)
        pooled_xyz.index_add_(0, inverse, xyz)
        feats_out.append(pooled_feat / count)
        coords_out.append(pooled_xyz / count.float())
        lengths.append(unique.numel())
    layout = VarLenTensor.layout_from_seqlen(lengths)
    return VarLenTensor(torch.cat(feats_out), layout), VarLenTensor(torch.cat(coords_out), layout)


class PhysicalRotaryEmbedding(nn.Module):
    """Apply a common physical-coordinate 3D RoPE to HR queries and LR keys."""
    def __init__(self, head_dim, rope_freq=(1.0, 10000.0)):
        super().__init__()
        if head_dim % 2:
            raise ValueError("head_dim must be even")
        self.head_dim = int(head_dim)
        freq_dim = head_dim // 2 // 3
        freq = torch.arange(freq_dim, dtype=torch.float32) / freq_dim
        self.register_buffer("freqs", rope_freq[0] / (rope_freq[1] ** freq), persistent=False)

    def forward(self, x, xyz):
        phases = torch.outer(xyz.reshape(-1).float(), self.freqs)
        phases = torch.polar(torch.ones_like(phases), phases).reshape(xyz.shape[0], -1)
        target = self.head_dim // 2
        if phases.shape[-1] < target:
            phases = torch.cat([
                phases,
                torch.ones(
                    phases.shape[0], target - phases.shape[-1],
                    dtype=phases.dtype, device=phases.device,
                ),
            ], dim=-1)
        xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        return torch.view_as_real(xc * phases.unsqueeze(-2)).reshape_as(x).to(x.dtype)


class LRHRLatentAligner(nn.Module):
    """Residual feature mapper pretrained on coordinate-aligned LR/HR latents."""
    def __init__(self, channels=32, hidden_channels=256, depth=3):
        super().__init__()
        if depth < 2:
            raise ValueError("alignment depth must be at least 2")
        layers = [nn.LayerNorm(channels), nn.Linear(channels, hidden_channels),
                  nn.SiLU()]
        for _ in range(depth - 2):
            layers.extend([
                nn.LayerNorm(hidden_channels),
                nn.Linear(hidden_channels, hidden_channels),
                nn.SiLU(),
            ])
        layers.extend([nn.LayerNorm(hidden_channels),
                       nn.Linear(hidden_channels, channels)])
        self.net = nn.Sequential(*layers)
        self.gate = nn.Parameter(torch.ones((), dtype=torch.float32))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features):
        delta = self.net(features)
        return features + self.gate.to(delta.dtype) * delta


class SparseLRGlobalAttention(nn.Module):
    """HR-to-bounded-LR cross-attention; never materializes HR full attention."""
    def __init__(self, channels, num_heads, rope_freq=(1.0, 10000.0),
                 gate_init=0.0, zero_init_output=False,
                 attention_temperature=1.0, coord_offset=0.0):
        super().__init__()
        if channels % num_heads:
            raise ValueError("channels must be divisible by num_heads")
        self.channels, self.num_heads = channels, num_heads
        self.head_dim = channels // num_heads
        self.attention_temperature = float(attention_temperature)
        self.coord_offset = float(coord_offset)
        if self.attention_temperature <= 0:
            raise ValueError('attention_temperature must be positive')
        self.norm_hr, self.norm_lr = nn.LayerNorm(channels), nn.LayerNorm(channels)
        self.to_q, self.to_kv = nn.Linear(channels, channels), nn.Linear(channels, channels * 2)
        self.q_norm = SparseMultiHeadRMSNorm(self.head_dim, num_heads)
        self.k_norm = SparseMultiHeadRMSNorm(self.head_dim, num_heads)
        self.rope = PhysicalRotaryEmbedding(self.head_dim, rope_freq)
        self.to_out = nn.Linear(channels, channels)
        if zero_init_output:
            nn.init.zeros_(self.to_out.weight)
            if self.to_out.bias is not None:
                nn.init.zeros_(self.to_out.bias)
        self.gate = nn.Parameter(torch.tensor(float(gate_init), dtype=torch.float32))

    def forward(self, hr, memory, memory_xyz, lr_coord_scale, strength):
        q = self.to_q(_fp32_layer_norm(self.norm_hr, hr.feats)).reshape(-1, self.num_heads, self.head_dim)
        kv = self.to_kv(_fp32_layer_norm(self.norm_lr, memory.feats)).reshape(-1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(dim=1)
        q = self.q_norm(hr.replace(q))
        q = q.replace(q.feats / self.attention_temperature)
        k, v = memory.replace(k), memory.replace(v)
        k = self.k_norm(k)
        q = q.replace(self.rope(q.feats, hr.coords[:, 1:].float()))
        k = k.replace(self.rope(k.feats, memory_xyz.feats * lr_coord_scale + self.coord_offset))
        out = sparse_scaled_dot_product_attention(q, k, v)
        out = self.to_out(out.feats.reshape(-1, self.channels))
        return hr.replace(hr.feats + (self.gate * strength).to(out.dtype) * out)


class SparseLRContextBlock(nn.Module):
    """A small pre-norm Transformer block for the bounded LR memory.

    The block operates only on the pooled LR memory (at most ``grid_size**3``
    tokens), so full self-attention is cheap.  Zero-initialized residual gates
    make the initial mapping exactly the identity; this preserves experiment B
    at initialization while allowing contextualization to be learned.
    """
    def __init__(self, channels, num_heads, mlp_ratio=2.0,
                 rope_freq=(1.0, 10000.0), residual_gate_init=0.0):
        super().__init__()
        if channels % num_heads:
            raise ValueError("channels must be divisible by num_heads")
        hidden = max(1, int(round(channels * float(mlp_ratio))))
        self.channels, self.num_heads = int(channels), int(num_heads)
        self.head_dim = self.channels // self.num_heads
        self.norm1, self.norm2 = nn.LayerNorm(channels), nn.LayerNorm(channels)
        self.to_qkv = nn.Linear(channels, channels * 3)
        self.q_norm = SparseMultiHeadRMSNorm(self.head_dim, num_heads)
        self.k_norm = SparseMultiHeadRMSNorm(self.head_dim, num_heads)
        self.rope = PhysicalRotaryEmbedding(self.head_dim, rope_freq)
        self.to_out = nn.Linear(channels, channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden), nn.GELU(), nn.Linear(hidden, channels)
        )
        self.attn_gate = nn.Parameter(torch.tensor(
            float(residual_gate_init), dtype=torch.float32
        ))
        self.mlp_gate = nn.Parameter(torch.tensor(
            float(residual_gate_init), dtype=torch.float32
        ))

    def forward(self, memory, memory_xyz, coord_scale):
        residual = memory.feats
        qkv = self.to_qkv(_fp32_layer_norm(self.norm1, residual)).reshape(
            -1, 3, self.num_heads, self.head_dim
        )
        q, k, v = qkv.unbind(dim=1)
        q = self.q_norm(memory.replace(q))
        k = self.k_norm(memory.replace(k))
        physical_xyz = memory_xyz.feats * coord_scale
        q = q.replace(self.rope(q.feats, physical_xyz))
        k = k.replace(self.rope(k.feats, physical_xyz))
        attn = sparse_scaled_dot_product_attention(q, k, memory.replace(v))
        attn = self.to_out(attn.feats.reshape(-1, self.channels))
        memory = memory.replace(
            residual + self.attn_gate.to(attn.dtype) * attn
        )
        ff = self.mlp(_fp32_layer_norm(self.norm2, memory.feats))
        return memory.replace(
            memory.feats + self.mlp_gate.to(ff.dtype) * ff
        )


class SparseLRContextEncoder(nn.Module):
    """Bottleneck LR Transformer whose cost is independent of HR token count."""
    def __init__(self, input_channels, context_channels=256, num_heads=8,
                 depth=2, mlp_ratio=2.0, rope_freq=(1.0, 10000.0)):
        super().__init__()
        if context_channels % num_heads:
            raise ValueError("context_channels must be divisible by num_heads")
        self.depth = int(depth)
        self.norm_in = nn.LayerNorm(input_channels)
        self.to_context = nn.Linear(input_channels, context_channels)
        self.blocks = nn.ModuleList([
            SparseLRContextBlock(
                context_channels, num_heads, mlp_ratio, rope_freq,
                residual_gate_init=1.0,
            ) for _ in range(self.depth)
        ])
        self.norm_out = nn.LayerNorm(context_channels)
        self.to_output = nn.Linear(context_channels, input_channels)
        # Preserve the pretrained denoiser exactly at initialization while
        # avoiding a dead upstream path: step 1 learns to_output, then its
        # non-zero weights pass gradients into the context stack on step 2.
        nn.init.zeros_(self.to_output.weight)
        if self.to_output.bias is not None:
            nn.init.zeros_(self.to_output.bias)
        self.gate = nn.Parameter(torch.ones((), dtype=torch.float32))

    def forward(self, memory, memory_xyz, coord_scale):
        if self.depth == 0:
            return memory
        h = memory.replace(self.to_context(
            _fp32_layer_norm(self.norm_in, memory.feats)
        ))
        for block in self.blocks:
            h = block(h, memory_xyz, coord_scale)
        delta = self.to_output(_fp32_layer_norm(self.norm_out, h.feats))
        return memory.replace(
            memory.feats + self.gate.to(delta.dtype) * delta
        )


class SparseLRSynchronizedGlobalEncoder(nn.Module):
    """Lightweight LR global stream explicitly conditioned on the HR timestep."""
    def __init__(self, input_channels, timestep_channels,
                 context_channels=256, num_heads=8, depth=2,
                 mlp_ratio=2.0, rope_freq=(1.0, 10000.0)):
        super().__init__()
        self.norm_in = nn.LayerNorm(input_channels)
        self.to_context = nn.Linear(input_channels, context_channels)
        self.time_proj = nn.Sequential(
            nn.SiLU(), nn.Linear(timestep_channels, context_channels)
        )
        self.blocks = nn.ModuleList([
            SparseLRContextBlock(
                context_channels, num_heads, mlp_ratio, rope_freq,
                residual_gate_init=1.0,
            ) for _ in range(int(depth))
        ])
        self.norm_out = nn.LayerNorm(context_channels)
        self.to_output = nn.Linear(context_channels, input_channels)
        nn.init.zeros_(self.to_output.weight)
        nn.init.zeros_(self.to_output.bias)
        self.gate = nn.Parameter(torch.ones((), dtype=torch.float32))

    def forward(self, memory, memory_xyz, coord_scale, timestep_embedding):
        h = memory.replace(self.to_context(
            _fp32_layer_norm(self.norm_in, memory.feats)
        ))
        time_input = timestep_embedding.to(self.time_proj[1].weight.dtype)
        time = self.time_proj(time_input).to(h.feats.dtype)
        time_tokens = torch.cat([
            time[i:i + 1].expand(layout.stop - layout.start, -1)
            for i, layout in enumerate(memory.layout)
        ], dim=0)
        h = h.replace(h.feats + time_tokens)
        for block in self.blocks:
            h = block(h, memory_xyz, coord_scale)
        delta = self.to_output(_fp32_layer_norm(self.norm_out, h.feats))
        return memory.replace(
            memory.feats + self.gate.to(delta.dtype) * delta
        )


class SparseLRPyramidEncoder(nn.Module):
    """Learned two-stage sparse compression for global LR guidance.

    Features are locally mixed before each 2x reduction.  The final 8^3
    support matches the deterministic baseline memory, while density and
    within-cell geometry statistics retain information discarded by a single
    mean pool.  A zero-initialized output keeps the pretrained model exact at
    initialization.
    """
    def __init__(self, input_channels, context_channels=256, num_heads=8,
                 depth=2, mlp_ratio=2.0, rope_freq=(1.0, 10000.0)):
        super().__init__()
        self.norm_in = nn.LayerNorm(input_channels)
        self.to_context = nn.Linear(input_channels, context_channels)
        self.level_embed = nn.Parameter(torch.zeros(3, context_channels))
        self.norm0 = nn.LayerNorm(context_channels)
        self.norm1 = nn.LayerNorm(context_channels)
        self.mix0 = SparseConv3d(context_channels, context_channels, 3)
        self.mix1 = SparseConv3d(context_channels, context_channels, 3)
        self.down0 = SparseDownsample(2)
        self.down1 = SparseDownsample(2)
        self.stats_mlp = nn.Sequential(
            nn.Linear(7, context_channels), nn.SiLU(),
            nn.Linear(context_channels, context_channels),
        )
        self.blocks = nn.ModuleList([
            SparseLRContextBlock(
                context_channels, num_heads, mlp_ratio, rope_freq,
                residual_gate_init=1.0,
            ) for _ in range(int(depth))
        ])
        self.norm_out = nn.LayerNorm(context_channels)
        self.to_output = nn.Linear(context_channels, input_channels)
        nn.init.zeros_(self.to_output.weight)
        if self.to_output.bias is not None:
            nn.init.zeros_(self.to_output.bias)
        self.gate = nn.Parameter(torch.ones((), dtype=torch.float32))

    @staticmethod
    def _coarse_stats(coords, grid_size=8, lr_resolution=32):
        xyz = coords[:, 1:].long()
        batch = coords[:, 0].long()
        factor = lr_resolution // grid_size
        cell = torch.div(xyz, factor, rounding_mode="floor").clamp_(0, grid_size - 1)
        key = batch * (grid_size ** 3) + cell[:, 0] * (grid_size ** 2) + cell[:, 1] * grid_size + cell[:, 2]
        unique, inverse = torch.unique(key, sorted=True, return_inverse=True)
        count = torch.bincount(inverse, minlength=unique.numel()).float()
        local = (xyz.float() % factor) / max(factor - 1, 1)
        mean = torch.zeros(unique.numel(), 3, device=coords.device)
        mean.index_add_(0, inverse, local)
        mean = mean / count[:, None]
        second = torch.zeros_like(mean)
        second.index_add_(0, inverse, local.square())
        var = (second / count[:, None] - mean.square()).clamp_min_(0)
        density = torch.log1p(count) / torch.log(torch.tensor(float(factor ** 3 + 1), device=coords.device))
        return torch.cat([density[:, None], mean, var], dim=-1)

    def forward(self, lr_h, baseline, memory_xyz, coord_scale):
        h = lr_h.replace(self.to_context(_fp32_layer_norm(self.norm_in, lr_h.feats)))
        h = h.replace(h.feats + self.level_embed[0].to(h.feats.dtype))
        h = self.mix0(h.replace(F.silu(_fp32_layer_norm(self.norm0, h.feats))))
        h = self.down0(h)
        h = h.replace(h.feats + self.level_embed[1].to(h.feats.dtype))
        h = self.mix1(h.replace(F.silu(_fp32_layer_norm(self.norm1, h.feats))))
        h = self.down1(h)
        stats = self._coarse_stats(lr_h.coords).to(h.feats.dtype)
        if stats.shape[0] != h.feats.shape[0]:
            raise RuntimeError("LR pyramid support does not match coarse statistics")
        h = h.replace(h.feats + self.level_embed[2].to(h.feats.dtype) + self.stats_mlp(stats))
        coarse = VarLenTensor(h.feats, h.layout)
        for block in self.blocks:
            coarse = block(coarse, memory_xyz, coord_scale)
        delta = self.to_output(_fp32_layer_norm(self.norm_out, coarse.feats))
        if delta.shape != baseline.feats.shape:
            raise RuntimeError("LR pyramid support does not match baseline memory")
        return baseline.replace(
            baseline.feats + self.gate.to(delta.dtype) * delta
        )


class SparseLRSelectedAttention(nn.Module):
    """Select coarse LR blocks per HR cell, then attend original fine LR tokens."""
    def __init__(self, channels, num_heads, rope_freq=(1.0, 10000.0),
                 topk_blocks=4, tokens_per_block=16, hr_cell_size=8,
                 lr_block_size=4, chunk_size=1024):
        super().__init__()
        self.channels, self.num_heads = int(channels), int(num_heads)
        self.head_dim = self.channels // self.num_heads
        self.topk_blocks = int(topk_blocks)
        self.tokens_per_block = int(tokens_per_block)
        self.hr_cell_size = int(hr_cell_size)
        self.lr_block_size = int(lr_block_size)
        self.chunk_size = int(chunk_size)
        self.rope = PhysicalRotaryEmbedding(self.head_dim, rope_freq)
        self.to_out = nn.Linear(channels, channels)
        nn.init.zeros_(self.to_out.weight)
        if self.to_out.bias is not None:
            nn.init.zeros_(self.to_out.bias)
        self.gate = nn.Parameter(torch.ones((), dtype=torch.float32))

    @staticmethod
    def _hash(cell, grid=8):
        return cell[:, 0] * grid * grid + cell[:, 1] * grid + cell[:, 2]

    def _representative_table(self, fine_xyz, coarse_xyz):
        fine_key = self._hash(torch.div(
            fine_xyz.long(), self.lr_block_size, rounding_mode="floor"
        ).clamp_(0, 7))
        coarse_key = self._hash(torch.div(
            coarse_xyz.long(), self.lr_block_size, rounding_mode="floor"
        ).clamp_(0, 7))
        order = torch.argsort(fine_key, stable=True)
        sorted_key = fine_key[order]
        table = torch.zeros(
            coarse_key.numel(), self.tokens_per_block,
            dtype=torch.long, device=fine_xyz.device,
        )
        valid = torch.zeros_like(table, dtype=torch.bool)
        left = torch.searchsorted(sorted_key, coarse_key, side="left")
        right = torch.searchsorted(sorted_key, coarse_key, side="right")
        for i in range(coarse_key.numel()):
            n = int((right[i] - left[i]).item())
            if n == 0:
                continue
            take = min(n, self.tokens_per_block)
            if n <= self.tokens_per_block:
                pos = torch.arange(n, device=fine_xyz.device)
            else:
                pos = torch.linspace(0, n - 1, self.tokens_per_block,
                                     device=fine_xyz.device).round().long()
            table[i, :take] = order[left[i] + pos[:take]]
            valid[i, :take] = True
        return table, valid, coarse_key

    def forward(self, hr, lr, memory, memory_xyz, projection,
                lr_coord_scale, strength):
        q = projection.to_q(_fp32_layer_norm(
            projection.norm_hr, hr.feats
        )).reshape(-1, self.num_heads, self.head_dim)
        coarse_kv = projection.to_kv(_fp32_layer_norm(
            projection.norm_lr, memory.feats
        )).reshape(-1, 2, self.num_heads, self.head_dim)
        fine_kv = projection.to_kv(_fp32_layer_norm(
            projection.norm_lr, lr.feats
        )).reshape(-1, 2, self.num_heads, self.head_dim)
        coarse_k, _ = coarse_kv.unbind(dim=1)
        fine_k, fine_v = fine_kv.unbind(dim=1)
        q = projection.q_norm(hr.replace(q)).feats
        coarse_k = projection.k_norm(memory.replace(coarse_k)).feats
        fine_k = projection.k_norm(lr.replace(fine_k)).feats
        q = self.rope(q, hr.coords[:, 1:].float())
        coarse_k = self.rope(coarse_k, memory_xyz.feats * lr_coord_scale)
        fine_k = self.rope(fine_k, lr.coords[:, 1:].float() * lr_coord_scale)

        outputs = []
        scale = self.head_dim ** -0.5
        for hs, ls, ms in zip(hr.layout, lr.layout, memory.layout):
            q_s, ck_s = q[hs], coarse_k[ms]
            fk_s, fv_s = fine_k[ls], fine_v[ls]
            hxyz = hr.coords[hs, 1:].long()
            fxyz = lr.coords[ls, 1:].long()
            cxyz = memory_xyz.feats[ms]
            hcell = torch.div(hxyz, self.hr_cell_size, rounding_mode="floor").clamp_(0, 7)
            hkey = self._hash(hcell)
            unique_hkey, inverse = torch.unique(hkey, sorted=True, return_inverse=True)
            cell_q = torch.zeros(
                unique_hkey.numel(), self.num_heads, self.head_dim,
                device=q_s.device, dtype=q_s.dtype,
            )
            cell_q.index_add_(0, inverse, q_s)
            count = torch.bincount(inverse, minlength=unique_hkey.numel()).to(q_s.dtype)
            cell_q = cell_q / count[:, None, None]
            with torch.no_grad():
                score = torch.einsum("chd,mhd->cmh", cell_q, ck_s).mean(-1) * scale
                k = min(self.topk_blocks, ck_s.shape[0])
                selected = score.topk(k, dim=-1).indices
                table, table_valid, coarse_key = self._representative_table(fxyz, cxyz)
                pos = torch.searchsorted(coarse_key, unique_hkey)
                safe = pos.clamp(max=max(0, coarse_key.numel() - 1))
                parent_valid = (pos < coarse_key.numel())
                if coarse_key.numel():
                    parent_valid &= coarse_key[safe] == unique_hkey
                    # Put the aligned parent first without duplicating it when
                    # it was already selected by the learned coarse score.
                    for row in range(selected.shape[0]):
                        if not bool(parent_valid[row]):
                            continue
                        parent = safe[row]
                        hit = torch.nonzero(selected[row] == parent, as_tuple=False)
                        if hit.numel() and int(hit[0, 0]) != 0:
                            j = int(hit[0, 0])
                            first = selected[row, 0].clone()
                            selected[row, 0] = parent
                            selected[row, j] = first
                        elif not hit.numel():
                            selected[row, -1] = parent
                cell_index = table[selected].reshape(unique_hkey.numel(), -1)
                cell_valid = table_valid[selected].reshape(unique_hkey.numel(), -1)
                token_index, token_valid = cell_index[inverse], cell_valid[inverse]
            for start in range(0, q_s.shape[0], self.chunk_size):
                stop = min(start + self.chunk_size, q_s.shape[0])
                idx = token_index[start:stop]
                mask = token_valid[start:stop]
                def attend(q_chunk, k_source, v_source, idx=idx, mask=mask):
                    kc, vc = k_source[idx], v_source[idx]
                    scores = torch.einsum("bhd,bkhd->bhk", q_chunk, kc) * scale
                    scores = scores.masked_fill(~mask[:, None, :], -torch.inf)
                    has_token = mask.any(dim=-1)
                    scores[~has_token] = 0
                    weight = torch.softmax(scores.float(), dim=-1).to(vc.dtype)
                    weight = weight * mask[:, None, :].to(weight.dtype)
                    return torch.einsum("bhk,bkhd->bhd", weight, vc)
                outputs.append(checkpoint(
                    attend, q_s[start:stop], fk_s, fv_s, use_reentrant=False
                ))
        out = self.to_out(torch.cat(outputs).reshape(-1, self.channels))
        return hr.replace(hr.feats + (self.gate * strength).to(out.dtype) * out)


class SparseLRDeterministicMultiRingAttention(nn.Module):
    """Bounded coordinate routing from each HR query to LR parent and rings."""
    def __init__(self, channels, num_heads, rope_freq=(1.0, 10000.0),
                 lr_resolution=32, hr_to_lr_scale=2,
                 ring_radii=(0, 1, 2, 4), ring_budgets=(1, 12, 12, 8),
                 chunk_size=1024, coord_offset=0.0, output_scale=1.0):
        super().__init__()
        if channels % num_heads:
            raise ValueError("channels must be divisible by num_heads")
        if len(ring_radii) != len(ring_budgets) or not ring_radii:
            raise ValueError("ring radii and budgets must be non-empty and equal length")
        if int(ring_radii[0]) != 0 or int(ring_budgets[0]) != 1:
            raise ValueError("first ring must be exact parent: radius=0, budget=1")
        self.channels, self.num_heads = int(channels), int(num_heads)
        self.head_dim = self.channels // self.num_heads
        self.lr_resolution = int(lr_resolution)
        self.hr_to_lr_scale = int(hr_to_lr_scale)
        self.chunk_size = int(chunk_size)
        self.coord_offset = float(coord_offset)
        self.output_scale = float(output_scale)
        if self.output_scale <= 0:
            raise ValueError('output_scale must be positive')
        offsets = []
        for radius, budget in zip(ring_radii, ring_budgets):
            radius, budget = int(radius), int(budget)
            candidates = [(x, y, z)
                for x in range(-radius, radius + 1)
                for y in range(-radius, radius + 1)
                for z in range(-radius, radius + 1)
                if max(abs(x), abs(y), abs(z)) == radius]
            if radius < 0 or budget <= 0:
                raise ValueError("radii must be non-negative and budgets positive")
            if budget < len(candidates):
                chosen = [0]
                while len(chosen) < budget:
                    best_i, best_d = None, -1
                    for i, point in enumerate(candidates):
                        if i in chosen:
                            continue
                        distance = min(sum(
                            (point[j] - candidates[k][j]) ** 2 for j in range(3)
                        ) for k in chosen)
                        if distance > best_d:
                            best_i, best_d = i, distance
                    chosen.append(best_i)
                candidates = [candidates[i] for i in chosen]
            offsets.extend(candidates[:budget])
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long),
                             persistent=False)
        self.relative_bias = nn.Parameter(
            torch.zeros(len(offsets), num_heads, dtype=torch.float32)
        )
        self.rope = PhysicalRotaryEmbedding(self.head_dim, rope_freq)
        self.to_out = nn.Linear(channels, channels)
        nn.init.zeros_(self.to_out.weight)
        if self.to_out.bias is not None:
            nn.init.zeros_(self.to_out.bias)
        self.gate = nn.Parameter(torch.ones((), dtype=torch.float32))

    def _hash(self, xyz):
        r = self.lr_resolution
        return xyz[..., 0] * r * r + xyz[..., 1] * r + xyz[..., 2]

    def forward(self, hr, lr, projection, strength):
        q_all = projection.to_q(_fp32_layer_norm(
            projection.norm_hr, hr.feats
        )).reshape(-1, self.num_heads, self.head_dim)
        kv = projection.to_kv(_fp32_layer_norm(
            projection.norm_lr, lr.feats
        )).reshape(-1, 2, self.num_heads, self.head_dim)
        k_all, v_all = kv.unbind(dim=1)
        q_all = projection.q_norm(hr.replace(q_all)).feats
        k_all = projection.k_norm(lr.replace(k_all)).feats
        q_all = self.rope(q_all, hr.coords[:, 1:].float())
        k_all = self.rope(k_all, lr.coords[:, 1:].float() * self.hr_to_lr_scale + self.coord_offset)
        outputs, scale = [], self.head_dim ** -0.5
        bias = self.relative_bias.transpose(0, 1).unsqueeze(0)
        for hs, ls in zip(hr.layout, lr.layout):
            if ls.stop == ls.start:
                outputs.append(torch.zeros_like(q_all[hs]))
                continue
            lr_hash, order = torch.sort(self._hash(lr.coords[ls, 1:].long()))
            parent = torch.div(hr.coords[hs, 1:].long(),
                               self.hr_to_lr_scale, rounding_mode="floor")
            for start in range(0, parent.shape[0], self.chunk_size):
                stop = min(start + self.chunk_size, parent.shape[0])
                target = parent[start:stop, None, :] + self.offsets[None, :, :]
                in_bounds = ((target >= 0) & (target < self.lr_resolution)).all(-1)
                target_hash = self._hash(target)
                pos = torch.searchsorted(lr_hash, target_hash)
                safe = pos.clamp(max=lr_hash.numel() - 1)
                valid = in_bounds & (pos < lr_hash.numel())
                valid &= lr_hash[safe] == target_hash
                index = order[safe] + ls.start
                q = q_all[hs][start:stop]

                def attend(q_chunk, k_source, v_source, rel_bias,
                           index=index, valid=valid, scale=scale):
                    k, v = k_source[index], v_source[index]
                    scores = torch.einsum("bhd,bkhd->bhk", q_chunk, k) * scale
                    scores = (scores + rel_bias.to(scores.dtype)).masked_fill(
                        ~valid[:, None, :], -torch.inf
                    )
                    has_neighbor = valid.any(dim=-1)
                    scores[~has_neighbor] = 0
                    weight = torch.softmax(scores.float(), dim=-1).to(v.dtype)
                    weight = weight * valid[:, None, :].to(weight.dtype)
                    return torch.einsum("bhk,bkhd->bhd", weight, v)

                outputs.append(checkpoint(
                    attend, q, k_all, v_all, bias, use_reentrant=False
                ))
        out = self.to_out(torch.cat(outputs).reshape(-1, self.channels))
        return hr.replace(hr.feats + (self.gate * strength * self.output_scale).to(out.dtype) * out)


class SparseLRParentAttention(nn.Module):
    """Each HR token reads the coordinate-aligned 3x3x3 LR neighborhood."""
    def __init__(self, channels, num_heads, lr_resolution=32, hr_to_lr_scale=2, chunk_size=2048):
        super().__init__()
        if channels % num_heads:
            raise ValueError("channels must be divisible by num_heads")
        self.channels, self.num_heads = channels, num_heads
        self.head_dim = channels // num_heads
        self.lr_resolution, self.hr_to_lr_scale = int(lr_resolution), int(hr_to_lr_scale)
        self.chunk_size = int(chunk_size)
        self.relative_bias = nn.Parameter(torch.zeros(27, num_heads, dtype=torch.float32))
        self.gate = nn.Parameter(torch.zeros((), dtype=torch.float32))
        offsets = torch.stack(torch.meshgrid(
            torch.arange(-1, 2), torch.arange(-1, 2), torch.arange(-1, 2), indexing="ij"
        ), dim=-1).reshape(-1, 3)
        self.register_buffer("offsets", offsets.long(), persistent=False)

    def _hash(self, xyz):
        r = self.lr_resolution
        return xyz[..., 0] * r * r + xyz[..., 1] * r + xyz[..., 2]

    def forward(self, hr, lr, projection, strength):
        """Reuse the corresponding global branch's Q/K/V/out projections."""
        q_all = projection.to_q(_fp32_layer_norm(projection.norm_hr, hr.feats)).reshape(-1, self.num_heads, self.head_dim)
        kv = projection.to_kv(_fp32_layer_norm(projection.norm_lr, lr.feats)).reshape(-1, 2, self.num_heads, self.head_dim)
        k_all, v_all = kv.unbind(dim=1)
        q_all = projection.q_norm(hr.replace(q_all)).feats
        k_all = projection.k_norm(lr.replace(k_all)).feats
        outputs, scale = [], self.head_dim ** -0.5
        bias = self.relative_bias.transpose(0, 1).unsqueeze(0)
        for hs, ls in zip(hr.layout, lr.layout):
            lr_hash, order = torch.sort(self._hash(lr.coords[ls, 1:].long()))
            parent = torch.div(hr.coords[hs, 1:].long(), self.hr_to_lr_scale, rounding_mode="floor")
            for start in range(0, parent.shape[0], self.chunk_size):
                stop = min(start + self.chunk_size, parent.shape[0])
                target = parent[start:stop, None, :] + self.offsets[None, :, :]
                in_bounds = ((target >= 0) & (target < self.lr_resolution)).all(dim=-1)
                target_hash = self._hash(target)
                pos = torch.searchsorted(lr_hash, target_hash)
                safe = pos.clamp(max=max(0, lr_hash.numel() - 1))
                valid = in_bounds & (pos < lr_hash.numel())
                if lr_hash.numel():
                    valid &= lr_hash[safe] == target_hash
                    index = order[safe] + ls.start
                else:
                    index = torch.zeros_like(safe)
                q = q_all[hs][start:stop]

                # Checkpoint the gather and attention. Materializing 27 full-width
                # K/V vectors per HR token is otherwise retained at every injected
                # block and defeats the memory goal of the local branch.
                def attend(
                    q_chunk, k_source, v_source, relative_bias,
                    index=index, valid=valid, scale=scale,
                ):
                    k, v = k_source[index], v_source[index]
                    scores = torch.einsum("bhd,bkhd->bhk", q_chunk, k) * scale
                    scores = (scores + relative_bias.to(scores.dtype)).masked_fill(
                        ~valid[:, None, :], -torch.inf
                    )
                    has_neighbor = valid.any(dim=-1)
                    scores[~has_neighbor] = 0
                    weight = torch.softmax(scores.float(), dim=-1).to(v.dtype)
                    weight = weight * valid[:, None, :].to(weight.dtype)
                    return torch.einsum("bhk,bkhd->bhd", weight, v)

                outputs.append(checkpoint(
                    attend, q, k_all, v_all, bias,
                    use_reentrant=False,
                ))
        out = projection.to_out(torch.cat(outputs).reshape(-1, self.channels))
        return hr.replace(hr.feats + (self.gate * strength).to(out.dtype) * out)


__all__ = [
    "smoothstep_schedule", "pool_sparse_lr_memory", "LRHRLatentAligner", "SparseLRGlobalAttention",
    "SparseLRContextBlock", "SparseLRContextEncoder", "SparseLRSynchronizedGlobalEncoder", "SparseLRPyramidEncoder",
    "SparseLRSelectedAttention", "SparseLRDeterministicMultiRingAttention", "SparseLRParentAttention",
]
