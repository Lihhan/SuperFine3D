"""Exact joint sparse attention and resolution-aware coordinate utilities."""
import itertools
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def map_voxel_centers(xyz, source_resolution, target_resolution):
    if source_resolution <= 0 or target_resolution <= 0:
        raise ValueError('Grid resolutions must be positive')
    return (xyz.float() + 0.5) * (target_resolution / source_resolution) - 0.5


def unique_stable(indices):
    # A duplicate K/V would change its softmax probability. Keep first occurrence.
    unique, inverse = torch.unique(indices, sorted=True, return_inverse=True)
    first = torch.full_like(unique, len(indices))
    first.scatter_reduce_(0, inverse, torch.arange(len(indices), device=indices.device), reduce='amin')
    return indices[first.sort().values]


def joint_attention(q, k, v, lr_k, lr_v, routes, lr_layout,
                    chunk_cells=2, checkpoint_chunks=True):
    """Single softmax over MSGA + LR; no full HR matrix or global KV expansion.

    At most chunk_cells LR copies are materialized at a time. During training,
    chunk checkpointing avoids retaining every gathered KV until backward.
    LSE-based fusion of independent FlashAttention outputs is deliberately not
    used: the returned LSE can be non-differentiable, yielding incorrect gradients.
    """
    if chunk_cells < 1 or not routes:
        raise ValueError('Non-empty routes and positive chunk_cells required')
    outputs, permutation = [], []
    for start in range(0, len(routes), chunk_cells):
        group = tuple(routes[start:start + chunk_cells])
        permutation.extend(route[1] for route in group)

        def run(q_, k_, v_, lk_, lv_, group=group):
            queries, keys, values, q_lengths, k_lengths = [], [], [], [], []
            for batch, qidx, kidx in group:
                kk, vv = k_[kidx], v_[kidx]
                if lk_ is not None:
                    sl = lr_layout[batch]
                    kk = torch.cat((kk, lk_[sl]), dim=0)
                    vv = torch.cat((vv, lv_[sl]), dim=0)
                queries.append(q_[qidx]); keys.append(kk); values.append(vv)
                q_lengths.append(len(qidx)); k_lengths.append(len(kk))
            if q_.is_cuda and q_.dtype in (torch.float16, torch.bfloat16):
                from flash_attn import flash_attn_varlen_func
                cuq = torch.tensor([0] + list(itertools.accumulate(q_lengths)), dtype=torch.int32, device=q_.device)
                cuk = torch.tensor([0] + list(itertools.accumulate(k_lengths)), dtype=torch.int32, device=q_.device)
                return flash_attn_varlen_func(torch.cat(queries), torch.cat(keys), torch.cat(values),
                                             cuq, cuk, max(q_lengths), max(k_lengths), dropout_p=0.0, causal=False)
            return torch.cat([
                F.scaled_dot_product_attention(a.transpose(0, 1)[None], b.transpose(0, 1)[None],
                                               c.transpose(0, 1)[None], dropout_p=0.0)[0].transpose(0, 1)
                for a, b, c in zip(queries, keys, values)
            ])

        if checkpoint_chunks and torch.is_grad_enabled() and any(
                x is not None and x.requires_grad for x in (q, k, v, lr_k, lr_v)):
            outputs.append(checkpoint(run, q, k, v, lr_k, lr_v, use_reentrant=False))
        else:
            outputs.append(run(q, k, v, lr_k, lr_v))
    indices = torch.cat(permutation)
    return torch.cat(outputs)[torch.argsort(indices)]
