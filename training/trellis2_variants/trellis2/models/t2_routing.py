"""Spatial counterpart of Ultra's window-first, bounded LR attention."""
import torch
from .dual_resolution_flow import attend


def partition_indices(coords, resolution, edge):
    if resolution < 1 or edge < 1:
        raise ValueError('Positive grid and partition sizes required')
    if (coords[:, 1:] < 0).any() or (coords[:, 1:] >= resolution).any():
        raise ValueError('Out-of-bounds native LR coordinates')
    keys = torch.cat((coords[:, :1].long(), coords[:, 1:].long() // edge), 1)
    unique, inverse = torch.unique(keys, dim=0, sorted=True, return_inverse=True)
    order = torch.argsort(inverse, stable=True)
    return {tuple(key): idx for key, idx in zip(unique.tolist(), order.split(torch.bincount(inverse).tolist()))}


def spatial_routes(hr_coords, lr_coords, routes, hr_resolution, lr_resolution, edge, partition_blocks=None):
    """Split HR queries at LR partition boundaries; retain their MSGA HR K/V.

    A query belongs to the partition containing its physical LR parent.
    Empty LR partitions supply zero LR tokens, without cross-partition fallback.
    """
    if min(hr_resolution, lr_resolution, edge) < 1:
        raise ValueError('Positive resolutions/edge required')
    if (hr_coords[:, 1:] < 0).any() or (hr_coords[:, 1:] >= hr_resolution).any():
        raise ValueError('Out-of-bounds HR coordinates')
    blocks = partition_indices(lr_coords, lr_resolution, edge) if partition_blocks is None else partition_blocks
    parent = torch.floor((hr_coords[:, 1:].double() + .5) * lr_resolution / hr_resolution).long()
    parts = parent // edge
    empty = torch.empty(0, dtype=torch.long, device=hr_coords.device)
    result = []
    for batch, qidx, kidx in routes:
        unique, inverse = torch.unique(parts[qidx], dim=0, sorted=True, return_inverse=True)
        order = torch.argsort(inverse, stable=True)
        for key, sub in zip(unique.tolist(), order.split(torch.bincount(inverse).tolist())):
            result.append((batch, qidx[sub], kidx, blocks.get((batch, *key), empty)))
    return result


def routed_joint_attention(q, k, v, lk, lv, routes, chunk_cells=2, checkpoint_chunks=True):
    """Reuse the tested joint softmax, treating each spatial LR subset as a layout."""
    from .m2_attention import joint_attention
    # Chunking bounds copies. No N_HR x N_HR mask or full LR gather per HR cell.
    outputs, indices = [], []
    from torch.utils.checkpoint import checkpoint
    for start in range(0, len(routes), chunk_cells):
        group = tuple(routes[start:start + chunk_cells])
        def run(q_, k_, v_, lk_, lv_, group=group):
            qs, ks, vs, qr, layouts = [], [], [], [], []
            qoff = koff = loff = 0
            lks, lvs = [], []
            for j, (_, qi, ki, li) in enumerate(group):
                qs.append(q_[qi]); ks.append(k_[ki]); vs.append(v_[ki])
                lks.append(lk_[li]); lvs.append(lv_[li])
                qr.append((j, torch.arange(qoff, qoff+len(qi), device=q_.device),
                           torch.arange(koff, koff+len(ki), device=q_.device)))
                layouts.append(slice(loff, loff+len(li)))
                qoff += len(qi); koff += len(ki); loff += len(li)
            return joint_attention(torch.cat(qs), torch.cat(ks), torch.cat(vs),
                                   torch.cat(lks), torch.cat(lvs), qr, layouts,
                                   chunk_cells=len(group), checkpoint_chunks=False)
        tensors = (q, k, v, lk, lv)
        out = checkpoint(run, *tensors, use_reentrant=False) if (
            checkpoint_chunks and torch.is_grad_enabled() and any(t.requires_grad for t in tensors)
        ) else run(*tensors)
        outputs.append(out); indices.extend(r[1] for r in group)
    return torch.cat(outputs)[torch.argsort(torch.cat(indices))]


def partition_attention(q, k, v, coords, resolution, edge):
    blocks = partition_indices(coords, resolution, edge)
    ids = list(blocks.values())
    return torch.cat([attend(q[idx], k[idx], v[idx]) for idx in ids])[torch.argsort(torch.cat(ids))]
