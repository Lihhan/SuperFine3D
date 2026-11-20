"""Deterministic fixed spatial partition; no rings, halo, shifts or truncation."""
import torch
from torch.utils.checkpoint import checkpoint
from pointelligence_grid.grid_sdpa import GridAttentionPrep, grid_sdpa


def build_fixed_grid(coords, cell_size=8):
    if coords.ndim != 2 or coords.shape[1] != 4 or cell_size < 1:
        raise ValueError('Expected [N, batch+xyz] coordinates and positive cell size')
    if not len(coords):
        raise ValueError('Empty HR support')
    labels = torch.cat((coords[:, :1].long(),
                        torch.div(coords[:, 1:].long(), cell_size, rounding_mode='floor')), 1)
    cells, inverse, counts = torch.unique(labels, dim=0, sorted=True,
                                          return_inverse=True, return_counts=True)
    perm = torch.argsort(inverse, stable=True)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(len(perm), device=perm.device)
    cu = torch.cat((counts.new_zeros(1), counts.cumsum(0))).to(torch.int32)
    return GridAttentionPrep(perm, inv, cu, int(counts.max().item()),
                             inverse, cells[:, 1:].int(), cells[:, 0])


def cell_attention(q, k, v, coords, prep):
    # T1's Q/K already contain its physical-coordinate RoPE. Keep that exact
    # convention and bypass Pointelligence's second rotation, not its kernel.
    return grid_sdpa(q, k, v, coords[:, 1:].float(), [prep],
                     q.shape[-1] ** -0.5, pre_rotated=True,
                     enable_telemetry=False)


def build_joint_plan(prep, lr_layout, nhr, max_cells=64, max_kv_tokens=131072):
    """Index-only extension of the same grid segments with read-only LR K/V."""
    cu = prep.cu_seqlens.tolist()
    batches = prep.cell_batch.tolist()
    plans=[]; start=0
    while start < len(batches):
        end=start; total=0; klens=[]
        while end < len(batches) and end-start < max_cells:
            sl=lr_layout[batches[end]]
            length=cu[end+1]-cu[end]+sl.stop-sl.start
            if end > start and total+length > max_kv_tokens: break
            total+=length; klens.append(length); end+=1
        counts=torch.tensor(klens,device=prep.perm.device,dtype=torch.long)
        offsets=torch.cat((counts.new_zeros(1),counts.cumsum(0)))
        cell=torch.repeat_interleave(torch.arange(end-start,device=counts.device), counts, output_size=total)
        pos=torch.arange(total,device=counts.device)-offsets[cell]
        own_starts=prep.cu_seqlens[start:end].long()
        own_counts=(prep.cu_seqlens[start+1:end+1]-prep.cu_seqlens[start:end]).long()
        lr_starts=torch.tensor([lr_layout[batches[i]].start for i in range(start,end)],device=counts.device)
        index=torch.where(pos < own_counts[cell], own_starts[cell]+pos,
                          nhr+lr_starts[cell]+pos-own_counts[cell])
        qcu=(prep.cu_seqlens[start:end+1]-cu[start]).contiguous()
        plans.append((cu[start],cu[end],index,qcu,offsets.int(),
                      max(cu[i+1]-cu[i] for i in range(start,end)),max(klens)))
        start=end
    return tuple(plans)


def grid_joint_attention(q,k,v,lk,lv,prep,plans,checkpoint_chunks=True):
    """One softmax over own HR cell + LR; other HR cells never enter K/V.

    Uses FlashAttention's unequal-Q/K varlen entry because the QKV-packed
    entry in grid_sdpa requires equal lengths. The cell partition is identical.
    """
    from flash_attn import flash_attn_varlen_func
    qs=q[prep.perm]
    keys=torch.cat((k[prep.perm],lk),0)
    values=torch.cat((v[prep.perm],lv),0)
    output=[]
    for a,b,idx,cq,ck,mq,mk in plans:
        def run(q_,k_,v_,a=a,b=b,idx=idx,cq=cq,ck=ck,mq=mq,mk=mk):
            return flash_attn_varlen_func(q_[a:b],k_[idx],v_[idx],cq,ck,mq,mk,
                                          dropout_p=0.0,causal=False)
        if checkpoint_chunks and torch.is_grad_enabled():
            output.append(checkpoint(run,qs,keys,values,use_reentrant=False))
        else: output.append(run(qs,keys,values))
    return torch.cat(output,0)[prep.inv_perm]
