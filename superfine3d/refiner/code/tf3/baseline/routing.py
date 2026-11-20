"""Spatial LR contexts, with all active tokens retained and batch isolation."""
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


def context_plan(hr_coords, lr_coords, prep, hr_resolution, lr_resolution,
                 context_edge, cell_edge, max_cells=64, max_kv_tokens=131072):
    if hr_resolution % lr_resolution:
        raise ValueError('TF2 requires an integer scale ratio')
    ratio = hr_resolution // lr_resolution
    if ratio < 1 or context_edge < 1 or ratio * context_edge % cell_edge:
        raise ValueError('HR cells must fit exactly inside mapped LR contexts')
    for coords, resolution in ((hr_coords, hr_resolution), (lr_coords, lr_resolution)):
        if coords.ndim != 2 or coords.shape[1] != 4:
            raise ValueError('Expected batch+xyz coordinates')
        if ((coords[:, 1:] < 0) | (coords[:, 1:] >= resolution)).any():
            raise ValueError('Coordinates outside configured grid')
    # CPU index construction once per support; never per layer or sampling step.
    buckets = {}
    for i, row in enumerate(lr_coords.detach().cpu().tolist()):
        b, x, y, z = row
        buckets.setdefault((b, x//context_edge, y//context_edge, z//context_edge), []).append(i)
    cu = prep.cu_seqlens.cpu().tolist()
    # Derive labels from actual first coordinate: do not depend on vendor field names.
    first = hr_coords[prep.perm[prep.cu_seqlens[:-1].long()]].cpu().tolist()
    width = ratio * context_edge
    keys, empty = [], 0
    for i, (b, x, y, z) in enumerate(first):
        lr = buckets.get((b, x//width, y//width, z//width), [])
        empty += not bool(lr)
        keys.append(list(range(cu[i], cu[i+1])) + [len(hr_coords)+j for j in lr])
    plans = []
    start = 0
    while start < len(keys):
        end, count = start, 0
        while end < len(keys) and end-start < max_cells:
            n = len(keys[end])
            if end > start and count+n > max_kv_tokens:
                break
            count += n
            end += 1
        lengths = [len(k) for k in keys[start:end]]
        dev = hr_coords.device
        ck = torch.tensor([0] + list(__import__('itertools').accumulate(lengths)), device=dev, dtype=torch.int32)
        idx = torch.tensor([j for k in keys[start:end] for j in k], device=dev, dtype=torch.long)
        cq = (prep.cu_seqlens[start:end+1]-cu[start]).contiguous()
        plans.append((cu[start], cu[end], idx, cq, ck,
                      max(cu[i+1]-cu[i] for i in range(start,end)), max(lengths)))
        start = end
    stats = {'cells': len(keys), 'empty_lr_context_cells': empty,
             'max_hr_tokens': max(cu[i+1]-cu[i] for i in range(len(keys))),
             'max_lr_context_tokens': max((len(k)-(cu[i+1]-cu[i]) for i,k in enumerate(keys)),default=0),
             'scale_ratio': ratio, 'context_edge_lr': context_edge}
    return tuple(plans), stats


def packed_joint(q,k,v,lk,lv,prep,plans,recompute=True):
    """Same joint softmax as TF1; only its permitted LR indices change."""
    qs = q[prep.perm]
    ks, vs = torch.cat((k[prep.perm],lk)), torch.cat((v[prep.perm],lv))
    outputs=[]
    for a,b,idx,cq,ck,mq,mk in plans:
        def run(q_,k_,v_,a=a,b=b,idx=idx,cq=cq,ck=ck,mq=mq,mk=mk):
            qq,kk,vv = q_[a:b],k_[idx],v_[idx]
            if qq.is_cuda and qq.dtype in (torch.float16,torch.bfloat16):
                from flash_attn import flash_attn_varlen_func
                return flash_attn_varlen_func(qq,kk,vv,cq,ck,mq,mk,dropout_p=0.,causal=False)
            result=[]
            for i in range(len(cq)-1):
                qi=qq[int(cq[i]):int(cq[i+1])].transpose(0,1)[None]
                ki=kk[int(ck[i]):int(ck[i+1])].transpose(0,1)[None]
                vi=vv[int(ck[i]):int(ck[i+1])].transpose(0,1)[None]
                result.append(F.scaled_dot_product_attention(qi,ki,vi)[0].transpose(0,1))
            return torch.cat(result)
        outputs.append(checkpoint(run,qs,ks,vs,use_reentrant=False)
                       if recompute and torch.is_grad_enabled() else run(qs,ks,vs))
    return torch.cat(outputs)[prep.inv_perm]
