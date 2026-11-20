"""Separate LR-only attention. HR K/V never enter this operation."""
import torch
from torch.nn import functional as F
from tf3.baseline.routing import context_plan


def lr_plan(hr, lr, prep, hr_edge, lr_edge, context_edge, cell_edge):
    joint, stats = context_plan(hr, lr, prep, hr_edge, lr_edge, context_edge, cell_edge)
    plans = []
    for a,b,idx,cq,ck,mq,mk in joint:
        q_indices, k_indices, q_lengths, k_lengths = [], [], [], []
        # Transfer metadata once per forward/support, never inside a DiT block.
        cq_cpu, ck_cpu, ids = cq.cpu().tolist(), ck.cpu().tolist(), idx.cpu().tolist()
        for j in range(len(cq_cpu)-1):
            keys = [v-len(hr) for v in ids[ck_cpu[j]:ck_cpu[j+1]] if v >= len(hr)]
            if not keys:
                continue
            queries = list(range(a+cq_cpu[j], a+cq_cpu[j+1]))
            q_indices.extend(queries); k_indices.extend(keys)
            q_lengths.append(len(queries)); k_lengths.append(len(keys))
        if not q_lengths:
            continue
        def tensor(v): return torch.tensor(v, device=hr.device, dtype=torch.long)
        def cumulative(v):
            return torch.cat((torch.zeros(1,device=hr.device,dtype=torch.int32), tensor(v).cumsum(0).int()))
        plans.append((tensor(q_indices),tensor(k_indices),cumulative(q_lengths),cumulative(k_lengths),max(q_lengths),max(k_lengths)))
    return tuple(plans), stats


def lr_attention(q, k, v, prep, plans):
    sorted_q = q[prep.perm]
    result = torch.zeros_like(sorted_q)
    for qi,ki,cq,ck,mq,mk in plans:
        qq,kk,vv = sorted_q[qi],k[ki],v[ki]
        if q.is_cuda and q.dtype in (torch.bfloat16,torch.float16):
            from flash_attn import flash_attn_varlen_func
            values=flash_attn_varlen_func(qq,kk,vv,cq,ck,mq,mk,dropout_p=0.,causal=False)
        else:
            values=[]
            for j in range(len(cq)-1):
                qa,qb,ka,kb=map(int,(cq[j],cq[j+1],ck[j],ck[j+1]))
                values.append(F.scaled_dot_product_attention(qq[qa:qb].transpose(0,1)[None],kk[ka:kb].transpose(0,1)[None],vv[ka:kb].transpose(0,1)[None])[0].transpose(0,1))
            values=torch.cat(values)
        result=result.index_copy(0,qi,values)
    return result[prep.inv_perm]
