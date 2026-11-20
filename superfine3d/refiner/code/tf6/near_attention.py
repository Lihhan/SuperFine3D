"""Pointelligence MSGA topology, fixed-budget nearest ring specialization.

All own-cell tokens plus representatives from the 26 touching cells enter
ONE softmax. Unlike upstream own-cell-relative budgets, the approved TF6
ring budget is an absolute 128 tokens shared across neighbours. No shifts.
Already-rotated Q/K are consumed directly; there is no second RoPE.
The implementation uses the same FA varlen primitive as Pointelligence.
"""
from dataclasses import dataclass
from itertools import product
import torch
from torch.nn import functional as F


@dataclass
class NearPlan:
    chunks: tuple
    neighbour_indices: tuple
    stats: dict


def _morton(xyz, edge):
    value = 0
    for bit in range(max(1, (edge-1).bit_length())):
        for axis in range(3):
            value |= ((xyz[axis] % edge >> bit) & 1) << (3*bit+axis)
    return value


def build_near_plan(coords, prep, cell_size=16, budget=128, chunk_cells=64):
    if budget < 0 or cell_size < 1 or chunk_cells < 1:
        raise ValueError('Invalid near MSGA settings')
    rows = coords.detach().cpu().tolist()
    if len({tuple(r) for r in rows}) != len(rows):
        raise ValueError('Duplicate support coordinates')
    cu = prep.cu_seqlens.cpu().tolist()
    perm = prep.perm.cpu().tolist()
    cells = []
    members = []
    for a, b in zip(cu[:-1], cu[1:]):
        own = perm[a:b]
        r = rows[own[0]]
        cell = (r[0], *(v//cell_size for v in r[1:]))
        if any((rows[i][0], *(v//cell_size for v in rows[i][1:])) != cell for i in own):
            raise ValueError('Fixed grid and MSGA partition disagree')
        cells.append(cell)
        members.append(sorted(own, key=lambda i: _morton(rows[i][1:], cell_size)))
    lookup = {c:i for i,c in enumerate(cells)}
    offsets = [o for o in product((-1,0,1), repeat=3) if o != (0,0,0)]
    # Fixed direction order. Every occupied neighbour receives one slot before
    # any receives a second; exhausted cells release their budget to others.
    neighbours, all_keys = [], []
    for ci,(batch,x,y,z) in enumerate(cells):
        ids = [lookup[(batch,x+dx,y+dy,z+dz)] for dx,dy,dz in offsets
               if (batch,x+dx,y+dy,z+dz) in lookup]
        allocation = [0]*len(ids)
        left = min(budget, sum(len(members[j]) for j in ids))
        while left:
            for pos,j in enumerate(ids):
                if left and allocation[pos] < len(members[j]):
                    allocation[pos] += 1
                    left -= 1
        selected = []
        for j,n in zip(ids,allocation):
            source = members[j]
            # Midpoints of equal intervals in Morton order: distinct, stable,
            # input-permutation invariant, spread through the entire cell.
            selected.extend(source[((2*k+1)*len(source))//(2*n)] for k in range(n))
        neighbours.append(tuple(selected))
        all_keys.append(perm[cu[ci]:cu[ci+1]]+selected)
    chunks = []
    device = coords.device
    for first in range(0,len(cells),chunk_cells):
        last = min(first+chunk_cells,len(cells))
        lengths = [len(v) for v in all_keys[first:last]]
        idx = torch.tensor([i for seq in all_keys[first:last] for i in seq],device=device,dtype=torch.long)
        ck = torch.tensor([0]+lengths,device=device,dtype=torch.int32).cumsum(0).int()
        cq = (prep.cu_seqlens[first:last+1]-cu[first]).contiguous()
        chunks.append((cu[first],cu[last],idx,cq,ck,max(cu[j+1]-cu[j] for j in range(first,last)),max(lengths)))
    return NearPlan(tuple(chunks),tuple(neighbours),dict(cells=len(cells),tokens=len(rows),
        budget=budget,max_neighbour_tokens=max(map(len,neighbours)),
        mean_neighbour_tokens=sum(map(len,neighbours))/len(cells),
        empty_neighbour_cells=sum(not n for n in neighbours)))


def near_attention(q,k,v,prep,plan,force_reference=False):
    """Native order in/out; gathers retain autograd for repeated neighbour K/V."""
    sorted_q = q[prep.perm]
    outputs = []
    for a,b,idx,cq,ck,mq,mk in plan.chunks:
        qq,kk,vv = sorted_q[a:b], k[idx], v[idx]
        if q.is_cuda and q.dtype in (torch.float16,torch.bfloat16) and not force_reference:
            from flash_attn import flash_attn_varlen_func
            y = flash_attn_varlen_func(qq,kk,vv,cq,ck,mq,mk,dropout_p=0.,causal=False)
        else:
            ys = []
            for qa,qb,ka,kb in zip(cq[:-1].tolist(),cq[1:].tolist(),ck[:-1].tolist(),ck[1:].tolist()):
                ys.append(F.scaled_dot_product_attention(qq[qa:qb].transpose(0,1)[None],
                    kk[ka:kb].transpose(0,1)[None],vv[ka:kb].transpose(0,1)[None])[0].transpose(0,1))
            y = torch.cat(ys)
        outputs.append(y)
    return torch.cat(outputs)[prep.inv_perm]
