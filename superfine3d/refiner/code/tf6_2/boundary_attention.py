"""Same 128-token ring budget; prioritize active cross-cell surface neighbours.

Routing uses integer native SLat coordinates only, never features/GT/mesh cracks.
No new tokens, coordinate movement, pooling, RoPE, or attention kernel changes.
"""
from collections import defaultdict
from itertools import product
import torch
from tf6.near_attention import NearPlan, _morton


OFFSETS = tuple(o for o in product((-1, 0, 1), repeat=3) if o != (0, 0, 0))


def _spread(groups, limit, rows, edge):
    """Allocate round-robin across nonempty spatial buckets, then Morton midpoints."""
    sources = [sorted(groups[k], key=lambda i: (_morton(rows[i][1:], edge), tuple(rows[i])))
               for k in sorted(groups) if groups[k]]
    allocation = [0] * len(sources)
    left = min(limit, sum(map(len, sources)))
    while left:
        for i, source in enumerate(sources):
            if left and allocation[i] < len(source):
                allocation[i] += 1
                left -= 1
    return [source[((2*k+1)*len(source))//(2*n)]
            for source, n in zip(sources, allocation) for k in range(n)]


def build_boundary_plan(coords, prep, cell_size=16, budget=128, chunk_cells=64,
                        boundary_budget=96):
    if cell_size < 2 or cell_size % 2 or not 0 <= boundary_budget <= budget or chunk_cells < 1:
        raise ValueError('Require even cell size >=2, 0<=boundary_budget<=budget, positive chunks')
    rows = coords.detach().cpu().tolist()
    if not rows or any(len(r) != 4 for r in rows):
        raise ValueError('Nonempty [batch,x,y,z] sparse support required')
    positions = {tuple(r): i for i, r in enumerate(rows)}
    if len(positions) != len(rows):
        raise ValueError('Duplicate support coordinates')
    cu, perm = prep.cu_seqlens.cpu().tolist(), prep.perm.cpu().tolist()
    cells, members = [], []
    for a, b in zip(cu[:-1], cu[1:]):
        own = perm[a:b]
        if not own:
            raise ValueError('Empty packed cell')
        r = rows[own[0]]
        cell = (r[0], *(v//cell_size for v in r[1:]))
        if any((rows[i][0], *(v//cell_size for v in rows[i][1:])) != cell for i in own):
            raise ValueError('Fixed grid and boundary MSGA partition disagree')
        cells.append(cell); members.append(own)
    lookup = {c: i for i, c in enumerate(cells)}
    neighbours, keys, face_counts, seam_counts, candidate_counts = [], [], [], [], []
    for ci, cell in enumerate(cells):
        batch, cx, cy, cz = cell
        ids = [lookup[(batch, cx+dx, cy+dy, cz+dz)] for dx, dy, dz in OFFSETS
               if (batch, cx+dx, cy+dy, cz+dz) in lookup]
        # Each candidate receives one deterministic bucket. Face adjacency takes
        # priority over edge/corner adjacency. Patches are 2x2 on each face.
        candidates = {}
        for i in members[ci]:
            _, x, y, z = rows[i]
            local = (x % cell_size, y % cell_size, z % cell_size)
            if all(0 < v < cell_size-1 for v in local):
                continue
            for dx, dy, dz in OFFSETS:
                other_cell = ((x+dx)//cell_size, (y+dy)//cell_size, (z+dz)//cell_size)
                if other_cell == (cx, cy, cz):
                    continue
                j = positions.get((batch, x+dx, y+dy, z+dz))
                if j is None:
                    continue
                priority = 0 if abs(dx)+abs(dy)+abs(dz) == 1 else 1
                direction = tuple(a-b for a, b in zip(other_cell, (cx, cy, cz)))
                patch = tuple(v//(cell_size//2) for v in local)
                label = (priority, direction, patch)
                if j not in candidates or label < candidates[j]:
                    candidates[j] = label
        selected = []
        for priority in (0, 1):
            groups = defaultdict(list)
            for j, label in candidates.items():
                if label[0] == priority:
                    groups[label[1:]].append(j)
            selected += _spread(groups, boundary_budget-len(selected), rows, cell_size)
        chosen = set(selected)
        # The remaining >=32 slots use the old neighbour-cell representative
        # policy, excluding selected IDs. Missing seam capacity is returned here.
        ordinary = {cells[j]: [k for k in members[j] if k not in chosen] for j in ids}
        selected += _spread(ordinary, budget-len(selected), rows, cell_size)
        if len(selected) != len(set(selected)):
            raise AssertionError('Duplicate neighbour K/V')
        neighbours.append(tuple(selected))
        keys.append(members[ci]+selected)
        face_counts.append(sum(j in candidates and candidates[j][0] == 0 for j in selected))
        seam_counts.append(sum(j in candidates for j in selected))
        candidate_counts.append(len(candidates))
    chunks = []
    for first in range(0, len(cells), chunk_cells):
        last = min(first+chunk_cells, len(cells))
        lengths = [len(v) for v in keys[first:last]]
        idx = torch.tensor([i for seq in keys[first:last] for i in seq], device=coords.device, dtype=torch.long)
        ck = torch.tensor([0]+lengths, device=coords.device, dtype=torch.int32).cumsum(0).int()
        cq = (prep.cu_seqlens[first:last+1]-cu[first]).contiguous()
        chunks.append((cu[first], cu[last], idx, cq, ck,
                       max(cu[j+1]-cu[j] for j in range(first, last)), max(lengths)))
    return NearPlan(tuple(chunks), tuple(neighbours), dict(
        routing='tf6_2_boundary_v1', cells=len(cells), tokens=len(rows), budget=budget,
        boundary_budget=boundary_budget, max_neighbour_tokens=max(map(len, neighbours)),
        mean_neighbour_tokens=sum(map(len, neighbours))/len(cells),
        mean_active_seam_candidates=sum(candidate_counts)/len(cells),
        mean_selected_face_neighbours=sum(face_counts)/len(cells),
        mean_selected_seam_neighbours=sum(seam_counts)/len(cells),
        empty_neighbour_cells=sum(not n for n in neighbours)))
