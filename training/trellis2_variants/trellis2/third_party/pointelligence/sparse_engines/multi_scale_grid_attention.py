"""MultiScaleGridAttention — fused multi-scale grid attention.

Designed in ``autoresearch/threads/attn_extreme/3_plans/sparse_intergrid_operators.md``;
pre-registered ``0_expectations/cycle8_o1_fused_multiscale.md``;
results ``2_conclusions/cycle_8_o1_fused_multiscale.md``.

ONE joint softmax per query token over:

  K/V(query in cell c) = [ c's own tokens, full all-to-all ]      (finest scale = GridAttention)
                       ⊕ [ representative tokens of c's
                           multi-scale radius-octave neighbour cells ]  (coarser scales)

So the finest scale is *identical* to ``GridAttention`` (``grid_sdpa``,
intra-cell all-to-all, same absolute 3D-RoPE) — it is segment 0 of the
single fused softmax — and each coarser ring shell adds
representative-token edges. Every point is a query and is updated every
layer; only the *neighbour* keys/values are subsampled — to the per-ring
own-cell-relative budget ``B_o = round(m_o · N)`` (``msga_ring_mult``,
cyc-37 redesign) — never the query side or the own-cell side.

Operator cost law (cycle-5 NCU; confirmed real-scene cycle-8 §3.5):
the inter edges are ONE extra FA-varlen launch over all cells with a
SMALL FIXED per-query-cell K/V budget — never O(num_cells) launches,
never full-resolution neighbour gather (that was the dead windowed op,
cycle-3 G3, 8-19x slower).

This module is self-contained: it does NOT depend on the (refuted,
graduated-but-legacy) 2-path / 3-path NSA-style operators. The
go-forward PNT attention surface is {GridAttention (`grid_sdpa`),
MultiScaleGridAttention (this)} only.

The radius-octave cell graph is an OPERATOR INPUT (the network builds
it — its partition / shift choice is a network concern).
``build_cell_multiscale_idx`` is the canonical builder for
tests/benches.
"""
from __future__ import annotations

import os
import weakref

from typing import List, Literal, Optional, Sequence, Tuple

import torch
from torch.library import custom_op

from flash_attn import flash_attn_varlen_func as _fa_varlen_fn

from .grid_sdpa import GridAttentionPrep, grid_sdpa, permute_rows


_RING_STENCILS: dict = {}
_BALANCED_STENCIL_ORDERS: dict = {}

# Indexed-gather structure cache — the indexed (materialize=False) output of
# ``_build_fused_kv`` is a pure function of the multiscale ctx + prep + ring
# budget, NOT of the K/V values, so it is identical for every block of a
# stage. Without the cache each block rebuilds ~1.8N rows of index arithmetic
# AND pays the `.tolist()` device→host sync (a launch-pipeline stall ×depth).
# Keyed by ``id(nbr_cell)`` with a ``weakref.finalize`` evictor: the entry
# lives exactly as long as the stage's ctx tensor, and the finalizer runs
# during the tensor's deallocation — before its id can be reused. (A
# WeakKeyDictionary is NOT usable here: dict probing rich-compares keys and
# ``Tensor.__eq__`` is elementwise.) Entry: (key, gather_i32, fused_cu_i32,
# max_k) with key = (ring_mult tuple, num_octaves); a key change (same ctx,
# different budget) recomputes and overwrites.
_IDX_GATHER_CACHE: dict = {}


def _indexed_gather_cached(nbr_cell, nbr_octave, nbr_cu, ring_mult_t,
                           num_octaves, cu_seqlens, k_rot, v_s, num_cells,
                           device, ring_mult):
    """``_build_fused_kv(materialize=False)`` with the per-ctx structure cache."""
    ck = (tuple(float(m) for m in ring_mult), int(num_octaves))
    nid = id(nbr_cell)
    ent = _IDX_GATHER_CACHE.get(nid)
    if ent is not None and ent[0] == ck:
        return ent[1], ent[2], ent[3]
    gather, _, fused_cu, max_k, _ = _build_fused_kv(
        nbr_cell.to(torch.int64), nbr_octave.to(torch.int64), nbr_cu,
        ring_mult_t, num_octaves, cu_seqlens, k_rot, v_s,
        num_cells, device, materialize=False,
    )
    if ent is None:
        weakref.finalize(nbr_cell, _IDX_GATHER_CACHE.pop, nid, None)
    _IDX_GATHER_CACHE[nid] = (ck, gather, fused_cu, max_k)
    return gather, fused_cu, max_k


_COMPOSED_KV_IDX_CACHE: dict = {}


def _composed_kv_idx_cached(gather, perm, nbr_cell, ring_mult, num_octaves):
    """V-gather-on-load (#2): the fused index ``gather`` (rows of the SORTED
    k_rot/v_s buffers) composed through ``perm`` into rows of the ORIGINAL-order
    buffers — ``perm[gather]``, so the kernel reads unsorted k_rot + raw V via
    one shared kv_row_indices. Value-independent (``perm`` and ``gather`` are both
    structural, fixed for an ``nbr_cell``'s lifetime) ⇒ cached per ctx, keyed
    like ``_indexed_gather_cached`` (plus ``id(perm)`` for safety). int32 out
    (values < N < 2^31), matching the raw fused index the kernel expects."""
    ck = (tuple(float(m) for m in ring_mult), int(num_octaves), id(perm))
    nid = id(nbr_cell)
    ent = _COMPOSED_KV_IDX_CACHE.get(nid)
    if ent is not None and ent[0] == ck:
        return ent[1]
    composed = perm.index_select(0, gather.long()).to(torch.int32)
    if ent is None:
        weakref.finalize(nbr_cell, _COMPOSED_KV_IDX_CACHE.pop, nid, None)
    _COMPOSED_KV_IDX_CACHE[nid] = (ck, composed)
    return composed


def _ring_stencil(octave: int, device: torch.device) -> torch.Tensor:
    """Integer-offset shell stencil for ring ``octave``: all v in Z^3 with
    Chebyshev norm in (2**(o-1), 2**o] (o=0: the 26-cell unit shell), self
    excluded. Shells are disjoint and tile space. Cached per (octave, device)."""
    key = (int(octave), device)
    st = _RING_STENCILS.get(key)
    if st is None:
        hi = 2 ** octave
        lo = 0 if octave == 0 else 2 ** (octave - 1)
        ax = torch.arange(-hi, hi + 1)
        v = torch.stack(torch.meshgrid(ax, ax, ax, indexing="ij"),
                        dim=-1).reshape(-1, 3)
        cheb = v.abs().amax(dim=1)
        st = v[(cheb > lo) & (cheb <= hi)].to(device=device, dtype=torch.int32)
        _RING_STENCILS[key] = st
    return st


def _balanced_stencil_order(octave: int, device: torch.device) -> torch.Tensor:
    """Deterministic farthest-direction ordering for stable ring coverage."""
    key = (int(octave), device)
    cached = _BALANCED_STENCIL_ORDERS.get(key)
    if cached is not None:
        return cached
    stencil = _ring_stencil(octave, torch.device("cpu")).float()
    unit = stencil / stencil.norm(dim=1, keepdim=True).clamp_min(1e-8)
    # Start on +X, then greedily maximize angular distance from selected rays.
    start = int(torch.argmax(unit[:, 0]).item())
    selected = [start]
    min_dist = torch.full((unit.shape[0],), float("inf"))
    for _ in range(1, unit.shape[0]):
        last = unit[selected[-1]]
        distance = 1.0 - unit @ last
        min_dist = torch.minimum(min_dist, distance)
        min_dist[selected] = -1.0
        selected.append(int(torch.argmax(min_dist).item()))
    cached = torch.tensor(selected, device=device, dtype=torch.long)
    _BALANCED_STENCIL_ORDERS[key] = cached
    return cached


@torch.compiler.disable
def build_cell_multiscale_idx(
    prep: GridAttentionPrep,
    *,
    num_octaves: int,            # = number of rings (o_max + 1)
    samples_per_ring: int = 16,
    generator: Optional[torch.Generator] = None,
    deterministic_coverage: bool = False,
    full_ring0: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stencil-SAMPLED radius-octave cell-neighbour graph. Returns
    ``(nbr_cell, nbr_octave, nbr_cu)`` — topology only (the per-ring budget
    is applied downstream from ``msga_ring_mult``).

    Ring ``o`` of a query cell is the disjoint Chebyshev shell of grid
    offsets in ``(2**(o-1), 2**o]`` partition cells. Each query cell draws
    ``samples_per_ring`` offsets from that shell WITHOUT replacement and
    keeps the ones landing on an occupied same-batch cell. Rationale (user
    direction 2026-06-11, supersedes the deleted exact radius-search
    builder): ring tokens are budget-downsampled anyway, so exhaustively
    enumerating every shell cell was wasted precision — sampling feeds the
    Hamilton budget directly.

    Cost: a handful of kernels over [C·k] tensors — no float work, no
    radius search, no dedup (without-replacement + disjoint shells + one
    occupied cell per grid coord make duplicates impossible), and no sort
    at num_octaves==1 (edges are emitted cell-major, CSR comes from one
    bincount+cumsum). Edge order within a cell is the generator's draw
    order — deterministic given (prep, generator state) — and is the
    Hamilton tie-break order. ``generator=None`` draws from the global RNG
    (same convention as the grid-shift draw itself).
    """
    if prep is None or prep.cell_grid is None or prep.cell_batch is None:
        raise ValueError(
            "build_cell_multiscale_idx: prep must carry cell_grid + "
            "cell_batch (built by GridShiftBuilder)")
    device = prep.cell_grid.device
    num_cells = int(prep.cu_seqlens.numel() - 1)
    if num_octaves <= 0:
        empty = torch.empty(0, device=device, dtype=torch.int64)
        nbr_cu = torch.zeros(num_cells + 1, device=device, dtype=torch.int32)
        return empty, empty.clone(), nbr_cu
    if samples_per_ring <= 0:
        raise ValueError(
            f"build_cell_multiscale_idx: samples_per_ring must be > 0, "
            f"got {samples_per_ring}")

    cell_grid = prep.cell_grid.to(torch.int64)                  # [C,3]
    cell_batch = prep.cell_batch                                 # [C]

    # Occupied-cell lookup keys. One cell per (batch, grid coord) by
    # construction of the partition, so keys are unique; margin covers the
    # largest stencil reach.
    reach = 2 ** (num_octaves - 1)
    lo = cell_grid.amin(0) - reach
    span = cell_grid.amax(0) + reach + 1 - lo                    # [3]
    sx, sy = int(span[1] * span[2]), int(span[2])

    def key_of(ic, batch):
        rel = ic - lo
        return ((batch * int(span[0]) + rel[:, 0]) * sx
                + rel[:, 1] * sy + rel[:, 2])

    occ_sorted, occ_order = torch.sort(key_of(cell_grid, cell_batch))
    qcell_base = torch.arange(num_cells, device=device, dtype=torch.int64)

    n_parts, o_parts, cnt_parts = [], [], []
    for o in range(num_octaves):
        st = _ring_stencil(o, device).to(torch.int64)            # [S,3]
        S = st.shape[0]
        if deterministic_coverage:
            # Probe every stencil position before applying the fixed budget.
            # This guarantees complete occupied-neighbour coverage in ring 0
            # and avoids wasting far-ring slots on empty coordinates.
            order = _balanced_stencil_order(o, device)
            ordered_st = st.index_select(0, order)
            k = S if (full_ring0 and o == 0) else min(int(samples_per_ring), S)
            tgt = cell_grid.unsqueeze(1) + ordered_st.view(1, S, 3)
            lookup_width = S
        else:
            k = min(int(samples_per_ring), S)
            r = torch.rand(num_cells, S, device=device, generator=generator)
            pick = r.argsort(dim=1)[:, :k]
            tgt = cell_grid.unsqueeze(1) + st[pick]
            lookup_width = k
        tkey = key_of(tgt.reshape(-1, 3),
                      torch.repeat_interleave(
                          cell_batch, lookup_width,
                          output_size=num_cells * lookup_width,
                      ))
        pos = torch.searchsorted(occ_sorted, tkey)
        pos_c = pos.clamp(max=num_cells - 1)
        hit = occ_sorted.index_select(0, pos_c) == tkey          # [C*k]
        hitm = hit.view(num_cells, lookup_width)
        posm = pos_c.view(num_cells, lookup_width)
        if deterministic_coverage and k < lookup_width:
            priority = torch.arange(lookup_width, device=device).view(1, -1)
            score = torch.where(hitm, priority, priority + lookup_width)
            take = torch.topk(score, k=k, dim=1, largest=False, sorted=True).indices
            hitm = hitm.gather(1, take)
            posm = posm.gather(1, take)
        selected_pos = posm[hitm]
        n_parts.append(occ_order.index_select(0, selected_pos))
        o_parts.append(torch.full_like(n_parts[-1], o))
        cnt_parts.append(hitm.sum(dim=1))                        # [C]

    if num_octaves == 1:
        n_all, o_all, counts = n_parts[0], o_parts[0], cnt_parts[0]
    else:
        # Regroup the per-octave cell-major segments by query cell (stable ⇒
        # octave-ascending within each cell). [E] is small (≤ C·k·noct).
        qc = torch.cat([torch.repeat_interleave(qcell_base, c) for c in cnt_parts])
        order = torch.argsort(qc, stable=True)
        n_all = torch.cat(n_parts)[order]
        o_all = torch.cat(o_parts)[order]
        counts = torch.stack(cnt_parts).sum(0)

    nbr_cu = torch.zeros(num_cells + 1, device=device, dtype=torch.int32)
    nbr_cu[1:] = counts.cumsum(0).to(torch.int32)
    return n_all.contiguous(), o_all.contiguous(), nbr_cu


def num_active_rings(ring_mult: Sequence[float]) -> int:
    """``o_max + 1`` = number of topology rings to build from a multiplier
    list. ``o_max = max{ o : m_o > 0 }`` (trailing/internal zeros shrink
    the reach); an empty or all-zero list ⇒ ``0`` (pure GridAttention).
    See ``msga_ring_budget_redesign.md`` §2.4.
    """
    active = [o for o, m in enumerate(ring_mult) if float(m) > 0.0]
    return (max(active) + 1) if active else 0


_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

# ring_mult is a per-model constant, but the operator rebuilt its device
# tensor every call (12×/forward in per_layer mode) — a blocking pageable H2D
# copy each time. Cache per (values, device); read-only downstream
# (_ring_budget_entry_reps only indexes it).
_RING_MULT_T_CACHE: dict = {}


def _ring_mult_tensor(ring_mult: Sequence[float],
                      device: torch.device) -> torch.Tensor:
    key = (tuple(float(m) for m in ring_mult), device)
    t = _RING_MULT_T_CACHE.get(key)
    if t is None:
        t = torch.tensor(key[0], device=device, dtype=torch.float32)
        _RING_MULT_T_CACHE[key] = t
    return t


def _validate_msga_inputs(q, k, v, coords, prep, multiscale, ring_mult, prep_kv):
    """Fail-loud input-contract check for ``multi_scale_grid_attention``.

    Downstream architectures (goal-5 PNT, goal-6 ablations) call the operator
    directly, so malformed inputs must raise an actionable ValueError/TypeError
    here — NOT a cryptic CUDA/index error deep in the gather, and NOT a silent
    wrong result. Pure host-side shape/dtype/contract checks (no device sync
    beyond the cheap shape reads already needed); raises, never asserts.
    """
    # q/k/v: 3-D [N,H,E], identical shape, float dtype.
    for name, t in (("q", q), ("k", k), ("v", v)):
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"multi_scale_grid_attention: {name} must be a Tensor, got {type(t)}")
        if t.dim() != 3:
            raise ValueError(
                f"multi_scale_grid_attention: {name} must be [N,H,E] (3-D), got shape {tuple(t.shape)}")
    if not (q.shape == k.shape == v.shape):
        raise ValueError(
            f"multi_scale_grid_attention: q/k/v must share shape, got "
            f"q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}")
    if not (q.dtype == k.dtype == v.dtype):
        raise ValueError(
            f"multi_scale_grid_attention: q/k/v must share dtype, got "
            f"q={q.dtype} k={k.dtype} v={v.dtype}")
    if q.dtype not in _FLOAT_DTYPES:
        raise ValueError(
            f"multi_scale_grid_attention: q/k/v dtype must be fp16/bf16/fp32, got {q.dtype}")
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("multi_scale_grid_attention: q/k/v must be CUDA tensors")
    N = q.shape[0]
    # coords: [N,3], same device.
    if not isinstance(coords, torch.Tensor) or coords.dim() != 2 or coords.shape[1] != 3:
        raise ValueError(
            f"multi_scale_grid_attention: coords must be [N,3], got "
            f"{tuple(coords.shape) if isinstance(coords, torch.Tensor) else type(coords)}")
    if coords.shape[0] != N:
        raise ValueError(
            f"multi_scale_grid_attention: coords N={coords.shape[0]} != q N={N}")
    if coords.device != q.device:
        raise ValueError(
            f"multi_scale_grid_attention: coords on {coords.device} but q on {q.device}")
    # prep: must carry the cell partition. NOTE: only cheap host-side metadata
    # checks here (shape/dtype/device/None) — NO ``.item()``/``.tolist()`` on
    # device tensors, since this runs every forward × every trunk block and a
    # device→host sync would be real per-iter overhead. The deeper invariant
    # "cu_seqlens[-1] == N" (partition covers every token) is guaranteed by the
    # grid-shift builder and is NOT re-checked here to avoid that sync.
    if prep is None or getattr(prep, "cell_id", None) is None or prep.cu_seqlens is None:
        raise ValueError(
            "multi_scale_grid_attention: prep must be a GridAttentionPrep with "
            "cell_id + cu_seqlens (built by the grid-shift builder)")
    # ring_mult: a sequence of finite, non-negative floats. Empty or all-zero
    # is VALID — it is the documented degenerate case (num_active_rings == 0 ⇒
    # the operator falls through to pure GridAttention), so do NOT reject it.
    if ring_mult is None:
        raise ValueError(
            "multi_scale_grid_attention: ring_mult must be a sequence of floats "
            "(empty/all-zero ⇒ GridAttention fallback), got None")
    for o, m in enumerate(ring_mult):
        mf = float(m)
        if mf < 0.0 or mf != mf:  # negative or NaN
            raise ValueError(
                f"multi_scale_grid_attention: ring_mult[{o}]={m} must be finite and >= 0")
    # multiscale: a 3-tuple (nbr_cell, nbr_octave, nbr_cu) of CUDA int tensors.
    if multiscale is None or len(multiscale) != 3:
        raise ValueError(
            "multi_scale_grid_attention: multiscale must be "
            "(nbr_cell[E], nbr_octave[E], nbr_cu[num_cells+1]) from build_cell_multiscale_idx")
    nbr_cell, nbr_octave, nbr_cu = multiscale
    num_cells = int(prep.cu_seqlens.numel() - 1)
    if nbr_cu.numel() != num_cells + 1:
        raise ValueError(
            f"multi_scale_grid_attention: multiscale nbr_cu has {nbr_cu.numel()} entries, "
            f"expected num_cells+1={num_cells + 1} (multiscale built for a different partition?)")
    if nbr_cell.numel() != nbr_octave.numel():
        raise ValueError(
            f"multi_scale_grid_attention: nbr_cell ({nbr_cell.numel()}) and nbr_octave "
            f"({nbr_octave.numel()}) must have equal length (one octave per edge)")
    # prep_kv (cyc-28 corruption probe): shared cell boundaries so FLOPs match.
    if prep_kv is not None and prep_kv is not prep:
        if prep_kv.cu_seqlens.shape != prep.cu_seqlens.shape:
            raise ValueError(
                "multi_scale_grid_attention: prep_kv.cu_seqlens must match prep.cu_seqlens "
                "shape (shared cell boundaries — prep_kv only reshuffles within cells)")


def _largest_remainder_plus(
    frac: torch.Tensor,         # [E] float — fractional parts (0..1)
    group_key: torch.Tensor,    # [E] int64 — (query cell, ring) group id
    remainder: torch.Tensor,    # [num_groups] int64 — +1s to hand out per group
    E: int,
    device: torch.device,
) -> torch.Tensor:              # [E] int64 (0/1)
    """Hamilton (largest-remainder) tie-break: +1 to the top-``remainder[g]``
    edges by fractional part within each group (ties → lower edge index)."""
    if E == 0:
        return torch.zeros(0, device=device, dtype=torch.int64)
    # frac descending (stable ⇒ ties keep ascending edge order), then group-major.
    idx1 = torch.argsort(frac, descending=True, stable=True)
    idx2 = torch.argsort(group_key[idx1], stable=True)
    order = idx1[idx2]                                       # group-major, frac-desc within group
    g_ord = group_key[order]                                 # [E] non-decreasing group ids
    pos = torch.arange(E, device=device, dtype=torch.int64)
    is_first = torch.ones(E, device=device, dtype=torch.bool)
    is_first[1:] = g_ord[1:] != g_ord[:-1]
    group_start = torch.cummax(
        torch.where(is_first, pos, torch.full_like(pos, -1)), dim=0).values
    rank = pos - group_start                                 # 0-based within-group rank
    keep_sorted = (rank < remainder[g_ord]).to(torch.int64)
    plus = torch.zeros(E, device=device, dtype=torch.int64)
    plus[order] = keep_sorted
    return plus


def _ring_budget_entry_reps(
    nbr_cell: torch.Tensor,     # [E] int64 — neighbour cell per edge
    nbr_octave: torch.Tensor,   # [E] int64 — ring index per edge
    edge_qc: torch.Tensor,      # [E] int64 — query cell per edge
    own_len: torch.Tensor,      # [num_cells] int64 — own-cell point counts (sorted)
    num_cells: int,
    num_octaves: int,
    device: torch.device,
    ring_mult_t: torch.Tensor,  # [num_octaves] float32 — per-ring multipliers
) -> torch.Tensor:              # [E] int64
    """MSGA ring-budget inter-token allocation (cyc-37 redesign).

    Ring ``o`` of query cell ``c`` (own size ``N_c``) gets a per-ring TOTAL
    budget ``B_{c,o} = round(m_o · N_c)`` (own-cell-relative ⇒
    scale-invariant), apportioned across the ring's neighbour cells
    **proportional to neighbour population** via **largest-remainder
    (Hamilton)** rounding, then capped per edge by availability (≤ the
    neighbour cell's size). Hamilton preserves the per-ring total exactly:
    independent per-edge ``round()`` would silently starve a ring whose
    neighbour-cell count exceeds ``B_{c,o}`` (the rings carrying the
    long-range context — spec §2.3/§4.3; cyc-37 review G-budget). A pure
    deterministic function of (graph, cell sizes, ring_mult); op and oracle
    compute it identically.
    """
    E = int(nbr_cell.numel())
    if E == 0:
        return torch.zeros(0, device=device, dtype=torch.int64)
    pop = own_len[nbr_cell]                                  # [E] int64 — nbr cell size = availability
    group_key = edge_qc * num_octaves + nbr_octave           # [E]
    num_groups = num_cells * num_octaves
    # Per-(query cell, ring) integer budget B_g = round(m_o · N_c).
    b_group = torch.zeros(num_groups, device=device, dtype=torch.float32)
    b_group[group_key] = torch.round(ring_mult_t[nbr_octave] * own_len[edge_qc].float())
    pop_sum = torch.zeros(num_groups, device=device, dtype=torch.int64)
    pop_sum.scatter_add_(0, group_key, pop)                  # exact (integer)
    # Real population-proportional allocation, then Hamilton apportionment.
    ideal = b_group[group_key] * pop.float() / pop_sum[group_key].float().clamp(min=1.0)
    floor_e = torch.floor(ideal)
    alloc_floor = torch.zeros(num_groups, device=device, dtype=torch.int64)
    alloc_floor.scatter_add_(0, group_key, floor_e.to(torch.int64))
    remainder = (b_group.to(torch.int64) - alloc_floor).clamp(min=0)  # [num_groups]
    plus = _largest_remainder_plus(ideal - floor_e, group_key, remainder, E, device)
    entry_reps = floor_e.to(torch.int64) + plus
    entry_reps = torch.minimum(entry_reps, pop).clamp(min=0)  # cap by availability
    return entry_reps


def _det_kv_bwd_enabled() -> bool:
    """Env gate for the deterministic fused-KV gather backward
    (``PNT_MSGA_DET_KV_BWD``). Default ON ("1"): it replaces autograd's
    nondeterministic atomic ``index_add_`` scatter with an ordered,
    atomic-free reduction — determinism is a strict improvement, and the
    values differ only by summation order, which the atomic path didn't
    even keep stable run-to-run. "0" restores the plain ``index_select``
    autograd path bit-exactly."""
    return os.environ.get("PNT_MSGA_DET_KV_BWD", "1") == "1"


def _det_gather_structure(
    src: torch.Tensor,          # [T] int64 — nbr-segment source rows (dups)
    dst: torch.Tensor,          # [T] int64 — their positions in the fused layout
    N: int,
    max_dup_bound: int,         # host upper bound on any row's occurrence count
    device: torch.device,
) -> tuple:
    """Transpose of the duplicated neighbour-segment gather, built ONCE at
    forward time for ``_GatherFusedKV``'s deterministic backward.

    Reorders the neighbour rows by **occurrence rank**: row order =
    (rank-within-equal-source-group, source row). Within one rank every
    source row appears AT MOST ONCE, so the backward can reduce rank-by-rank
    with unique-index ``index_add_`` calls (see ``_GatherFusedKV``). Returns
    ``(dst_by_rank, src_by_rank, rank_bounds)`` where ``rank_bounds`` is the
    HOST list of per-rank slice offsets (length ``bound + 1``; trailing
    empty rounds from a LOOSE ``max_dup_bound`` are skipped at zero cost,
    never wrong values).

    Device-side cost: two stable int32-key argsorts over [T] + O(T) index
    ops; ONE batched D2H copy (``rank_bounds`` — a single small ``tolist``,
    on top of ``_build_fused_kv``'s existing single sync).
    """
    T = int(src.numel())
    # int32 SORT KEY for radix speed (repo convention; values < N < 2^31).
    t_order = torch.argsort(src.to(torch.int32), stable=True)
    src_sorted = src.index_select(0, t_order)
    dst_sorted = dst.index_select(0, t_order)
    pos = torch.arange(T, device=device, dtype=torch.int64)
    is_start = torch.ones(T, device=device, dtype=torch.bool)
    is_start[1:] = src_sorted[1:] != src_sorted[:-1]
    # Occurrence rank within the equal-source group (integer cummax values
    # are exact ⇒ deterministic).
    seg_start = torch.cummax(
        torch.where(is_start, pos, torch.full_like(pos, -1)), dim=0).values
    rank = (pos - seg_start).to(torch.int32)
    r_order = torch.argsort(rank, stable=True)
    src_by_rank = src_sorted.index_select(0, r_order)
    dst_by_rank = dst_sorted.index_select(0, r_order)
    # Per-rank slice offsets. rank < max_occ <= bound (in-degree bound, see
    # _build_fused_kv), so a fixed-size histogram needs no device max().
    bound = max(1, min(int(max_dup_bound), T))
    counts = torch.zeros(bound, device=device, dtype=torch.int64)
    counts.scatter_add_(0, rank.to(torch.int64),
                        torch.ones(T, device=device, dtype=torch.int64))
    bounds_t = torch.zeros(bound + 1, device=device, dtype=torch.int64)
    bounds_t[1:] = counts.cumsum(0)
    rank_bounds = bounds_t.tolist()              # one batched D2H
    return dst_by_rank, src_by_rank, rank_bounds


@custom_op("pnt_msga::gather_fused_kv", mutates_args=())
def _gather_fused_kv(
    k_rot: torch.Tensor, v_s: torch.Tensor, gather: torch.Tensor,
    own_dst: Optional[torch.Tensor], dst_by_rank: Optional[torch.Tensor],
    src_by_rank: Optional[torch.Tensor], rank_bounds: Optional[List[int]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused-KV gather (``k/v.index_select(0, gather)``) with a DETERMINISTIC
    structure-exploiting backward (gate: ``PNT_MSGA_DET_KV_BWD``, default on).

    torch.library.custom_op (not autograd.Function) so it traces under
    torch.compile as a registered opaque leaf; register_autograd below carries
    the IDENTICAL deterministic backward (forward + backward compute are
    byte-for-byte unchanged — only the API wrapper differs).

    autograd's index_select backward is ``zeros.index_add_(0, gather, grad)``
    — an atomic, nondeterministic-order scatter (measured 82.3 ms/iter at
    waymo scale, the #1 single cost). The fused layout has structure the
    scatter ignores:

      * own segment — every source row appears EXACTLY once, at
        ``own_dst[tok]`` ⇒ its grad contribution is a pure permutation
        gather ``grad.index_select(0, own_dst)`` (coalesced, no sort).
      * neighbour segment — duplicated rows ⇒ grad is a segment-sum of grad
        rows grouped by source row, reduced as an ORDERED, ATOMIC-CONFLICT-
        FREE reduction: forward builds (two stable int32-key argsorts, ONCE)
        the rows reordered by occurrence rank within their equal-source
        group. Within one rank every source row appears AT MOST ONCE, so
        backward runs one unique-index ``index_add_`` per rank into a
        ≥fp32 accumulator, in fixed host-sequential rank order. Unique
        indices ⇒ each output element receives exactly ONE add per round ⇒
        the value is independent of intra-kernel thread order; the rank
        loop fixes the summation order across rounds ⇒ bitwise run-to-run
        deterministic, with O(T·D) total traffic (each grad row touched
        once — no log(max-dup) full-width passes).

    Determinism choice — alternatives rejected: the in-repo Triton
    ``large_segment_reduce("sum")`` ``tl.atomic_add``s partial block sums
    into the output (fast AND slow path), so any segment spanning >1
    ``BLOCK_N`` block reduces in nondeterministic order;
    ``torch.segment_reduce``'s CUDA determinism is undocumented (unprovable
    without reading shipped binaries); the padded ``[num_unique, max_occ]``
    matrix is deterministic but memory-unbounded under duplication skew.

    The rank count is bounded host-side by the max neighbour-cell in-degree
    (see ``_build_fused_kv``), batched into the existing single D2H sync.
    NOT double-differentiable (the env="0" atomic path is, if that is ever
    needed).
    """
    return k_rot.index_select(0, gather), v_s.index_select(0, gather)


@_gather_fused_kv.register_fake
def _gather_fused_kv_fake(k_rot, v_s, gather, own_dst, dst_by_rank,
                          src_by_rank, rank_bounds):
    # output rows = gather length (a row per gathered triplet); feature dims
    # carry through. Pure function of input shapes — data-independent meta.
    T = gather.shape[0]
    return (k_rot.new_empty((T,) + tuple(k_rot.shape[1:])),
            v_s.new_empty((T,) + tuple(v_s.shape[1:])))


def _gather_fused_kv_setup_context(ctx, inputs, output):
    k_rot, v_s, gather, own_dst, dst_by_rank, src_by_rank, rank_bounds = inputs
    ctx.save_for_backward(own_dst, dst_by_rank, src_by_rank)
    ctx.rank_bounds = rank_bounds
    ctx.N = k_rot.size(0)


@torch.autograd.function.once_differentiable
def _gather_fused_kv_backward(ctx, grad_k, grad_v):
    own_dst, dst_by_rank, src_by_rank = ctx.saved_tensors
    if dst_by_rank is not None:
        # Accumulate in (at least) fp32 — never downcast (fp64 stays
        # fp64). Own contribution first (a pure permutation gather),
        # then one unique-index index_add_ per occurrence rank.
        acc_dtype = torch.promote_types(grad_k.dtype, torch.float32)
        if own_dst is not None:
            dk_acc = grad_k.index_select(0, own_dst).to(acc_dtype)
            dv_acc = grad_v.index_select(0, own_dst).to(acc_dtype)
        else:
            shp = (ctx.N,) + grad_k.shape[1:]
            dk_acc = torch.zeros(shp, device=grad_k.device,
                                 dtype=acc_dtype)
            dv_acc = torch.zeros(shp, device=grad_v.device,
                                 dtype=acc_dtype)
        rb = ctx.rank_bounds
        for j in range(len(rb) - 1):
            lo, hi = rb[j], rb[j + 1]
            if hi == lo:
                continue       # empty round (loose dup bound) — no-op
            idx = src_by_rank[lo:hi]
            dk_acc.index_add_(
                0, idx,
                grad_k.index_select(0, dst_by_rank[lo:hi]).to(acc_dtype))
            dv_acc.index_add_(
                0, idx,
                grad_v.index_select(0, dst_by_rank[lo:hi]).to(acc_dtype))
        dk = dk_acc.to(grad_k.dtype)
        dv = dv_acc.to(grad_v.dtype)
    elif own_dst is not None:
        # own-only gather (no neighbour rows): exact permutation, no
        # accumulator round-trip — bit-identical to grad reordering.
        dk = grad_k.index_select(0, own_dst)
        dv = grad_v.index_select(0, own_dst)
    else:
        dk = grad_k.new_zeros((ctx.N,) + grad_k.shape[1:])
        dv = grad_v.new_zeros((ctx.N,) + grad_v.shape[1:])
    return dk, dv, None, None, None, None, None


_gather_fused_kv.register_autograd(
    _gather_fused_kv_backward,
    setup_context=_gather_fused_kv_setup_context)


def _build_fused_kv(
    nbr_cell: torch.Tensor,     # [E] int64, CSR by query cell
    nbr_octave: torch.Tensor,   # [E] int64
    nbr_cu: torch.Tensor,       # [num_cells + 1] int32
    ring_mult_t: torch.Tensor,  # [num_octaves] float32 — per-ring multipliers
    num_octaves: int,
    cu_seqlens: torch.Tensor,   # [num_cells + 1] int32 — own-cell offsets
    k_rot: torch.Tensor,        # [N, H, E] sorted, pre-rotated K
    v_s: torch.Tensor,          # [N, H, E] sorted, raw V
    num_cells: int,
    device: torch.device,
    coords_s: Optional[torch.Tensor] = None,  # [N,3] sorted — gathered for
                                              # the fused-RoPE KV coord frame
    own_segment: bool = True,                 # False ⇒ neighbour-only K/V
                                              # (LSE-merge dual-launch path)
    materialize: bool = True,                 # False ⇒ skip the K/V
                                              # index_selects; return the
                                              # int32 gather index instead
                                              # (PNT_MSGA_INDEXED_KV path)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, Optional[torch.Tensor]]:
    """Per query cell c: fused K/V = [ own k_rot[cu[c]:cu[c+1]] ] ⊕
    [ for each edge e of c: ``entry_reps[e]`` tokens **strided over the
    neighbour cell's stored token order** ], where ``entry_reps`` is the
    per-ring own-cell-relative budget (``_ring_budget_entry_reps``). The
    strided source positions are ``cu[j] + floor(arange(b)·s/b)`` (s = nbr
    size, b = entry_reps[e]) — an even subsample across the neighbour
    cell's *cell-sorted* index range.

    NOTE: this is a uniform stride over the cell's TOKEN ORDER, which is the
    grid-hash sort order (``torch.sort(cell_hash)``); since all tokens in a
    cell share one hash, that order is effectively the original-index
    permutation — NOT a spatial ordering. So the subsample is an unbiased
    *count*-uniform draw of the cell's points, not a spatially-stratified
    (Morton/FPS) one. (A spatially-principled representative selection is a
    deliberate-design question owned by Goal-5; it would change outputs and
    is gated on a quality A/B, not an operator refactor.)

    Vectorised; cross-cell overlap of a shared neighbour is realised by
    gather-duplication. Returns int32 ``fused_cu``.

    ``own_segment=False`` (LSE-merge dual-launch composition) omits the
    own-segment scatter entirely and returns the NEIGHBOUR-ONLY gather
    ``(k_nbr, v_nbr, nbr_only_cu, max_nbr, coords_nbr)`` — same
    ``entry_reps`` budget, same per-edge strided source positions
    (byte-identical to the fused path's neighbour segment), only the
    destination layout drops the per-cell own prefix.

    ``materialize=False`` (indexed fused-KV, ``PNT_MSGA_INDEXED_KV=1``):
    builds the SAME gather index but SKIPS the two
    ``index_select``s — returns ``(gather_int32, None, fused_cu, max_k,
    None)``; the FA-3 fork's ``kv_row_indices`` kernel param materializes
    the rows inside the attention gmem loads instead. Byte-identical
    behaviour to before when ``materialize=True`` (the default).
    """
    N = k_rot.size(0)
    cu = cu_seqlens.long()
    own_len = cu[1:] - cu[:-1]
    E_nbr = int(nbr_cell.numel())

    deg = (nbr_cu[1:] - nbr_cu[:-1]).to(torch.int64)
    edge_qc = torch.repeat_interleave(
        torch.arange(num_cells, device=device, dtype=torch.int64), deg,
        output_size=E_nbr)

    entry_reps = _ring_budget_entry_reps(
        nbr_cell, nbr_octave, edge_qc, own_len,
        num_cells, num_octaves, device, ring_mult_t)

    nbr_kv_per_cell = torch.zeros(num_cells, device=device, dtype=torch.int64)
    if E_nbr > 0:
        nbr_kv_per_cell.scatter_add_(0, edge_qc, entry_reps)
    Lk = (own_len + nbr_kv_per_cell) if own_segment else nbr_kv_per_cell
    fused_cu = torch.zeros(num_cells + 1, device=device, dtype=torch.int64)
    fused_cu[1:] = Lk.cumsum(0)

    # Two host scalars are genuinely needed (a buffer alloc size + flash's
    # host-int max_seqlen); batch BOTH into ONE device→host copy instead of
    # separate .item() syncs. total_nbr is NOT synced — it is the exact
    # integer identity total_kv − N (own segment is N tokens; the rest is the
    # neighbour gather), so it costs zero extra round-trips. (own_segment=False:
    # the gather IS the neighbour segment, so total_nbr == total_kv.)
    det_bwd = (materialize and _det_kv_bwd_enabled() and torch.is_grad_enabled()
               and (k_rot.requires_grad or v_s.requires_grad))
    if det_bwd and E_nbr > 0:
        # Third host scalar — the det backward's occurrence-rank bound —
        # batched into the SAME single D2H copy (no extra sync). Each edge
        # picks DISTINCT rows of its neighbour cell (entry_reps ≤ cell size
        # ⇒ the strided offsets floor(l·s/b), b ≤ s, are strictly
        # increasing), so a source row repeats in the nbr segment at most
        # indeg(its cell) times. A loose bound only adds no-op rounds,
        # never wrong values (integer bincount is exact ⇒ deterministic).
        indeg_max = torch.bincount(nbr_cell, minlength=num_cells).max()
        total_kv, max_k, max_dup_bound = torch.stack(
            (fused_cu[-1], Lk.max(), indeg_max)).tolist()
    else:
        total_kv, max_k = torch.stack((fused_cu[-1], Lk.max())).tolist()
        max_dup_bound = 0
    total_nbr = (total_kv - N) if own_segment else total_kv

    gather = torch.empty(total_kv, device=device, dtype=torch.int64)

    if own_segment:
        # own segment (output_size: own lengths cover all N tokens by the
        # partition invariant cu_seqlens[-1] == N)
        tok_cell = torch.repeat_interleave(
            torch.arange(num_cells, device=device, dtype=torch.int64), own_len,
            output_size=N)
        tok = torch.arange(N, device=device, dtype=torch.int64)
        own_dst = fused_cu[tok_cell] + (tok - cu[:-1][tok_cell])
        gather[own_dst] = tok

    # neighbour segments — per-edge UNIFORM STRIDE over the neighbour cell.
    # Guard on total_nbr (== entry_reps.sum()): when zero the whole block is a
    # no-op (empty aranges, empty scatter), so no separate sum().item() sync is
    # needed — total_nbr is already derived above.
    det_struct = None
    if E_nbr > 0 and total_nbr > 0:
        edge_cu = torch.zeros(E_nbr + 1, device=device, dtype=torch.int64)
        edge_cu[1:] = entry_reps.cumsum(0)
        cum = edge_cu[:-1]
        has_edges = deg > 0
        first_idx = nbr_cu[:-1].to(torch.int64)
        first_edge_cum = torch.zeros(num_cells, device=device, dtype=torch.int64)
        first_edge_cum[has_edges] = cum[first_idx[has_edges]]
        within = cum - first_edge_cum[edge_qc]
        own_off = own_len[edge_qc] if own_segment else 0
        edge_dst_base = fused_cu[edge_qc] + own_off + within

        edge_id = torch.repeat_interleave(
            torch.arange(E_nbr, device=device, dtype=torch.int64), entry_reps,
            output_size=total_nbr)
        local = (torch.arange(total_nbr, device=device, dtype=torch.int64)
                 - edge_cu[edge_id])                          # 0..b_e-1 within the edge
        # Uniform stride within the neighbour cell: floor(local·s/b), clamped.
        b_of = entry_reps[edge_id]                            # [total_nbr]
        s_of = own_len[nbr_cell[edge_id]]                     # neighbour cell size
        start_of = cu[:-1][nbr_cell[edge_id]]                 # sorted-order cell start
        offset = (local * s_of) // b_of.clamp(min=1)
        src = start_of + offset.clamp(max=s_of - 1)
        dst = edge_dst_base[edge_id] + local
        gather[dst] = src
        if det_bwd:
            det_struct = _det_gather_structure(src, dst, N, max_dup_bound,
                                               device)

    if not materialize:
        # Indexed fused-KV: no K/V HBM materialization — the int32 gather
        # index rides in the k slot; the kernel does the row lookup.
        return gather.to(torch.int32), None, fused_cu.to(torch.int32), max_k, None

    if det_bwd:
        # Deterministic structure-exploiting backward (PNT_MSGA_DET_KV_BWD,
        # default on): forward is the SAME index_select bit-exactly; only
        # the backward changes (atomic scatter → permutation gather +
        # ordered occurrence-rank segment-sum). See _GatherFusedKV.
        ds = det_struct if det_struct is not None else (None, None, None)
        k_fused, v_fused = _gather_fused_kv(
            k_rot, v_s, gather, own_dst if own_segment else None, *ds)
    else:
        k_fused = k_rot.index_select(0, gather)
        v_fused = v_s.index_select(0, gather)
    # coords carry no grad — the plain gather stays.
    coords_fused = (coords_s.index_select(0, gather)
                    if coords_s is not None else None)
    return k_fused, v_fused, fused_cu.to(torch.int32), max_k, coords_fused


def _build_nbr_only_kv(
    nbr_cell: torch.Tensor,     # [E] int64, CSR by query cell
    nbr_octave: torch.Tensor,   # [E] int64
    nbr_cu: torch.Tensor,       # [num_cells + 1] int32
    ring_mult_t: torch.Tensor,  # [num_octaves] float32
    num_octaves: int,
    cu_seqlens: torch.Tensor,   # [num_cells + 1] int32 — own-cell offsets
    k_rot: torch.Tensor,        # [N, H, E] sorted, pre-rotated K
    v_s: torch.Tensor,          # [N, H, E] sorted, raw V
    num_cells: int,
    device: torch.device,
    coords_s: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, Optional[torch.Tensor]]:
    """NEIGHBOUR-ONLY K/V gather for the LSE-merge dual-launch composition.

    Returns ``(k_nbr, v_nbr, nbr_only_cu, max_nbr, coords_nbr)`` where
    ``k_nbr/v_nbr`` are the [total_nbr = total_kv − N] budgeted neighbour
    rows (same ``entry_reps`` budget + per-edge strided source positions as
    the fused path's neighbour segment, byte-identical), ``nbr_only_cu`` is
    the int32 [C+1] cumulative neighbour-row count per query cell and
    ``max_nbr`` the host-int max segment length (may be 0). The own-cell
    segment is NOT materialised — the own launch uses k_rot/v_s in place.
    """
    return _build_fused_kv(
        nbr_cell, nbr_octave, nbr_cu, ring_mult_t, num_octaves,
        cu_seqlens, k_rot, v_s, num_cells, device,
        coords_s=coords_s, own_segment=False,
    )


def _sdpa_varlen_block_diagonal(q, k, v, cu_q, cu_k, scale):
    """Masked block-diagonal SDPA equivalent of the varlen flash launch.

    Decisive-test fallback for the cyc-35 fused-KV corruption forensic
    (forensic wf_b8fde84a). Mathematically equivalent to
    ``flash_attn_varlen_func(q, k, v, cu_q, cu_k, ...)`` with MISMATCHED
    cu (cu_q == own-cell lengths, cu_k == own+nbr lengths): per query cell
    c, attends q[cu_q[c]:cu_q[c+1]] over k/v[cu_k[c]:cu_k[c+1]] with one
    independent softmax (no cross-cell leakage). Same dtype as ``q`` (bf16
    on the cluster path); accumulates the per-cell matmuls in fp32 for
    numerical headroom, then casts the output back to q.dtype. Loops over
    cells in Python (num_cells is O(thousands), each cell is a tiny GEMM) —
    this is a diagnostic kernel, not the perf path.
    """
    cu_q_l = cu_q.long().tolist()
    cu_k_l = cu_k.long().tolist()
    num_cells = len(cu_q_l) - 1
    out = torch.empty_like(q)
    for c in range(num_cells):
        qs, qe = cu_q_l[c], cu_q_l[c + 1]
        ks, ke = cu_k_l[c], cu_k_l[c + 1]
        if qe == qs:
            continue
        q_c = q[qs:qe].float()                       # [Lq, H, D]
        k_c = k[ks:ke].float()                       # [Lk, H, D]
        v_c = v[ks:ke].float()                       # [Lk, H, D]
        # [H, Lq, D] @ [H, D, Lk] -> [H, Lq, Lk]
        scores = torch.einsum("qhd,khd->hqk", q_c, k_c) * scale
        attn = torch.softmax(scores, dim=-1)
        out_c = torch.einsum("hqk,khd->qhd", attn, v_c)   # [Lq, H, D]
        out[qs:qe] = out_c.to(out.dtype)
    return out


_FA3_AUTO_CACHE: dict = {}


def _fa3_auto_decision(head_dim: int, cc_major: int, wheel_ok: bool) -> bool:
    """Pure decision table for 'auto' backend resolution (unit-testable)."""
    return head_dim >= 64 and cc_major >= 9 and wheel_ok


def _fa3_auto(head_dim: int, device: torch.device) -> bool:
    key = (int(head_dim), device)
    hit = _FA3_AUTO_CACHE.get(key)
    if hit is None:
        try:
            import flash_attn_3d_rope_fa3  # noqa: F401
            wheel_ok = True
        except ImportError:
            wheel_ok = False
        cc_major = torch.cuda.get_device_capability(device)[0]
        hit = _fa3_auto_decision(head_dim, cc_major, wheel_ok)
        _FA3_AUTO_CACHE[key] = hit
    return hit


_INDEXED_KV_AUTO_CACHE: dict = {}


def _indexed_kv_auto_decision(head_dim: int, cc: tuple, wheel_ok: bool) -> bool:
    """Pure decision table for indexed-KV 'auto' resolution (unit-testable).

    Stage-B verdict (2026-06-12, H200 cell 310253119, 3_plans/
    msga_indexed_kv.md §7): on sm_90 with the native fa3 wheel the indexed
    path is fwd-BITWISE vs materialized-through-same-kernel and wins
    fwd+bwd −21–35% with −5–10 GB peak at ~1M tokens — adopt by default.
    sm_89 adopted 2026-06-12 (§11 Ada bench, arms I2-vs-A2 at matched k:
    fwd+bwd −36/−41/−36% scannet/waymo/scannet3oct, peak −0.13/−0.76 GB;
    CI2 reconfirms prerotate over fused rotation on fa2) — the indexed
    launch rides the fa2 fork there via backend-auto. sm_80 stays out: the
    fork wheels ship sm_89+sm_90 cubins only, and the case is unmeasured.
    ``wheel_ok`` must certify the BACKEND-MATCHED wheel (fa3 on sm_90, fa2
    on sm_89 — see ``_indexed_kv_auto``). head_dim == 64 exactly: the
    measured case on both arches AND the backend-auto agreement point;
    below 64 indexed timing is unmeasured (fa3 pays pad-to-64; fa2 hd<64
    is f-leg parity-covered but unbenched) — available via the explicit
    "1" only.
    """
    cc_t = tuple(cc)
    return head_dim == 64 and wheel_ok and (cc_t[0] >= 9 or cc_t >= (8, 9))


_FRAG_MEANQ_THRESHOLD = 96   # mean cell occupancy (N / num_cells) at/below
# which the indexed launch routes to fa2@64 instead of fa3@192 — the SAME
# fragmentation signal the fa2/fa3 small-tile fwd dispatch uses (total_q/b).


def _frag_route_to_fa2(mean_q: float) -> bool:
    """Whether an AUTO indexed launch (otherwise fa3 at sm_90/hd64) should
    route to fa2 because the workload is fragmented.

    fa3-indexed rides the PagedKVNonTMA pipeline (no TMA — traded for
    no-materialization) at a 192 tile, so on tiny-cell workloads its
    tile-quantization padding tax dominates and fa2@64-indexed is faster at
    FLAT memory (H200 cell 310808864: nuScenes fa3 +102% fwd / +21% fb,
    waymo +11% / +5%; fa3 wins only dense indoor scannet ~2-7%). The
    crossover lands on the existing mean-q small-tile threshold. fa2's
    own kernel then auto-picks the 64 tile (FA3D_FA2_FWD_TILE auto-rule).
    PNT_MSGA_FRAG_ROUTE=0 reverts to fa3-always at sm_90/hd64 (e.g. to A/B
    or if a future fa3 indexed-TMA path changes the economics)."""
    return (os.environ.get("PNT_MSGA_FRAG_ROUTE", "1") != "0"
            and mean_q <= _FRAG_MEANQ_THRESHOLD)


def _indexed_kv_auto(head_dim: int, device: torch.device) -> bool:
    key = (int(head_dim), device)
    hit = _INDEXED_KV_AUTO_CACHE.get(key)
    if hit is None:
        cc = tuple(torch.cuda.get_device_capability(device))
        # The indexed launch rides fa3 on sm_90 (backend-auto via _fa3_auto)
        # and fa2 on sm_8x — certify the wheel the launch will actually use.
        wheel_name = ("flash_attn_3d_rope_fa3" if cc[0] >= 9
                      else "flash_attn_3d_rope_fa2")
        try:
            __import__(wheel_name)
            wheel_ok = True
        except ImportError:
            wheel_ok = False
        hit = _indexed_kv_auto_decision(head_dim, cc, wheel_ok)
        _INDEXED_KV_AUTO_CACHE[key] = hit
    return hit


def _fa_varlen_attention(q, k, v, cu_q, cu_k, max_q, max_k, scale,
                         fa_backend="auto"):
    """Single flash_attn varlen launch (inlined — no NSA-module dep).

    PNT_MSGA_SDPA_FALLBACK=1 swaps ONLY this launch for a masked
    block-diagonal SDPA over the SAME q/k/v/cu_q/cu_k (decisive cyc-35
    corruption test — same trained weights, same architecture, same dtype;
    only the attention kernel changes). Default ('0') returns the original
    flash call BIT-IDENTICALLY (early return before any new code touches
    the inputs).

    ``fa_backend='fa3'`` (or env ``PNT_MSGA_FA_BACKEND=fa3``, which
    overrides the kwarg) routes this launch to the FA-3 fork's unpacked
    mismatched-cu entry (``flash_attn_3d_rope_fa3_varlen_func``,
    pass-through — q/k arrive pre-rotated, same math) — sm_90 only,
    head_dim>=64 native. 'auto'/'fa2' keep the stock flash_attn launch
    BIT-IDENTICALLY (fa3 is opt-in until the H200 parity cell clears it;
    a different kernel is not bit-exact vs FA-2 even at identical math).
    """
    cu_q = cu_q.cuda() if cu_q.is_cpu else cu_q
    cu_k = cu_k.cuda() if cu_k.is_cpu else cu_k
    if os.environ.get("PNT_MSGA_SDPA_FALLBACK", "0") == "1":
        return _sdpa_varlen_block_diagonal(q, k, v, cu_q, cu_k, scale)
    fa_backend = os.environ.get("PNT_MSGA_FA_BACKEND", fa_backend)
    if fa_backend == "auto" and _fa3_auto(q.shape[-1], q.device):
        # cyc-42 variant-matrix verdict (2026-06-11): native-sm_90a FA3
        # passthrough is parity-green and never loses vs stock FA2 (waymo-
        # scale fwd+bwd −13.4%; Hopper bwd kernel) — so 'auto' resolves to
        # fa3 at head_dim>=64 on sm_90 with the wheel importable. Explicit
        # fa_backend='fa2' (or env PNT_MSGA_FA_BACKEND=fa2) forces stock.
        fa_backend = "fa3"
    if fa_backend == "fa3":
        from .flash_attn_3d_rope import flash_attn_3d_rope_fa3_varlen_func
        return flash_attn_3d_rope_fa3_varlen_func(
            q, k, v, cu_q, cu_k, max_q, max_k,
            softmax_scale=scale, causal=False,
        )
    return _fa_varlen_fn(
        q, k, v, cu_q, cu_k, max_q, max_k,
        dropout_p=0.0, softmax_scale=scale, causal=False,
    )


def _pad_head(t: torch.Tensor, pad: int) -> torch.Tensor:
    """flash_attn requires head_dim % 8 == 0; pad trailing channels."""
    return torch.nn.functional.pad(t, [0, pad]) if pad else t


@custom_op("pnt_msga::lse_merged_attention", mutates_args=())
def _lse_merged_attention_op(
    q: torch.Tensor, k_own: torch.Tensor, v_own: torch.Tensor,
    k_nbr: torch.Tensor, v_nbr: torch.Tensor, cu: torch.Tensor,
    nbr_cu: torch.Tensor, max_q: int, max_nbr: int, scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """LSE-merge dual-launch KV composition (``PNT_MSGA_KV_COMPOSE=merged``).

    torch.library.custom_op (not autograd.Function) so it traces under
    torch.compile as a registered opaque leaf; register_autograd below carries
    the IDENTICAL exact ring-attention backward. Returns ``(out, lse)`` — the
    wrapper keeps ``out``; ``lse`` (the joint log-sum-exp, already computed in
    this forward) is carried out so the backward REUSES it instead of
    recomputing the two flash forwards (compute-neutral conversion).

    Forward: TWO flash varlen launches —
      * own:  q_rot vs (k_rot, v_s), cu_q == cu_k == prep.cu_seqlens (no
        gather at all; the own segment of the fused layout is a pure copy of
        the contiguous sorted k_rot/v_s, so the tensors are used in place);
      * nbr:  q_rot vs the budgeted NEIGHBOUR-ONLY gather (total_kv − N
        rows), cu_k = nbr-only cumulative lengths.
    merged EXACTLY into the single joint softmax via log-sum-exp: with
    fp32 ``lse_own/lse_nbr`` [H, total_q] from flash,
    ``m = max(lse_own, lse_nbr)``, ``w_b = exp(lse_b − m)``,
    ``out = (w_own·out_own + w_nbr·out_nbr) / (w_own + w_nbr)`` and the
    joint LSE is ``m + log(w_own + w_nbr)``. Cells with ZERO neighbour rows
    produce empty k segments — flash_attn 2.8.3 returns out=0 there but
    lse=+inf (NOT −inf; verified empirically), so the nbr lse is sanitised
    +inf→−inf before the merge, after which it degenerates to the own
    branch (w_nbr = exp(−inf − m) = 0). If total_nbr == 0 overall the nbr
    launch is skipped entirely.

    Backward (EXACT — ring-attention/context-parallel decomposition): naive
    autograd through the merge would be WRONG (flash's returned lse carries
    no grad), so each branch calls ``_flash_attn_varlen_backward`` with the
    JOINT out + JOINT lse and the branch's own k/v — that reproduces the
    branch's exact contribution to the joint gradient (flash bwd recomputes
    P_branch = exp(qkᵀ·scale − lse_joint) and D = rowsum(dout ∘ out_joint)
    internally). ``dq = dq_own + dq_nbr``; dk/dv per branch flow back to
    their inputs (the nbr gather's index_select backward scatters dk_nbr/
    dv_nbr onto k_rot/v_s).
    """
    from flash_attn import flash_attn_interface as _fai
    head_size_og = q.size(2)
    pad = (8 - head_size_og % 8) % 8
    q, k_own, v_own, k_nbr, v_nbr = (
        _pad_head(t.contiguous(), pad)
        for t in (q, k_own, v_own, k_nbr, v_nbr))
    out_own, lse_own, _, _ = _fai._flash_attn_varlen_forward(
        q, k_own, v_own, cu, cu, max_q, max_q,
        dropout_p=0.0, softmax_scale=scale, causal=False,
    )
    total_nbr = int(k_nbr.size(0))
    if total_nbr == 0:
        out, lse = out_own, lse_own
    else:
        out_nbr, lse_nbr, _, _ = _fai._flash_attn_varlen_forward(
            q, k_nbr, v_nbr, cu, nbr_cu, max_q, max_nbr,
            dropout_p=0.0, softmax_scale=scale, causal=False,
        )
        # flash returns lse=+inf (and out=0) for ZERO-LENGTH k segments
        # (flash_attn 2.8.3, verified) — sanitise to −inf so the merge
        # weight is exp(−inf − m) = 0, not exp(+inf − +inf) = NaN.
        lse_nbr = torch.where(
            lse_nbr == float("inf"),
            torch.full_like(lse_nbr, float("-inf")), lse_nbr)
        # Exact joint-softmax merge in fp32. lse_own is finite everywhere
        # (own cells are never empty), so m is finite and exp(−inf − m)=0
        # handles empty nbr segments without NaNs.
        m = torch.maximum(lse_own, lse_nbr)              # [H, Tq] fp32
        w_own = torch.exp(lse_own - m)
        w_nbr = torch.exp(lse_nbr - m)
        denom = w_own + w_nbr
        wo = (w_own / denom).transpose(0, 1).unsqueeze(-1)   # [Tq, H, 1]
        wn = (w_nbr / denom).transpose(0, 1).unsqueeze(-1)
        out = (wo * out_own.float()
               + wn * out_nbr.float()).to(out_own.dtype)
        lse = m + torch.log(denom)                       # joint LSE
    out = out[..., :head_size_og] if pad else out
    return out, lse


@_lse_merged_attention_op.register_fake
def _lse_merged_attention_fake(q, k_own, v_own, k_nbr, v_nbr, cu, nbr_cu,
                               max_q, max_nbr, scale):
    # out is q's shape (joint attention output, per query token); lse is
    # [H, Tq] fp32 (flash's per-head per-query log-sum-exp). Both are pure
    # functions of input shapes — data-independent meta.
    H = q.shape[1]
    Tq = q.shape[0]
    out = torch.empty_like(q)
    lse = q.new_empty((H, Tq), dtype=torch.float32)
    return out, lse


def _lse_merged_attention_setup_context(ctx, inputs, output):
    q, k_own, v_own, k_nbr, v_nbr, cu, nbr_cu, max_q, max_nbr, scale = inputs
    out, lse = output
    ctx.save_for_backward(q, k_own, v_own, k_nbr, v_nbr, out, lse, cu, nbr_cu)
    ctx.max_q = max_q
    ctx.max_nbr = max_nbr
    ctx.scale = scale


@torch.autograd.function.once_differentiable
def _lse_merged_attention_backward(ctx, dout, dlse):
    # dlse is the (unused) cotangent of the carried-out lse — the joint LSE is
    # a forward-only intermediate (flash's lse carries no grad), so dlse is
    # ignored exactly as the autograd.Function never exposed lse as an output.
    from flash_attn import flash_attn_interface as _fai
    (q, k_own, v_own, k_nbr, v_nbr, out, lse, cu,
     nbr_cu) = ctx.saved_tensors
    head_size_og = dout.size(2)
    pad = (8 - head_size_og % 8) % 8
    # re-pad/contiguify is a NO-OP on every real path (head_dim%8==0 ⇒ pad=0 ⇒
    # _pad_head returns t unchanged; rotated inputs are contiguous), so this
    # introduces no extra compute. `out`/`lse` (the costly forward
    # intermediates) are reused via save_for_backward, never recomputed.
    q, k_own, v_own, k_nbr, v_nbr = (
        _pad_head(t.contiguous(), pad)
        for t in (q, k_own, v_own, k_nbr, v_nbr))
    out = _pad_head(out.contiguous(), pad)
    dout = _pad_head(dout.contiguous(), pad)
    dq = torch.empty_like(q)
    dk_own = torch.empty_like(k_own)
    dv_own = torch.empty_like(v_own)
    # deterministic=False matches flash_attn_varlen_func's default (the
    # current single-launch behaviour); dropout/window/softcap/alibi off.
    _fai._flash_attn_varlen_backward(
        dout, q, k_own, v_own, out, lse, dq, dk_own, dv_own,
        cu, cu, ctx.max_q, ctx.max_q,
        dropout_p=0.0, softmax_scale=ctx.scale, causal=False,
        window_size_left=-1, window_size_right=-1, softcap=0.0,
        alibi_slopes=None, deterministic=False, rng_state=None,
    )
    if int(k_nbr.size(0)) > 0:
        dq_nbr = torch.empty_like(q)
        dk_nbr = torch.empty_like(k_nbr)
        dv_nbr = torch.empty_like(v_nbr)
        _fai._flash_attn_varlen_backward(
            dout, q, k_nbr, v_nbr, out, lse, dq_nbr, dk_nbr, dv_nbr,
            cu, nbr_cu, ctx.max_q, ctx.max_nbr,
            dropout_p=0.0, softmax_scale=ctx.scale, causal=False,
            window_size_left=-1, window_size_right=-1, softcap=0.0,
            alibi_slopes=None, deterministic=False, rng_state=None,
        )
        dq = dq + dq_nbr
    else:
        dk_nbr = torch.zeros_like(k_nbr)
        dv_nbr = torch.zeros_like(v_nbr)
    if pad:
        dq, dk_own, dv_own, dk_nbr, dv_nbr = (
            t[..., :head_size_og] for t in (dq, dk_own, dv_own, dk_nbr, dv_nbr))
    return (dq, dk_own, dv_own, dk_nbr, dv_nbr,
            None, None, None, None, None)


_lse_merged_attention_op.register_autograd(
    _lse_merged_attention_backward,
    setup_context=_lse_merged_attention_setup_context)


def _lse_merged_attention(q_rot, k_rot, v_s, k_nbr, v_nbr, cu, nbr_cu,
                          max_q, max_nbr, scale):
    """Dual-launch + exact LSE merge (see ``_lse_merged_attention_op``)."""
    cu = cu.cuda() if cu.is_cpu else cu
    nbr_cu = nbr_cu.cuda() if nbr_cu.is_cpu else nbr_cu
    out, _lse = _lse_merged_attention_op(
        q_rot, k_rot, v_s, k_nbr, v_nbr, cu, nbr_cu, max_q, max_nbr, scale)
    return out


# `@torch.compiler.disable` (recursive=True by default) — KEPT, by design, and
# NOT a compute-neutral candidate for removal (G30):
#   * the K/V-gather *builders* (`_build_fused_kv`, `_ring_budget_entry_reps`,
#     `build_cell_multiscale_idx`) emit tensors whose leading dim is a
#     VALUE-dependent count (Hamilton ring budget, neighbour in-degree) — an
#     irreducibly data-dependent output shape that no `register_fake` can give;
#   * the function does host-side backend/env dispatch (`os.environ`,
#     `__import__(wheel_name)` arch probing) that MUST run eager.
# The gradient-bearing inner helpers (gather + LSE-merge) ARE now compile-safe
# `custom_op`s (`pnt_msga::gather_fused_kv`, `pnt_msga::lse_merged_attention`,
# register_autograd + register_fake) — so they trace as registered leaves
# wherever the surrounding region IS compiled. But the attention call itself
# (`flash_attn`) is opaque CUDA — Dynamo graph-breaks there regardless. One
# clean opaque boundary here beats mid-function breaks, and (measured) the
# trunk block already fuses optimally around it (norm/proj/MLP each compile as
# their own graph; only fusion ACROSS the attention is blocked — inherent, you
# never fuse a LayerNorm into a flash kernel). Forcing this open would inject an
# unbacked SymInt into the consumer (a break with NO fusion gain) or require
# extra compute — both ruled out. Cycle 26 Phase B.
@torch.compiler.disable
def multi_scale_grid_attention(
    q: torch.Tensor,                       # [N, H, E]
    k: torch.Tensor,                       # [N, H, E]
    v: torch.Tensor,                       # [N, H, E]
    coords: torch.Tensor,                  # [N, 3] fp32
    prep: GridAttentionPrep,
    scale: float,
    *,
    multiscale: tuple,                     # (nbr_cell[E]i64, nbr_octave[E]i64, nbr_cu[C+1]i32)
    ring_mult: Sequence[float],            # per-ring multipliers (own-cell-relative budget)
    chunk_dim: Optional[int] = None,
    rope_base: float = 10000.0,
    coord_scale: float = 1.0,
    use_fused_rope: bool = False,
    use_rotary_tables: bool = False,
    fa_backend: Literal["auto", "fa2", "fa3"] = "auto",
    enable_telemetry: bool = True,
    # cyc-28 corrupt-attention probe: when prep_kv is provided, gather K and V
    # via prep_kv.perm instead of prep.perm (Q + coords-for-Q still use prep).
    # prep_kv.cu_seqlens MUST equal prep.cu_seqlens (cell boundaries shared so
    # FLOPs are byte-identical to the un-corrupted forward). Output scatter
    # uses prep.inv_perm (residual stream lands at the right tokens). Default
    # None preserves pre-cyc-28 behaviour bit-exact.
    prep_kv: Optional[GridAttentionPrep] = None,
) -> torch.Tensor:                          # [N, H, E] in original order
    """Fused multi-scale grid attention.

    ONE joint softmax per query token over its own cell (full
    all-to-all, == GridAttention, segment 0) ∪ its multi-scale
    radius-octave neighbour representatives. Same in/out contract as
    ``grid_sdpa``: ``[N, H, E]`` in original token order, same dtype as
    ``q`` (fp16/bf16 pass through; fp32 runs the kernel in bf16 and
    restores fp32 — the ``grid_sdpa`` convention). Param-free; every
    point is a query and is updated.
    """
    # ── Input contract (fail loudly; downstream goal-5/goal-6 architectures
    # call this directly). Raise ValueError/TypeError with actionable messages
    # rather than bare asserts (which `python -O` strips and which give no
    # context deep inside the gather/flash kernels). ──
    _validate_msga_inputs(q, k, v, coords, prep, multiscale, ring_mult, prep_kv)
    device = q.device
    num_cells = int(prep.cu_seqlens.numel() - 1)
    nbr_cell, nbr_octave, nbr_cu = multiscale

    # cyc-28: when prep_kv given, K/V (and their RoPE coords) gather through a
    # different perm than Q's. cu_seqlens must match so cell sizes (and FLOPs)
    # are byte-identical.
    if prep_kv is None:
        prep_kv = prep

    if num_cells <= 1 or num_active_rings(ring_mult) == 0:
        # Two degenerate cases collapse to pure GridAttention (own-cell
        # all-to-all), returned bit-identically:
        #   - num_cells<=1: no neighbour cells exist.
        #   - ring_mult empty / all-zero (num_active_rings==0): the spec's
        #     GridAttention degenerate case (cyc-37 §2.4) — guards the op
        #     contract for direct callers (the model routes empty mult to the
        #     GridAttention module, never here).
        # cyc-28 note: with only ONE cell, prep_kv-shuffle gives all KV tokens
        # to that single cell anyway, which is what grid_sdpa already does.
        # The within-cell randomization of which Q sees which K is the only
        # remaining corruption signal at this scale; achieved by feeding
        # KV-permuted features into grid_sdpa via a synthetic prep_kv path is
        # complex and not worth it (single-cell is the degenerate case).
        # Fall through to bit-identical grid_sdpa; corruption is a no-op here.
        #
        # cyc-35 FIX (2026-06-04): FORCE use_fused_rope=False on this degenerate
        # single-cell path. The fused-RoPE path routes to the FA3-RoPE wheel
        # (fa2+coords raises NotImplementedError), whose **sm_90 (Hopper)
        # mainloop returns wrong logits for the single-segment varlen shape**
        # this path produces — exercised ONLY on the cluster (local Ada uses the
        # wheel's sm_80 mainloop, which is correct). Small TTA test fragments are
        # the only inputs that hit num_cells<=1 (large scenes / train / whole-
        # scene val never do), so the corruption surfaced only in cluster TTA-
        # test on small scenes (ScanNet test 0.61 vs val 0.745). The Triton-
        # prerotate + stock-flash path (use_fused_rope=False) is sm_90-correct.
        # CONFIRMED by job 306925216: same h1 model_best, single variable =
        # use_fused_rope — arm A (True/FA3 wheel) 0.6107 vs arm B (False) 0.7614
        # (== race-free local Ada TTA-test 0.7613 to 4 decimals). Single-cell
        # fragments are tiny, so forcing the unfused path costs ~nothing.
        # Root fix (FA3-wheel sm_90 mainloop) tracked as a needs-human gap.
        return grid_sdpa(
            q, k, v, coords, [prep], scale, aggregate="mean",
            chunk_dim=chunk_dim, rope_base=rope_base, coord_scale=coord_scale,
            use_fused_rope=False, use_rotary_tables=use_rotary_tables,
            fa_backend=fa_backend, enable_telemetry=enable_telemetry,
        )

    from .prerotate_rope_triton import (
        compute_chunk_dim_per_axis, prerotate_3d_rope, prerotate_3d_rope_qk,
        prerotate_3d_rope_qk_gathered, prerotate_3d_rope_gathered)
    E = q.size(2)
    cdv = chunk_dim if chunk_dim is not None else compute_chunk_dim_per_axis(E)

    # Gather-on-load (PNT_MSGA_GATHER_ON_LOAD, default ON since 2026-06-14 —
    # H200 smoke 311239920: byte-exact fwd vs OFF + speedup on sm_90; set =0 to
    # disable, which is bit-for-bit the pre-fusion path). When ON and the
    # production default prerotate path is
    # taken (prep_kv is prep, NOT combined-fused), the q/k feature gather and
    # the coords gather are FOLDED INTO the prerotate Triton kernel
    # (gather-on-load), eliminating the three separate index_select launches +
    # their HBM round-trips. The decision also needs `combined_fused` (computed
    # below), so the final gate is taken at the prerotate block; this flag is
    # the opt-in switch. Opt-in + trivially revertible, mirroring
    # PNT_MSGA_INDEXED_KV / PNT_MSGA_FRAG_ROUTE.
    _gather_on_load_env = os.environ.get("PNT_MSGA_GATHER_ON_LOAD", "1") == "1"
    # v_s (sorted V) is needed by every downstream compose path (merged /
    # indexed-#1 / materialized / fa3-fused-materialized) EXCEPT the V-gather-on-
    # load (#2) path, which reads RAW V through a perm-composed kv_row_indices —
    # so its materialization is DEFERRED until after the #2 gate below (it needs
    # indexed_kv/combined_fused/gather_on_load, all decided further down). q_s/k_s
    # are needed only by the combined-fused and fa3-fused-materialized branches
    # (NOT reached on the production default), so they too are deferred and
    # skipped when gather-on-load actually fires (see the prerotate block).
    # One absolute-prerotate frame each for Q-side and KV-side so the single
    # joint softmax is positionally coherent within its own (corrupted or
    # not) gather.
    # FA-3 fused-RoPE main path (sm_90, opt-in): skip the Triton prerotate
    # entirely — raw q_s/k_s go to the kernel, which rotates Q from coords_s
    # and K from the GATHERED fused-KV coords (coords_3d_k), max-channels
    # chunk split by default. Requires the shared-frame case (prep_kv is
    # prep); the cyc-28 corruption probe keeps the prerotate path.
    fa_backend_eff = os.environ.get("PNT_MSGA_FA_BACKEND", fa_backend)
    # KV composition: "fused" (default, byte-identical legacy path — ONE
    # varlen launch over the materialised [own ⊕ nbr] gather) vs "merged"
    # (dual-launch + exact LSE merge; PREROTATE path only — the fa3
    # fused-RoPE branch below keeps the fused composition).
    kv_compose = os.environ.get("PNT_MSGA_KV_COMPOSE", "fused")
    if kv_compose not in ("fused", "merged"):
        raise ValueError(
            f"multi_scale_grid_attention: PNT_MSGA_KV_COMPOSE must be "
            f"'fused' or 'merged', got {kv_compose!r}")

    # MSGA indexed fused-KV decision (hoisted above BOTH the fa3
    # fused-materialized branch and the prerotate block: the combined
    # indexed×fused mode must take precedence over the materialized fused
    # branch AND skip the Triton prerotation — the kernel rotates in-smem).
    # Skip the k_fused/v_fused HBM materialization — the fork's
    # kv_row_indices kernel param reads K/V rows straight from the source
    # buffers through the SAME gather index. Shared-frame only
    # (prep_kv is prep — the cyc-28 corruption probe keeps the prerotate
    # path) and fused-compose only (the merged dual-launch keeps its
    # prerotate+materialized form and takes precedence, as before).
    # Default AUTO since the Stage-B verdict (cell 310253119; decision table
    # in _indexed_kv_auto_decision): on sm_90 + fa3 wheel indexed wins
    # outright; sm_8x keeps materialized. The auto path defers to an
    # explicit fa2 escape (kwarg or PNT_MSGA_FA_BACKEND=fa2) and to the
    # PNT_MSGA_SDPA_FALLBACK diagnostic, and excludes GENUINELY-fused calls
    # (fused_eff below — fused rotation was validated-correct but rejected
    # for production by the §9 bench, +46–95% fb). dk/dv grads run at
    # dq-equivalent atomic determinism under indexed (out stays bitwise vs
    # materialized); PNT_MSGA_INDEXED_KV=0 forces materialized (+det-kv
    # default), =1 forces indexed (any arch).
    #
    # Backend routing: the indexed launch rides the backend the call
    # RESOLVES to — explicit kwarg/env wins; 'auto' resolves fa3 exactly
    # when the fa3 flip applies (_fa3_auto: sm_90 + hd>=64 + wheel), else
    # fa2 (the production backend below head_dim 64). Under the EXPLICIT
    # path (PNT_MSGA_INDEXED_KV=1) BOTH backends allow prerotate-indexed
    # AND fused-indexed (the combined indexed×fused mode — kernel rotates
    # Q from coords_s, K from the gathered coords_fused; rows load through
    # the index).
    _idx_env = os.environ.get("PNT_MSGA_INDEXED_KV", "auto")
    # Backend for the indexed launch. Explicit kwarg/env wins. AUTO:
    # fa3 at sm_90/hd64+wheel EXCEPT on fragmented workloads, where
    # fa2@64-indexed is faster at flat memory (cell 310808864) — route by
    # mean cell occupancy (N / num_cells), the small-tile signal. Below
    # sm_90/hd64 -> fa2. (Changes only WHICH backend the indexed launch
    # uses; `indexed_kv` below still gates on fa_backend_eff, unchanged.)
    _mean_q = q.shape[0] / max(num_cells, 1)
    if fa_backend_eff in ("fa2", "fa3"):
        _idx_backend = fa_backend_eff
    elif _fa3_auto(q.shape[-1], device):
        _idx_backend = "fa2" if _frag_route_to_fa2(_mean_q) else "fa3"
    else:
        _idx_backend = "fa2"
    # use_fused_rope arrives from the production model blocks as a legacy
    # speed HINT ("no-op under MSGA; speeds grid_sdpa" — pnt.py) with
    # fa_backend left on 'auto'. It denotes a GENUINE fused-rotation
    # request only when paired with an EXPLICIT backend (kwarg/env
    # fa2|fa3): the §9 verdict rejected fused rotation for production
    # (+46–95% fb vs prerotate at indexed loads), so the hint must neither
    # veto the indexed AUTO path nor trigger fused rotation by itself.
    # Under 'auto' the flag stays inert for MSGA — its historical contract
    # — and production lands on prerotate-indexed (arm I).
    fused_eff = use_fused_rope and fa_backend_eff in ("fa2", "fa3")
    indexed_kv = (
        (_idx_env == "1"
         or (_idx_env != "0"
             and not fused_eff
             and fa_backend_eff != "fa2"
             and os.environ.get("PNT_MSGA_SDPA_FALLBACK", "0") != "1"
             and _indexed_kv_auto(q.shape[-1], device)))
        and kv_compose == "fused"
        and prep_kv is prep)
    # Combined mode: indexed && genuinely-fused — explicit-opt-in only;
    # kv_compose=="fused" keeps the merged path's prerotate inputs intact.
    combined_fused = indexed_kv and fused_eff

    # Gather-on-load FINAL gate: only the production default prerotate path —
    # shared frame (prep_kv is prep) AND NOT combined-fused (combined feeds RAW
    # q_s/k_s to the kernel, so there is no prerotate gather to fold). When OFF,
    # q_s/k_s are gathered here exactly as before (bit-for-bit). When ON, the
    # q/k/coords prerotate gathers are folded into the kernel and q_s/k_s are not
    # materialized (downstream uses q_rot/k_rot + v_s only on this path).
    gather_on_load = _gather_on_load_env and (prep_kv is prep) and not combined_fused
    if not gather_on_load:
        # permute_rows = index_select with a permutation-aware (gather,
        # non-atomic) backward — bit-identical grads, no scatter.
        q_s = permute_rows(q, prep.perm, prep.inv_perm)
        k_s = permute_rows(k, prep_kv.perm, prep_kv.inv_perm)

    # V-gather-on-load (#2, PNT_MSGA_V_GATHER_ON_LOAD, default ON since 2026-06-15,
    # H200-validated job 311451065: byte-exact fwd on the fa3-indexed path +
    # +3.3–6.1% MSGA-isolated). On the indexed-KV path, eliminate V's per-block
    # gather by reading RAW V through a perm-composed kv_row_indices. The fork
    # addresses K and V via ONE SHARED index, so reading raw V forces reading K
    # from the UNSORTED buffer too: Q stays gathered+rotated (sorted, read
    # contiguously by cu_q) but K is rotated IN PLACE (no gather), and
    # prep_kv.perm is composed into the cached fused index so the kernel reads
    # K/V straight from the original-order buffers. Built ON TOP of
    # gather-on-load (#1, the Q fold); gated to the prerotate-indexed production
    # path (indexed_kv, not combined-fused — so inert on sm_8x where indexed is
    # auto-off). Forward is byte-exact (relabelled index over the same KV set in
    # the same per-cell order); dk/dv land at the indexed path's existing
    # atomic-determinism floor. =0 disables = the #1 path bit-for-bit.
    _v_gather_on_load_env = os.environ.get("PNT_MSGA_V_GATHER_ON_LOAD", "1") == "1"
    v_gather_on_load = (_v_gather_on_load_env and gather_on_load
                        and indexed_kv and not combined_fused)
    # Deferred V materialization (see the flag-read comment above): raw V on the
    # #2 path, sorted V everywhere else.
    v_s = None if v_gather_on_load else permute_rows(
        v, prep_kv.perm, prep_kv.inv_perm)

    if (not indexed_kv and fused_eff and fa_backend_eff == "fa3"
            and prep_kv is prep):
        coords_s = coords.index_select(0, prep.perm)
        num_octaves = num_active_rings(ring_mult)
        ring_mult_t = _ring_mult_tensor(ring_mult, device)
        k_fused, v_fused, fused_cu, max_k, coords_fused = _build_fused_kv(
            nbr_cell.to(torch.int64), nbr_octave.to(torch.int64), nbr_cu,
            ring_mult_t, num_octaves, prep.cu_seqlens, k_s, v_s,
            num_cells, device, coords_s=coords_s,
        )
        cu_q = prep.cu_seqlens
        max_q = int(prep.max_seqlen)
        in_dtype = q.dtype
        q_in, k_in, v_in = q_s, k_fused, v_fused
        if in_dtype not in (torch.float16, torch.bfloat16):
            q_in = q_in.to(torch.bfloat16)
            k_in = k_in.to(torch.bfloat16)
            v_in = v_in.to(torch.bfloat16)
        from .flash_attn_3d_rope import flash_attn_3d_rope_fa3_varlen_func
        out_s = flash_attn_3d_rope_fa3_varlen_func(
            q_in, k_in, v_in, cu_q, fused_cu, max_q, max_k,
            softmax_scale=scale, causal=False,
            coords_q=coords_s, coords_k=coords_fused,
            chunk_dim=cdv, rope_base=rope_base, coord_scale=coord_scale,
        )
        if out_s.dtype != in_dtype:
            out_s = out_s.to(in_dtype)
        return permute_rows(out_s, prep.inv_perm, prep.perm)    # [N,H,E] orig

    # (indexed_kv / _idx_backend / combined_fused decided ABOVE the fa3
    # fused-materialized branch — see the hoisted decision comment there.)
    if prep_kv is prep:
        # Common path: Q and K share the SAME coordinate frame
        # (coords_s_q == coords_s_kv). Fuse into one launch so the per-token
        # sin/cos is computed once, not twice — bit-identical to two separate
        # prerotate_3d_rope calls. Gated on `prep_kv is prep` (identity), NOT on
        # tensor-equality, so the fast path is taken exactly when the two
        # frames are provably the same object.
        if v_gather_on_load:
            # V-gather-on-load (#2): Q gathered+rotated into SORTED order (read
            # contiguously by cu_q); K rotated IN PLACE into UNSORTED order (no
            # gather) so the kernel can read it — alongside raw V — through the
            # perm-composed kv_row_indices below. q_rot is byte-identical to the
            # #1 q_rot; k_rot[m] == #1 k_rot[perm.index_of(m)] (same per-row
            # rotation, original order), so k_rot[composed_idx] == #1
            # k_rot[gather_idx] row-for-row.
            q_rot = prerotate_3d_rope_gathered(
                q, coords, prep.perm, prep.inv_perm, cdv, rope_base, coord_scale)
            k_rot = prerotate_3d_rope(k, coords, cdv, rope_base, coord_scale)
        elif gather_on_load:
            # Gather-on-load (PNT_MSGA_GATHER_ON_LOAD=1): fold the q/k feature
            # gather AND the coords gather INTO the prerotate kernel — no
            # separate q_s/k_s/coords_s index_select launches. perm_q == perm_k
            # == prep.perm here (prep_kv is prep). Output is byte-identical to
            # the permute_rows + coords.index_select + prerotate_3d_rope_qk
            # composition (fwd AND bwd; index_select is a pure row copy and the
            # per-row rotation arithmetic is unchanged).
            q_rot, k_rot = prerotate_3d_rope_qk_gathered(
                q, k, coords, prep.perm, prep_kv.perm,
                prep.inv_perm, prep_kv.inv_perm, cdv, rope_base, coord_scale)
        else:
            coords_s = coords.index_select(0, prep.perm)
            if combined_fused:
                # Combined indexed×fused (fa2 or fa3): the kernel rotates Q/K
                # in-smem from coords_s / the gathered coords_fused — feed RAW
                # q_s/k_s.
                q_rot, k_rot = q_s, k_s
            else:
                q_rot, k_rot = prerotate_3d_rope_qk(
                    q_s, k_s, coords_s, cdv, rope_base, coord_scale)
    else:
        # cyc-28 corruption probe: Q and K gather through different perms ⇒
        # different coordinate frames ⇒ MUST rotate separately (the fused QK
        # path assumes one shared frame and would be wrong here).
        coords_s_q = coords.index_select(0, prep.perm)
        coords_s_kv = coords.index_select(0, prep_kv.perm)
        q_rot = prerotate_3d_rope(q_s, coords_s_q, cdv, rope_base, coord_scale)
        k_rot = prerotate_3d_rope(k_s, coords_s_kv, cdv, rope_base, coord_scale)

    num_octaves = num_active_rings(ring_mult)
    ring_mult_t = _ring_mult_tensor(ring_mult, device)

    if kv_compose == "merged":
        # LSE-merge dual-launch: own launch uses q_rot/k_rot/v_s IN PLACE
        # (no gather), nbr launch uses the budgeted neighbour-only gather;
        # exact joint softmax recovered via the LSE merge (fwd) and the
        # ring-attention decomposition (bwd) in _LseMergedAttention.
        in_dtype = q.dtype
        q_in, k_in, v_in = q_rot, k_rot, v_s
        if in_dtype not in (torch.float16, torch.bfloat16):
            # flash requires fp16/bf16 — grid_sdpa convention for fp32.
            # Convert BEFORE the nbr gather (the gather is an exact copy, so
            # gathering bf16 rows equals converting gathered fp32 rows).
            q_in = q_in.to(torch.bfloat16)
            k_in = k_in.to(torch.bfloat16)
            v_in = v_in.to(torch.bfloat16)
        k_nbr, v_nbr, nbr_only_cu, max_nbr, _ = _build_nbr_only_kv(
            nbr_cell.to(torch.int64), nbr_octave.to(torch.int64), nbr_cu,
            ring_mult_t, num_octaves, prep.cu_seqlens, k_in, v_in,
            num_cells, device,
        )
        out_s = _lse_merged_attention(
            q_in, k_in, v_in, k_nbr, v_nbr,
            prep.cu_seqlens, nbr_only_cu, int(prep.max_seqlen), max_nbr,
            scale,
        )                                                    # [N,H,E] sorted
        if out_s.dtype != in_dtype:
            out_s = out_s.to(in_dtype)
        return permute_rows(out_s, prep.inv_perm, prep.perm)  # [N,H,E] orig

    # (indexed_kv / _idx_backend / combined_fused decided above the
    # prerotate block — see the hoisted decision comment there.)
    if indexed_kv:
        # Value-independent structure — served from the per-ctx cache
        # (one build per stage instead of one per block; see
        # _IDX_GATHER_CACHE).
        # v_s is None on the #2 path (raw V); pass raw v so the (value-
        # independent) index build still gets a same-shaped V handle. The
        # builder reads only k_rot.size(0) and v's requires_grad (the latter
        # short-circuited off under materialize=False), never V values.
        k_fused, fused_cu, max_k = _indexed_gather_cached(
            nbr_cell, nbr_octave, nbr_cu, ring_mult_t, num_octaves,
            prep.cu_seqlens, k_rot, v_s if v_s is not None else v,
            num_cells, device, ring_mult)
        v_fused = None
    else:
        k_fused, v_fused, fused_cu, max_k, _ = _build_fused_kv(
            nbr_cell.to(torch.int64), nbr_octave.to(torch.int64), nbr_cu,
            ring_mult_t, num_octaves, prep.cu_seqlens, k_rot, v_s,
            num_cells, device, materialize=True,
        )

    cu_q = prep.cu_seqlens
    # max_q == max(cu_q diffs) is exactly prep.max_seqlen by the
    # GridAttentionPrep contract (the builder computes it from the same cell
    # sizes) — reuse the host int instead of re-syncing it, mirroring how
    # grid_sdpa already feeds prep.max_seqlen straight to flash.
    max_q = int(prep.max_seqlen)

    # flash_attn requires fp16/bf16 — grid_sdpa convention for fp32.
    in_dtype = q.dtype

    if indexed_kv:
        gather_idx = k_fused                       # int32 [total_kv] row indices
        if v_gather_on_load:
            # #2: compose prep_kv.perm into the fused index so the kernel reads
            # K (unsorted k_rot) and V (RAW v) straight from the original-order
            # buffers. perm[gather_idx] is value-independent ⇒ cached per ctx.
            # k_rot[perm[g]] == sorted-k_rot[g] and v[perm[g]] == v_s[g], so the
            # forward is byte-identical to #1 (same KV rows, same per-cell order).
            gather_idx = _composed_kv_idx_cached(
                gather_idx, prep_kv.perm, nbr_cell, ring_mult, num_octaves)
        # SOURCE buffers, no gather. Combined mode: q_rot/k_rot are the
        # RAW q_s/k_s (rotation moved in-kernel; see the prerotate block).
        # #2 (v_gather_on_load): k_rot is UNSORTED, v_in is RAW v — both read
        # through the perm-composed gather_idx above.
        q_in, k_in, v_in = q_rot, k_rot, (v if v_gather_on_load else v_s)
        if in_dtype not in (torch.float16, torch.bfloat16):
            # Convert the sources, not the gather output — the gather is an
            # exact row copy, so converting before == converting after
            # (same convention as the merged-compose path above).
            q_in = q_in.to(torch.bfloat16)
            k_in = k_in.to(torch.bfloat16)
            v_in = v_in.to(torch.bfloat16)
        # Prerotate-indexed when use_fused_rope is False; combined
        # indexed×fused otherwise — the kernel rotates Q from coords_s and
        # K from the GATHERED coords_fused (the same cheap [total_kv,3]
        # gather the materialized fused path does; rows stay
        # un-materialized). Identical wiring on both backends.
        coords_fused = (coords_s.index_select(0, gather_idx)
                        if combined_fused else None)
        if _idx_backend == "fa2":
            # fa2 indexed lane (hd<64 production backend; explicit
            # PNT_MSGA_INDEXED_KV=1 only).
            from .flash_attn_3d_rope import flash_attn_3d_rope_fa2_varlen_func
            _idx_func = flash_attn_3d_rope_fa2_varlen_func
        else:
            # fa3 indexed lane (auto on sm_90+hd64 for prerotate; combined
            # is explicit PNT_MSGA_INDEXED_KV=1 only until the H200 cell
            # validates — see the hoisted decision comment).
            from .flash_attn_3d_rope import flash_attn_3d_rope_fa3_varlen_func
            _idx_func = flash_attn_3d_rope_fa3_varlen_func
        out_s = _idx_func(
            q_in, k_in, v_in, cu_q, fused_cu, max_q, max_k,
            softmax_scale=scale, causal=False,
            coords_q=coords_s if combined_fused else None,
            coords_k=coords_fused,
            chunk_dim=cdv if combined_fused else None,
            rope_base=rope_base, coord_scale=coord_scale,
            kv_row_indices=gather_idx,
        )                                                        # [N,H,E] sorted
        if out_s.dtype != in_dtype:
            out_s = out_s.to(in_dtype)
        return permute_rows(out_s, prep.inv_perm, prep.perm)     # [N,H,E] orig

    if in_dtype not in (torch.float16, torch.bfloat16):
        q_rot = q_rot.to(torch.bfloat16)
        k_fused = k_fused.to(torch.bfloat16)
        v_fused = v_fused.to(torch.bfloat16)

    out_s = _fa_varlen_attention(
        q_rot, k_fused, v_fused, cu_q, fused_cu, max_q, max_k, scale,
        fa_backend=fa_backend,
    )                                                            # [N,H,E] sorted
    if out_s.dtype != in_dtype:
        out_s = out_s.to(in_dtype)
    return permute_rows(out_s, prep.inv_perm, prep.perm)         # [N,H,E] orig
