"""GridShiftBuilder: geometric grid partition emitter for grid_sdpa.

Replaces the CSR-emitting ``ShiftedGridEdgeBuilder`` in the grid_sdpa kernel
path. The first half of ``shifted_grid_v2`` (random shift -> cell hash ->
sort tokens by cell hash -> find cell boundaries) computes exactly the
metadata that ``GridAttentionPrep`` needs. The CSR step that the legacy
builder performs after that is wasted work for grid_sdpa, which consumes
``cu_seqlens`` directly.

For ``num_partitions > 1`` the builder draws ``num_partitions`` independent
random shifts in a single ``build()`` call. Producing the M partitions
together avoids redundant work in the surrounding model code; the cell-hash
math itself is per-partition (independent random shifts give independent
hashes), so sharing inside a single call is just a Python/launch-overhead
saving, not a kernel-level one.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import torch
from torch import Tensor

from ..sparse_engines.grid_sdpa import GridAttentionPrep


# Spatial primes used to fold (cx, cy, cz, batch) into a single int64 hash.
# Identical to ``ShiftedGridEdgeBuilder``'s primes so the cell partition is
# bit-for-bit comparable across builders for a given (shift, points).
_P1 = 73856093
_P2 = 19349663
_P3 = 83492791
_P_BATCH = 2654435761


class GridShiftBuilder:
    """Emit ``GridAttentionPrep`` partitions from a shifted-grid hash.

    Args:
        cell_scaler: Multiplier on ``grid_size`` to determine cell width
            ``w = grid_size * cell_scaler``. Larger -> denser attention,
            broader receptive field per cell.
        num_partitions: Number of independent random shifts (M-fold overlap).
            ``1`` matches the legacy single-partition path.
    """

    is_stochastic: bool = True

    def __init__(
        self,
        cell_scaler: float = 14.0,
        num_partitions: int = 1,
    ):
        if num_partitions < 1:
            raise ValueError(f"num_partitions must be >= 1, got {num_partitions}")
        self.cell_scaler = cell_scaler
        self.num_partitions = num_partitions

    @torch.compiler.disable
    def build(
        self,
        points: Tensor,         # [N, 3] fp32
        sample_inds: Tensor,    # [N] int64 -- batch element id per point
        sample_sizes: Tensor,   # [B] int64 -- points per batch element
        grid_size: float,
        shift_fractions: Optional[Sequence[Sequence[float]]] = None,
        **kwargs,
    ) -> List[GridAttentionPrep]:
        """Return ``num_partitions`` independent partitions of the points."""
        N = points.size(0)
        device = points.device
        w = grid_size * self.cell_scaler
        B = sample_sizes.numel()

        if shift_fractions is not None and len(shift_fractions) != self.num_partitions:
            raise ValueError(
                "shift_fractions must have one (x,y,z) entry per partition"
            )
        preps: List[GridAttentionPrep] = []
        for partition_index in range(self.num_partitions):
            shift_fraction = (
                None if shift_fractions is None
                else shift_fractions[partition_index]
            )
            preps.append(self._build_one(
                points, sample_inds, B, N, w, device,
                shift_fraction=shift_fraction,
            ))
        return preps

    def _build_one(
        self,
        points: Tensor,
        sample_inds: Tensor,
        B: int,
        N: int,
        w: float,
        device: torch.device,
        shift_fraction: Optional[Sequence[float]] = None,
    ) -> GridAttentionPrep:
        # ── 1. Random shift per batch element ──
        if shift_fraction is None:
            shifts = torch.rand(B, 3, device=device) * w       # (B, 3)
        else:
            if len(shift_fraction) != 3:
                raise ValueError("each shift fraction must have three values")
            frac = torch.as_tensor(
                shift_fraction, device=device, dtype=points.dtype
            )
            if not bool(torch.all((frac >= 0) & (frac < 1))):
                raise ValueError("fixed shift fractions must lie in [0, 1)")
            shifts = (frac * w).view(1, 3).expand(B, 3)
        per_point_shift = shifts[sample_inds]                  # (N, 3)
        shifted_points = points + per_point_shift              # (N, 3)

        # ── 2. Hash into grid cells ──
        cell_coords = torch.floor(shifted_points / w).int()    # (N, 3)
        cx = cell_coords[:, 0].long()
        cy = cell_coords[:, 1].long()
        cz = cell_coords[:, 2].long()
        cell_hash = (
            sample_inds.long() * _P_BATCH
            + cx * _P1 + cy * _P2 + cz * _P3
        )

        # ── 3. Sort points by cell hash to group co-cell points ──
        sorted_hash, sort_idx = torch.sort(cell_hash)
        sort_idx_l = sort_idx.long()

        # cell_id in original order: sorted position -> cell index.
        # We want: cell_id[orig_i] = which cell does orig_i belong to.
        # Step a: assign a dense cell index in sorted order.
        boundary = torch.ones(N, dtype=torch.bool, device=device)
        boundary[1:] = sorted_hash[1:] != sorted_hash[:-1]
        sorted_cell_id = torch.cumsum(boundary.to(torch.int64), dim=0) - 1  # [N]
        # Step b: scatter back to original order.
        cell_id_orig = torch.empty(N, dtype=torch.int64, device=device)
        cell_id_orig[sort_idx_l] = sorted_cell_id

        # ── 4. Cell boundaries -> cu_seqlens ──
        cell_starts = torch.nonzero(boundary, as_tuple=False).squeeze(1)  # [num_cells]
        num_cells = cell_starts.numel()
        cell_ends = torch.empty(num_cells, dtype=torch.int64, device=device)
        if num_cells > 1:
            cell_ends[:-1] = cell_starts[1:]
        cell_ends[-1] = N
        cell_sizes = cell_ends - cell_starts

        max_seqlen = int(cell_sizes.max().item()) if num_cells > 0 else 0

        # cu_seqlens[i+1] = end-of-cell-i in sorted order. Length = num_cells + 1.
        cu_seqlens = torch.empty(num_cells + 1, dtype=torch.int32, device=device)
        cu_seqlens[0] = 0
        cu_seqlens[1:] = cell_ends.to(torch.int32)

        # perm: original index -> position in sorted order
        # Convention: sorted = original[perm]. With sort_idx returned by torch.sort,
        # sorted_hash = cell_hash[sort_idx], so perm = sort_idx (long).
        perm = sort_idx_l
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(N, device=device, dtype=torch.int64)

        # Per-cell metadata for the ring builder: integer grid coord + batch
        # id of each cell = those of its first sorted token (all tokens in a
        # cell share both). Two tiny gathers — no float work.
        first_orig = sort_idx_l.index_select(0, cell_starts)       # [C]
        cell_grid = cell_coords.index_select(0, first_orig)        # [C,3] int32
        cell_batch = sample_inds.index_select(0, first_orig).long()  # [C]

        return GridAttentionPrep(
            perm=perm,
            inv_perm=inv_perm,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            cell_id=cell_id_orig,
            cell_grid=cell_grid,
            cell_batch=cell_batch,
        )
