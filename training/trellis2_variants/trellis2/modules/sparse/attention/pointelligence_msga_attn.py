"""TRELLIS.2 adapter around the vendored Pointelligence MSGA runtime."""
import math
from typing import Sequence

import torch

from .. import SparseTensor
from trellis2.third_party.pointelligence.layers.grid_shift_builder import GridShiftBuilder
from trellis2.third_party.pointelligence.sparse_engines.multi_scale_grid_attention import (
    build_cell_multiscale_idx,
    multi_scale_grid_attention,
    num_active_rings,
)
from trellis2.third_party.pointelligence.sparse_engines.prerotate_rope_triton import (
    compute_msga_rope_params,
)


def sparse_pointelligence_msga(
    q: SparseTensor,
    k: SparseTensor,
    v: SparseTensor,
    *,
    cell_scaler: float,
    ring_mult: Sequence[float],
    samples_per_ring: int,
    rope_mode: str = "pointelligence",
    num_partitions: int = 1,
    partition_index: int = 0,
    fa_backend: str = "auto",
    stable_coverage: bool = False,
    fixed_shift_fractions=((0.0, 0.0, 0.0), (0.5, 0.5, 0.5)),
    full_ring0: bool = False,
) -> SparseTensor:
    """Run Pointelligence MSGA on TRELLIS sparse tokens.

    The Shape SLat integer coordinates are the voxel-native coordinate system,
    so ``grid_size=1`` and ``cell_scaler`` is measured in latent-grid pitches.
    One or more shifted partitions are built once per SparseTensor and cached;
    each DiT block selects a stable partition by ``partition_index``.  This
    preserves activation-checkpoint recomputation while allowing cyclic shifted
    partitions across blocks.
    """
    if not (q.coords.data_ptr() == k.coords.data_ptr() == v.coords.data_ptr()):
        raise ValueError("MSGA q/k/v must share the same Shape SLat coordinates")
    if rope_mode not in {"pointelligence", "trellis"}:
        raise ValueError(f"Unknown MSGA RoPE mode: {rope_mode}")
    if fa_backend not in {"auto", "fa2", "fa3"}:
        raise ValueError(f"Unknown MSGA FlashAttention backend: {fa_backend}")
    if int(num_partitions) < 1:
        raise ValueError("MSGA num_partitions must be >= 1")

    ring_mult = tuple(float(x) for x in ring_mult)
    num_partitions = int(num_partitions)
    partition_index = int(partition_index) % num_partitions
    n_oct = len(ring_mult) if stable_coverage else num_active_rings(ring_mult)
    cache_name = (
        f"pointelligence_msga_cs{float(cell_scaler):g}"
        f"_oct{n_oct}_spr{int(samples_per_ring)}"
        f"_rope{rope_mode}_parts{num_partitions}"
        f"_stable{int(stable_coverage)}_fullr0{int(full_ring0)}"
    )
    cached = q.get_spatial_cache(cache_name)
    if cached is None:
        coords = q.coords[:, 1:].float()
        sample_inds = q.coords[:, 0].long()
        sample_sizes = torch.bincount(sample_inds, minlength=q.shape[0])
        # Complete each partition before drawing the next shift. This makes
        # partition 0 bit-identical whether one or several partitions are
        # requested, while all shifted partitions remain independent samples.
        partitions = []
        if stable_coverage and len(fixed_shift_fractions) < num_partitions:
            raise ValueError("not enough fixed shifts for requested partitions")
        for partition_id in range(num_partitions):
            shift_fraction = (
                fixed_shift_fractions[partition_id]
                if stable_coverage else None
            )
            prep = GridShiftBuilder(
                cell_scaler=float(cell_scaler),
                num_partitions=1,
            ).build(
                coords, sample_inds, sample_sizes, grid_size=1.0,
                shift_fractions=None if shift_fraction is None else [shift_fraction],
            )[0]
            graph = build_cell_multiscale_idx(
                prep,
                num_octaves=n_oct,
                samples_per_ring=int(samples_per_ring),
                deterministic_coverage=stable_coverage,
                full_ring0=full_ring0,
            )
            partitions.append((prep, graph))
        chunk_dim, rope_base, coord_scale = compute_msga_rope_params(
            q.feats.shape[-1],
            grid_size=1.0,
            cell_scaler=float(cell_scaler),
            num_octaves=n_oct,
        )
        # With TRELLIS RoPE, Q/K have already been rotated in the official
        # coordinate frame.  Zero scale makes the vendored rotation identity.
        if rope_mode == "trellis":
            coord_scale = 0.0
        cached = (partitions, chunk_dim, rope_base, coord_scale)
        q.register_spatial_cache(cache_name, cached)

    partitions, chunk_dim, rope_base, coord_scale = cached
    prep, graph = partitions[partition_index]
    out = multi_scale_grid_attention(
        q.feats,
        k.feats,
        v.feats,
        q.coords[:, 1:].float(),
        prep,
        1.0 / math.sqrt(q.feats.shape[-1]),
        multiscale=graph,
        ring_mult=ring_mult,
        chunk_dim=chunk_dim,
        rope_base=rope_base,
        coord_scale=coord_scale,
        use_fused_rope=False,
        fa_backend=fa_backend,
    )
    return q.replace(out)
