"""grid_sdpa: cell-internal sparse self-attention via FlashAttention varlen.

Implements the interface contract documented in
``autoresearch/threads/grid_sdpa/0_expectations/interface_design.md``.

The model side passes raw Q, K, V plus per-token coordinates and a list of
``GridAttentionPrep`` partitions. Each partition is a complete non-overlapping
cover of the N tokens by spatial cells. Inside each partition, attention runs
all-to-all within each cell using ``flash_attn_varlen_qkvpacked_func``.

Three operating modes are all expressed through the same call:

* len(preps) == 1   -- per-layer swin (caller supplies a fresh prep per layer)
                       or stage-amortized non-swin (one prep reused across
                       all blocks in a stage).
* len(preps) >= 2   -- M-fold overlapping partitions evaluated in parallel and
                       aggregated by mean/sum at the output level.

For the stage-amortized non-swin case where multiple blocks in a stage share
a permutation, ``grid_sdpa_with_sorted`` exposes the inner kernel directly:
caller permutes Q,K,V,coords once at stage entry, calls
``grid_sdpa_with_sorted`` per block, un-permutes once at stage exit.

The signature does not expose pre-rotated tensors -- callers always pass raw
Q,K,V plus coords. The current implementation pre-rotates Q,K with the
absolute-coordinate Triton kernel and then calls FlashAttention's varlen
QKV-packed entry point. A future fused-RoPE kernel can drop in behind the
same signature.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence

import torch

from flash_attn import flash_attn_varlen_qkvpacked_func

from .prerotate_rope_triton import prerotate_3d_rope

# chunk_dim values already warned about (fused-inline → Triton fallback), so the
# RuntimeWarning fires once per distinct chunk_dim instead of every forward.
_FUSED_INLINE_FALLBACK_WARNED: set = set()


# ─── Instability telemetry buffer (always-on, low-overhead) ───────────────────
# Accumulates per-forward extrema for the trainer to drain after each iter.
# Drained via `pop_instability_telemetry()`. Values are floats; on rank>0 these
# are still populated (the trainer will only read the rank-0 copy).
#
# Motivation: cycle 5b/5c PNT-Grid runs collapsed at val mIoU after train-loss
# drift that we had no telemetry for. The post-RoPE max-abs is the most direct
# observable for the "Nyquist saturation -> unbounded gradient" hypothesis.
_INSTABILITY: dict[str, float] = {
    "q_post_rope_max_abs": 0.0,
    "k_post_rope_max_abs": 0.0,
    "max_cell_points": 0,
    "rope_nan": 0,            # 1 if any post-rotation tensor was non-finite
}


def pop_instability_telemetry() -> dict[str, float]:
    """Return current per-forward telemetry and reset to zeros for the next forward.

    Should be called once per training iteration (e.g., from the model's
    Segmentor.forward at the very end). Safe to call from any rank.
    """
    out = dict(_INSTABILITY)
    _INSTABILITY["q_post_rope_max_abs"] = 0.0
    _INSTABILITY["k_post_rope_max_abs"] = 0.0
    _INSTABILITY["max_cell_points"] = 0
    _INSTABILITY["rope_nan"] = 0
    return out


def _record_post_rope(q_rot: torch.Tensor, k_rot: torch.Tensor, max_seqlen: int) -> None:
    """Update the global telemetry buffer with this layer's post-RoPE extrema."""
    # detach to avoid building extra autograd nodes; .item() syncs once per layer
    # but the values are scalars and the sync is cheap relative to attention.
    with torch.no_grad():
        q_max = float(q_rot.detach().abs().max().item())
        k_max = float(k_rot.detach().abs().max().item())
        q_nan = bool(torch.logical_not(torch.isfinite(q_rot)).any().item())
        k_nan = bool(torch.logical_not(torch.isfinite(k_rot)).any().item())
    if q_max > _INSTABILITY["q_post_rope_max_abs"]:
        _INSTABILITY["q_post_rope_max_abs"] = q_max
    if k_max > _INSTABILITY["k_post_rope_max_abs"]:
        _INSTABILITY["k_post_rope_max_abs"] = k_max
    if max_seqlen > _INSTABILITY["max_cell_points"]:
        _INSTABILITY["max_cell_points"] = int(max_seqlen)
    if q_nan or k_nan:
        _INSTABILITY["rope_nan"] = 1


@dataclass(frozen=True)
class GridAttentionPrep:
    """One partition. A complete non-overlapping cover of N tokens by cells."""
    perm: torch.Tensor          # [N] int64 -- permutation that sorts tokens by cell_id
    inv_perm: torch.Tensor      # [N] int64 -- inverse of perm
    cu_seqlens: torch.Tensor    # [num_cells + 1] int32 -- cumulative cell sizes
    max_seqlen: int             # largest cell size in this partition
    cell_id: Optional[torch.Tensor] = None  # [N] int64 in original order
    cell_grid: Optional[torch.Tensor] = None   # [num_cells, 3] int32 -- integer
                                # grid coords of each cell (shifted-grid units)
    cell_batch: Optional[torch.Tensor] = None  # [num_cells] int64 -- batch id


# ─── Permutation helpers (exposed for stage-amortized callers) ────────────────

def permute_to_sorted(x: torch.Tensor, prep: GridAttentionPrep) -> torch.Tensor:
    """Reorder x[N, ...] from original to cell-sorted order."""
    return x.index_select(0, prep.perm)


def permute_from_sorted(x_sorted: torch.Tensor, prep: GridAttentionPrep) -> torch.Tensor:
    """Reorder x[N, ...] from cell-sorted back to original order."""
    return x_sorted.index_select(0, prep.inv_perm)


@torch.library.custom_op("tf1_pnt_grid_sdpa::permute_rows", mutates_args=())
def permute_rows(x: torch.Tensor, perm: torch.Tensor,
                 inv_perm: torch.Tensor) -> torch.Tensor:
    """Permutation-aware ``x.index_select(0, perm)``: bit-identical forward and
    backward, with the backward's atomic ``index_add_`` replaced by a gather
    through ``inv_perm``. ``perm``/``inv_perm`` must be mutually inverse
    permutations (true for any ``GridAttentionPrep`` pair by construction).

    Autograd lowers index_select's backward to ``new_zeros().index_add_(0,
    perm, grad)`` — the atomic-RMW ``indexFuncLargeIndex`` scatter — even when
    ``perm`` is a permutation. For a true permutation that backward is a pure
    copy: each output row is written exactly once and ``0 + g == g`` bitwise,
    so ``grad.index_select(0, inv_perm)`` produces a byte-identical gradient
    while swapping the atomic scatter for a coalesced gather (also removing a
    nondeterministic-order fp reduction from the backward graph).

    A torch.library.custom_op (not autograd.Function) so it traces under
    torch.compile as an opaque-but-registered leaf instead of graph-breaking."""
    return x.index_select(0, perm)


@permute_rows.register_fake
def _permute_rows_fake(x, perm, inv_perm):
    # output shape is a pure function of input shapes (perm length × x[1:]) —
    # no data-dependence, so the meta kernel is exact.
    return x.new_empty((perm.shape[0],) + tuple(x.shape[1:]))


def _permute_rows_setup_context(ctx, inputs, output):
    _x, _perm, inv_perm = inputs
    ctx.save_for_backward(inv_perm)


def _permute_rows_backward(ctx, grad):
    (inv_perm,) = ctx.saved_tensors
    return grad.index_select(0, inv_perm), None, None


permute_rows.register_autograd(
    _permute_rows_backward, setup_context=_permute_rows_setup_context)


# ─── Inner kernel: assumes Q,K,V,coords already in sorted order ───────────────

def grid_sdpa_with_sorted(
    q_sorted: torch.Tensor,        # [N, H, E]
    k_sorted: torch.Tensor,        # [N, H, E]
    v_sorted: torch.Tensor,        # [N, H, E]
    coords_sorted: torch.Tensor,   # [N, 3] fp32
    cu_seqlens: torch.Tensor,      # [num_cells + 1] int32
    max_seqlen: int,
    scale: float,
    *,
    chunk_dim: Optional[int] = None,
    rope_base: float = 10000.0,
    coord_scale: float = 1.0,
    use_fused_rope: bool = False,
    use_rotary_tables: bool = False,
    fa_backend: Literal["auto", "fa2", "fa3"] = "auto",
    enable_telemetry: bool = True,
    pre_rotated: bool = False,
) -> torch.Tensor:
    """Run cell-internal attention on already-sorted Q, K, V, coords.

    Returns output in the same sorted order (caller un-permutes).

    Two paths:
    - ``use_fused_rope=False`` (default): pre-rotate Q,K via the Triton kernel,
      pack QKV, call upstream `flash_attn_varlen_qkvpacked_func`. Production
      path; backward parity verified end-to-end.
    - ``use_fused_rope=True``: pack raw Q,K,V, dispatch to the FA-3 vendored
      fork with ``coords`` — kernel rotates Q,K in smem during forward,
      hybrid backward applies inverse rotation to dQ,dK. Lever B path.

    flash_attn requires fp16 / bf16 — if the input is fp32 we cast to bf16
    for the kernel call and back to the input dtype on output.

    ``enable_telemetry``: when True (default), records per-forward post-RoPE
    max-abs into the global ``_INSTABILITY`` buffer for trainer-side
    instability monitoring. Calls ``.item()`` on max-abs reductions which
    forces a device-host sync — **incompatible with cuda-graph capture**.
    Pass ``enable_telemetry=False`` from cuda-graph contexts (training
    loops that wrap forward in ``torch.cuda.CUDAGraph``). Only affects
    the unfused path; Lever B path doesn't run telemetry.
    """
    assert q_sorted.dim() == 3 and k_sorted.dim() == 3 and v_sorted.dim() == 3
    assert q_sorted.shape == k_sorted.shape == v_sorted.shape

    # Auto-derive corrected chunk_dim_per_axis when caller passes None.
    # All production paths (PNT_Segmentor → GridAttention) explicitly pick
    # the right value; this default is for standalone / bench callers.
    if chunk_dim is None:
        from .prerotate_rope_triton import compute_chunk_dim_per_axis
        head_dim = q_sorted.shape[-1]
        chunk_dim = compute_chunk_dim_per_axis(head_dim)

    # The FA-3 vendored fork's INLINE-coords rotary kernel requires the rotary
    # span (3*chunk_dim) to be a multiple of 16 (flash_api.cpp host check). The
    # maximal-even chunk_dim often isn't — e.g. head_dim=64 -> 20 -> 60 -> 60%16
    # != 0 — so fusing inline would raise on sm_90. Fall back to the
    # Triton-prerotate path (correctness-equivalent, the production reference)
    # whenever inline fusion would violate it. The rotary-TABLE variant pads to
    # %16 internally, so it stays exempt. Done unconditionally (not arch-gated)
    # so the model behaves identically on sm_89 and sm_90; the fused kernel is a
    # perf optimization, never a correctness requirement.
    # (Removed 2026-06-11) The (3*chunk_dim) % 16 inline fallback is obsolete:
    # the FA-3 fork now identity-pads the rotary tail in-kernel (channels in
    # [3*chunk_dim, rotary_dim_padded) get cos=1/sin=0), so the max-channels
    # split (chunk_dim = 2*(head_dim//6), e.g. 20 at head_dim=64) runs the
    # fused inline path directly.

    # cu_seqlens must be int32 contiguous on the same device.
    cu_seqlens = cu_seqlens.to(torch.int32)

    if pre_rotated and use_fused_rope:
        raise ValueError("Already rotated Q/K cannot use fused RoPE")
    if use_fused_rope:
        # Lever B path: pack raw Q,K,V; let the fused kernel rotate.
        qkv = torch.stack([q_sorted, k_sorted, v_sorted], dim=1)
        in_dtype = qkv.dtype
        if in_dtype not in (torch.float16, torch.bfloat16):
            qkv_kernel = qkv.to(torch.bfloat16)
        else:
            qkv_kernel = qkv

        from .flash_attn_3d_rope import flash_attn_3d_rope_varlen_qkvpacked_func
        out_sorted = flash_attn_3d_rope_varlen_qkvpacked_func(
            qkv_kernel,
            cu_seqlens,
            max_seqlen,
            softmax_scale=scale,
            causal=False,
            fa_backend=fa_backend,
            coords=coords_sorted,
            chunk_dim=chunk_dim,
            rope_base=rope_base,
            use_rotary_tables=use_rotary_tables,
            coord_scale=coord_scale,
        )
    else:
        # Production path: pre-rotate Q, K with absolute coordinates.
        # coord_scale is folded into the angle inside the Triton kernel —
        # no host-side coords*coord_scale alloc.
        q_rot = q_sorted if pre_rotated else prerotate_3d_rope(q_sorted, coords_sorted, chunk_dim, rope_base, coord_scale)
        k_rot = k_sorted if pre_rotated else prerotate_3d_rope(k_sorted, coords_sorted, chunk_dim, rope_base, coord_scale)

        # Instability telemetry: post-RoPE saturation indicators.
        # Disabled under cuda-graph capture (`.item()` calls would crash
        # with cudaErrorStreamCaptureUnsupported).
        if enable_telemetry:
            _record_post_rope(q_rot, k_rot, max_seqlen)

        qkv = torch.stack([q_rot, k_rot, v_sorted], dim=1)
        in_dtype = qkv.dtype
        if in_dtype not in (torch.float16, torch.bfloat16):
            qkv_kernel = qkv.to(torch.bfloat16)
        else:
            qkv_kernel = qkv

        out_sorted = flash_attn_varlen_qkvpacked_func(
            qkv_kernel,
            cu_seqlens,
            max_seqlen=max_seqlen,
            dropout_p=0.0,
            softmax_scale=scale,
        )

    if out_sorted.dtype != in_dtype:
        out_sorted = out_sorted.to(in_dtype)

    return out_sorted


# ─── Top-level entry: original-order Q,K,V plus list of preps ────────────────

def grid_sdpa(
    q: torch.Tensor,                       # [N, H, E]
    k: torch.Tensor,                       # [N, H, E]
    v: torch.Tensor,                       # [N, H, E]
    coords: torch.Tensor,                  # [N, 3] fp32
    preps: Sequence[GridAttentionPrep],
    scale: float,
    *,
    aggregate: Literal["mean", "sum"] = "mean",
    chunk_dim: Optional[int] = None,
    rope_base: float = 10000.0,
    coord_scale: float = 1.0,
    use_fused_rope: bool = False,
    use_rotary_tables: bool = False,
    fa_backend: Literal["auto", "fa2", "fa3"] = "auto",
    enable_telemetry: bool = True,
    pre_rotated: bool = False,
) -> torch.Tensor:                          # [N, H, E] in original order
    """Sparse self-attention over geometric grid cells.

    With one prep: cell-internal all-to-all attention on a single partition.
    Equivalent to the legacy sparse_sdpa for shifted_grid_v2 graphs. Can be
    called per-layer with a fresh prep to implement swin.

    With M >= 2 preps: each partition is evaluated independently and the
    outputs are aggregated by the caller-selected reduction (mean / sum).
    Each token participates in M cells, giving cross-cell information flow
    within a single layer (no swin needed).

    Implementation: pre-rotate Q,K via Triton, pack QKV, call
    flash_attn_varlen_qkvpacked_func per partition. A future fused-RoPE kernel
    can replace this body without changing the signature.
    """
    assert len(preps) >= 1, "grid_sdpa requires at least one partition"
    assert aggregate in ("mean", "sum"), f"unknown aggregate '{aggregate}'"

    # Per-partition path: permute -> grid_sdpa_with_sorted -> un-permute.
    # Routing through grid_sdpa_with_sorted guarantees bitwise equivalence
    # with the stage-amortized path (correctness contract item 4).
    outs: List[torch.Tensor] = []
    for prep in preps:
        q_s = q.index_select(0, prep.perm)
        k_s = k.index_select(0, prep.perm)
        v_s = v.index_select(0, prep.perm)
        coords_s = coords.index_select(0, prep.perm)

        out_s = grid_sdpa_with_sorted(
            q_s, k_s, v_s, coords_s,
            prep.cu_seqlens, prep.max_seqlen, scale,
            chunk_dim=chunk_dim, rope_base=rope_base,
            coord_scale=coord_scale,
            use_fused_rope=use_fused_rope,
            use_rotary_tables=use_rotary_tables,
            fa_backend=fa_backend,
            enable_telemetry=enable_telemetry,
            pre_rotated=pre_rotated,
        )
        outs.append(out_s.index_select(0, prep.inv_perm))

    if len(outs) == 1:
        return outs[0]

    stacked = torch.stack(outs, dim=0)
    if aggregate == "mean":
        return stacked.mean(dim=0)
    return stacked.sum(dim=0)
