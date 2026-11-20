"""Varlen forward+backward wrapper for the FA-2 vendored fork.

Uses ``torch.library.custom_op`` + ``register_autograd`` + ``register_fake``
so the op is an opaque graph node under ``torch.compile`` (no Dynamo
graph break). FA-2's C++ ops are pybind11-bound (``m.def``, not
``TORCH_LIBRARY``), so we wrap the pybind11 calls inside a Python
``custom_op``. From Dynamo's perspective the op is opaque; inside the
op body the pybind11 calls run normally.

Lever B (3D-RoPE training fusion in FA-2 fwd + bwd):
- Forward: when ``coords`` is provided, the wrapper passes the raw (unrotated)
  Q,K plus the rotary spec (cos/sin tables in the table variant, or the coords
  pointer + chunk_dim/rope_base/coord_scale in the default inline variant) to
  the FA-2 ``varlen_fwd`` op; the kernel rotates Q,K in shared memory before
  the QK GEMM consumes them.
- Backward (Phase B.3 + Stage 4, fully in-kernel — no Triton calls): the
  wrapper passes raw Q,K + the same rotary spec to ``varlen_bwd``. The kernel
  rotates Q,K in smem before the SdP GEMM and inverse-rotates dQ,dK in the dK
  epilogue / dQ postprocess (R(-θ) in registers from per-element coords). This
  replaced the earlier hybrid (Triton prerotate → rotary-unaware FA-2 bwd →
  Triton inverse_prerotate); R(-θ) commutes with the atomic-add accumulation
  in dQ since rotation is linear, so both are mathematically equivalent.

When ``coords`` is None: bit-identical pass-through to upstream FA-2.
"""
from typing import Optional, Tuple

import torch

# The compiled CUDA extension module name from setup.py:PACKAGE_NAME + "_C"
import flash_attn_3d_rope_fa2_C as _ext  # noqa: F401


def _maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


# ───────────────────────────────────────────────────────────────────────
# Custom op: packed-QKV varlen fwd with optional 3D-RoPE (table variant).
#   Inputs : qkv, cu_seqlens, scalar attrs, optional coords + RoPE params.
#   Outputs: (out, softmax_lse, rng_state). softmax_lse + rng_state are
#            needed by FA-2 backward and are returned as op outputs.
# ───────────────────────────────────────────────────────────────────────
@torch.library.custom_op(
    "flash_attn_3d_rope_fa2_py::varlen_qkvpacked_fwd",
    mutates_args=(),
    device_types="cuda",
)
def varlen_qkvpacked_fwd(
    qkv: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    return_attn_probs: bool,
    coords: Optional[torch.Tensor],
    chunk_dim: int,
    rope_base: float,
    coord_scale: float,
    use_rotary_tables: bool,
    kv_row_indices: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    qkv = _maybe_contiguous(qkv)

    rotary_cos = None
    rotary_sin = None
    coords_arg = None
    if coords is not None:
        head_dim = qkv.shape[-1]
        max_rotary_dim = min(head_dim, 3 * chunk_dim)
        rotary_dim = (max_rotary_dim // chunk_dim) * chunk_dim
        assert rotary_dim >= chunk_dim, \
            f"head_dim={head_dim} < chunk_dim={chunk_dim}; need at least one full axis"
        assert coords.dim() == 2 and coords.size(0) == qkv.size(0) and coords.size(1) == 3, \
            f"coords must be [{qkv.size(0)}, 3], got {coords.shape}"
        if use_rotary_tables:
            # Table variant: build cos/sin once, kernel reads gmem.
            from sparse_engines.build_3d_rope_tables import build_3d_rope_cos_sin_tables
            # `pad_to=16` ensures flash_api.cpp's % 16 == 0 host check passes.
            rotary_cos, rotary_sin, _padded_rotary_dim = build_3d_rope_cos_sin_tables(
                coords.float().contiguous(), rotary_dim,
                chunk_dim=chunk_dim, base=rope_base,
                coord_scale=coord_scale,
                dtype=qkv.dtype,
                pad_to=16,
            )
        else:
            # Inline variant: pass coords + scalars; kernel computes cos/sin
            # in registers per attention block (no gmem cos/sin traffic).
            coords_arg = coords.float().contiguous()

    out, softmax_lse, _S_dmask, rng_state = _ext.varlen_fwd(
        qkv[:, 0],                      # q
        qkv[:, 1],                      # k
        qkv[:, 2],                      # v
        None,                           # out (let kernel allocate)
        cu_seqlens,                     # cu_seqlens_q
        cu_seqlens,                     # cu_seqlens_k (same for self-attention)
        None,                           # seqused_k
        None,                           # leftpad_k
        None,                           # block_table
        None,                           # alibi_slopes
        max_seqlen,                     # max_seqlen_q
        max_seqlen,                     # max_seqlen_k
        dropout_p,
        softmax_scale,
        False,                          # zero_tensors
        causal,
        -1,                             # window_size_left
        -1,                             # window_size_right
        0.0,                            # softcap
        return_attn_probs,
        None,                           # gen_
        rotary_cos,                     # Lever B table variant: cos/sin from gmem
        rotary_sin,
        True if coords is not None else False,  # is_rotary_interleaved
        coords_arg,                     # Lever B inline variant: coords for in-register cos/sin
        chunk_dim,
        rope_base,
        coord_scale,
        None,                           # coords_3d_k — packed self-attention: Q/K share one
                                        # token set, the kernel falls back to coords_3d
        kv_row_indices,                 # MSGA indexed fused-KV — LAST arg in the C++ schema
    )
    return out, softmax_lse, rng_state


@varlen_qkvpacked_fwd.register_fake
def _varlen_qkvpacked_fwd_fake(
    qkv, cu_seqlens, max_seqlen, dropout_p, softmax_scale, causal,
    return_attn_probs, coords, chunk_dim, rope_base, coord_scale,
    use_rotary_tables, kv_row_indices=None,
):
    total_q, _three, num_heads, head_dim = qkv.shape
    out = qkv.new_empty((total_q, num_heads, head_dim))
    # FA-2 varlen softmax_lse layout: [num_heads, total_q] fp32.
    softmax_lse = qkv.new_empty((num_heads, total_q), dtype=torch.float32)
    rng_state = torch.empty(2, dtype=torch.int64, device=qkv.device)
    return out, softmax_lse, rng_state


def _setup_context(ctx, inputs, output):
    (qkv, cu_seqlens, max_seqlen, dropout_p, softmax_scale, causal,
     return_attn_probs, coords, chunk_dim, rope_base, coord_scale,
     use_rotary_tables, kv_row_indices) = inputs
    out, softmax_lse, rng_state = output
    ctx.save_for_backward(qkv, out, softmax_lse, cu_seqlens, rng_state, coords,
                          kv_row_indices)
    ctx.max_seqlen = max_seqlen
    ctx.dropout_p = dropout_p
    ctx.softmax_scale = softmax_scale
    ctx.causal = causal
    ctx.chunk_dim = chunk_dim
    ctx.rope_base = rope_base
    ctx.coord_scale = coord_scale
    ctx.use_rotary_tables = use_rotary_tables
    ctx.had_rotary = coords is not None


def _backward(ctx, grad_out, grad_softmax_lse, grad_rng_state):
    (qkv, out, softmax_lse, cu_seqlens, rng_state, coords,
     kv_row_indices) = ctx.saved_tensors

    # Lever B Phase B.3 + Stage 4 (FA-2): the bwd kernel rotates Q,K in
    # smem before the SdP gemm and inverse-rotates dQ,dK in the dK
    # epilogue / dQ postprocess. Wrapper passes raw qkv + rotary params
    # to the op. No Triton calls.
    rotary_cos = None
    rotary_sin = None
    coords_arg = None
    if ctx.had_rotary:
        head_dim = qkv.shape[-1]
        max_rotary_dim = min(head_dim, 3 * ctx.chunk_dim)
        rotary_dim = (max_rotary_dim // ctx.chunk_dim) * ctx.chunk_dim
        if ctx.use_rotary_tables:
            from sparse_engines.build_3d_rope_tables import build_3d_rope_cos_sin_tables
            rotary_cos, rotary_sin, _padded = build_3d_rope_cos_sin_tables(
                coords.float().contiguous(), rotary_dim,
                chunk_dim=ctx.chunk_dim, base=ctx.rope_base,
                coord_scale=ctx.coord_scale,
                dtype=qkv.dtype,
                pad_to=16,
            )
        else:
            coords_arg = coords.float().contiguous()

    dqkv = torch.empty_like(qkv)
    dq, dk, dv = dqkv[:, 0], dqkv[:, 1], dqkv[:, 2]
    indexed = kv_row_indices is not None
    if indexed:
        # MSGA indexed fused-KV: the bwd kernel scatters dK/dV through the
        # index into CALLER-ZEROED fp32 SOURCE-shaped accumulators (it never
        # zeroes them itself); downcast into the dqkv views after the call.
        dk = torch.zeros(qkv[:, 1].shape, dtype=torch.float32,
                         device=qkv.device)
        dv = torch.zeros(qkv[:, 2].shape, dtype=torch.float32,
                         device=qkv.device)

    _ext.varlen_bwd(
        _maybe_contiguous(grad_out),    # dout
        qkv[:, 0],                      # q (raw)
        qkv[:, 1],                      # k (raw)
        qkv[:, 2],                      # v
        out,                            # out
        softmax_lse,                    # softmax_lse
        dq, dk, dv,
        cu_seqlens, cu_seqlens,
        None,                           # alibi_slopes
        ctx.max_seqlen, ctx.max_seqlen,
        ctx.dropout_p, ctx.softmax_scale,
        False,                          # zero_tensors
        ctx.causal,
        -1, -1, 0.0,                    # window left/right, softcap
        False,                          # deterministic
        None,                           # gen_
        rng_state,
        rotary_cos,                     # Lever B table variant
        rotary_sin,
        True if ctx.had_rotary else False,  # is_rotary_interleaved
        coords_arg,                     # Lever B inline variant
        ctx.chunk_dim if ctx.had_rotary else 0,
        ctx.rope_base,
        ctx.coord_scale,
        None,                           # coords_3d_k — packed: falls back to coords_3d
        kv_row_indices,                 # MSGA indexed fused-KV — LAST arg in the C++ bwd schema
    )

    # Stage 4 v2: kernel applies R(-θ) to dQ/dK in registers (before the
    # smem write) using thr_mma.partition_C-derived per-element coords.
    # Replaces the smem-side approach that had a partition mismatch.
    # Wrapper is now Triton-free for rotary in bwd.

    if indexed:
        # Sequential cast-and-free (peak-memory discipline — the fp32
        # accumulators are the indexed path's main bwd transient).
        dqkv[:, 1].copy_(dk.to(qkv.dtype))
        del dk
        dqkv[:, 2].copy_(dv.to(qkv.dtype))
        del dv

    # dq,dk,dv are views into the contiguous dqkv [N,3,H,hd] that varlen_bwd
    # filled in place (indexed mode: dk/dv copied in just above), so dqkv IS
    # torch.stack([dq,dk,dv], dim=1) bit-for-bit. Return it directly — the
    # former torch.stack re-copied the full gradient ([N,3,H,hd]) into a
    # fresh buffer for no reason (~14% of the backward at production scale,
    # cycle-18 NCU: a [N,3,H,hd] CatArrayBatchedCopy).
    # Op has 13 inputs; only qkv is differentiable (coords are positions,
    # kv_row_indices is an int index → None).
    return (dqkv, None, None, None, None, None, None, None, None, None, None,
            None, None)


varlen_qkvpacked_fwd.register_autograd(_backward, setup_context=_setup_context)


# ───────────────────────────────────────────────────────────────────────
# User-facing wrapper. Drops softmax_lse + rng_state intermediates that
# exist only for backward.
# ───────────────────────────────────────────────────────────────────────
def flash_attn_3d_rope_fa2_varlen_qkvpacked_func(
    qkv: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    return_attn_probs: bool = False,
    coords: Optional[torch.Tensor] = None,
    chunk_dim: int = 16,
    rope_base: float = 10000.0,
    coord_scale: float = 1.0,
    use_rotary_tables: bool = False,
    kv_row_indices: Optional[torch.Tensor] = None,
):
    """FA-2 drop-in with optional in-kernel 3D-RoPE in two variants.

    ``coords=None``: bit-identical pass-through to upstream (no rotary).

    ``coords`` provided + ``use_rotary_tables=True``: build cos/sin
    tables in gmem (via ``build_3d_rope_cos_sin_tables``); kernel reads
    them during the in-kernel rotation. Higher gmem bandwidth pressure
    but simpler kernel arithmetic.

    ``coords`` provided + ``use_rotary_tables=False`` (default): no
    tables — kernel takes the coords pointer + (chunk_dim, rope_base,
    coord_scale) and computes cos/sin per attention block in registers
    via ``apply_rotary_3d_inline_interleaved``. Per cycle 5p Lever C
    analysis, this is the bandwidth-bound win at small rotary_dim
    (e.g. hd=32 / chunk_dim=8 / rotary_dim=24 → only 12 fp16 cos values
    per token).

    Backward (both variants): fully in-kernel (Phase B.3 + Stage 4) — the
    FA-2 bwd kernel rotates Q,K in smem before the SdP GEMM and inverse-rotates
    dQ,dK in the dK epilogue / dQ postprocess. No Triton calls. Mathematically
    equivalent to the earlier Triton-hybrid (R(-θ) commutes with atomic-add
    accumulation since rotation is linear).

    Args:
        qkv: [total_q, 3, num_heads, head_dim] in cell-sorted order.
        cu_seqlens: [num_cells + 1] int32 cumulative cell sizes.
        max_seqlen: int — largest cell size.
        coords: optional [total_q, 3] fp32 per-token absolute coords.
        chunk_dim: channels per spatial axis (default 16).
        rope_base: RoPE frequency base (default 10000.0).
        coord_scale: scalar multiplier folded into cos/sin (default 1.0).
        use_rotary_tables: select table variant (True) or inline variant
            (False, default).
        kv_row_indices: optional [total_q] int32 — MSGA indexed fused-KV.
            The kernel reads K/V row ``kv_row_indices[r]`` of the packed
            ``qkv[:, 1] / qkv[:, 2]`` for each varlen position ``r`` (the
            ``index_select`` materialization moves into the gmem loads);
            bwd scatters dK/dV back through the same index (fp32 atomicAdd
            accumulators handled internally). ``None`` (default) is
            byte-identical legacy behavior. Composes with ``coords``
            (the combined indexed×fused mode — rotation stays positional).
    """
    if softmax_scale is None:
        softmax_scale = qkv.shape[-1] ** -0.5
    out, _softmax_lse, _rng_state = varlen_qkvpacked_fwd(
        qkv, cu_seqlens, max_seqlen, dropout_p, softmax_scale, causal,
        return_attn_probs, coords, chunk_dim, rope_base, coord_scale,
        use_rotary_tables, kv_row_indices,
    )
    return out
