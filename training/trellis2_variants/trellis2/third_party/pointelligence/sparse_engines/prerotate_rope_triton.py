"""Fast 3D-RoPE pre-rotation via Triton.

Pre-rotates Q and K using absolute 3D coordinates so that the standard
(non-RoPE) sparse attention kernel can be used. This is 1.5x faster than
fusing RoPE per-edge inside the attention kernel because:
- Pre-rotation is O(N × H × E) — per point, not per edge
- The attention kernel avoids per-edge sin/cos + shfl overhead
"""

import math
from typing import Tuple

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton


def compute_chunk_dim_per_axis(head_dim: int) -> int:
    """Maximal EVEN per-axis channel count for 3D-RoPE.

    Returns the largest even ``c`` with ``3*c <= head_dim`` — i.e.
    ``c = 2 * (head_dim // 6)`` — so the three spatial axes use as many
    channels as the head allows and the unrotated passthrough shrinks to
    ``head_dim - 3*c == head_dim % 6 ∈ {0, 2, 4}``. Each head's channels
    are partitioned ``[0, c) -> x``, ``[c, 2c) -> y``, ``[2c, 3c) -> z``,
    ``[3c, head_dim) -> unrotated passthrough``; ``c`` is even because RoPE
    rotates real/imag pairs ``(2k, 2k+1)`` (``half_dim = c // 2`` pairs/axis).

    Examples::

        head_dim=16  -> 4    (12 rotated, 4 passthrough)   [unchanged]
        head_dim=18  -> 6    (18 rotated, 0 passthrough)
        head_dim=32  -> 10   (30 rotated, 2 passthrough)
        head_dim=64  -> 20   (60 rotated, 4 passthrough)
        head_dim=128 -> 42   (126 rotated, 2 passthrough)

    History: earlier runs snapped ``c`` DOWN to a power of two (head_dim=32
    -> 8, leaving a full quarter of the head — 8 channels — as dead
    passthrough). That was a Triton-``tl.arange`` vectorization artifact,
    not a modelling choice. The prerotate kernels now pad ``half_dim`` up to
    a power of two and mask the padding, so a non-pow2 ``half_dim`` (e.g. 5
    at head_dim=32) is handled transparently and every usable channel is
    rotated. See attn_extreme cyc-17 / the attn_ablation cyc-1 handoff (Fix 4).
    """
    assert head_dim > 0, f"head_dim must be positive, got {head_dim}"
    out = 2 * (head_dim // 6)
    assert out >= 2, (
        f"head_dim={head_dim} is too small for 3D-RoPE: "
        f"chunk_dim_per_axis would be {out} < 2 (need head_dim >= 6)"
    )
    return out


def compute_rope_params(
    chunk_dim_per_axis: int,
    cell_scaler: float,
    grid_size: float,
    nyquist_factor: float = 2.0,
) -> Tuple[float, float]:
    """Cell-aware ``(rope_base, coord_scale)`` for grid_sdpa cell attention.

    Cell-internal attention only ever computes Q·K between tokens that share
    a GridShiftBuilder cell. The cell has spatial extent
    ``L = cell_scaler * grid_size`` and contains ~``cell_scaler`` voxels per
    axis. Outside-cell distances never appear in any softmax, so the only
    scales the per-axis RoPE band needs to span are:

      * λ_min = nyquist_factor * grid_size  (highest-frequency pair's wavelength)
      * λ_max = L = cell_scaler * grid_size (cell extent — lowest-freq pair)

    The ``nyquist_factor`` controls how aggressive the highest-frequency pair
    is. ``nyquist_factor=2`` matches the literal Nyquist of the voxel grid
    (period = 2 voxels), which sounds principled but in practice puts pair[0]
    in saturation: at PNT's cell_scaler=14, pair[0] does 7 full rotations
    within the cell, and sub-voxel coord noise (RandomJitter, GridSample
    rounding, RandomScale) corrupts the rotation phase across multiple
    periods. Cycle 5b's adaptive-RoPE arm collapsed under this regime at
    S3DIS ep 21. ``nyquist_factor=4`` gives 2× oversampling — pair[0] period
    is 4 voxels and within-cell rotation count drops to cell_scaler/4 = 3.5,
    well below saturation while still giving plenty of fine spatial
    resolution.

    Conceptually: normalize coords by ``L`` so each cell becomes a unit cube.
    Then the per-axis frequency band sweeps wavelengths
    ``nyquist_factor/cell_scaler..1`` in those normalized units. In metres::

        coord_scale = 2π / λ_min  = 2π / (nyquist_factor * grid_size)
        rope_base   = (λ_max / λ_min) ** (half_dim / (half_dim − 1))
                    = (cell_scaler / nyquist_factor) ** (half_dim / (half_dim − 1))

    Note that **rope_base depends only on (cell_scaler, nyquist_factor,
    half_dim)** — constant across all encoder/decoder stages. Only
    coord_scale changes per stage, halving each pooling step (because
    grid_size doubles).

    Compare to the legacy fixed ``base=10000`` and ``coord_scale=1.0`` (i.e.
    coords in metres directly): with PNT's ``head_dim=16`` (so chunk_dim=4,
    half_dim=2), cell_scaler=14, and the now-default nyquist_factor=2, we
    get base = 7² = 49 — orders of magnitude smaller. The legacy 10000 was
    inherited from 1D-RoPE in LLMs and badly mismatches our spatial scales.

    Args:
        chunk_dim_per_axis: per-axis channel count from
            :func:`compute_chunk_dim_per_axis`. Must be even and ≥ 2.
        cell_scaler: number of voxels per cell axis (PNT's GridShiftBuilder
            ``cell_scaler``). Common values 7, 14.
        grid_size: voxel grid size in metres at the stage where this RoPE
            is applied (doubles per pooling step in PNT).
        nyquist_factor: ratio λ_min / grid_size. ``2.0`` (default) matches
            the literal Nyquist of the voxel grid. ``4.0`` is 2x oversampled
            (recommended after cycle 5b's at-Nyquist instability). Must
            satisfy ``cell_scaler > nyquist_factor`` so the band has > 1
            wavelength of dynamic range.

    Returns:
        ``(rope_base, coord_scale)`` for the legacy ``angle = coord *
        coord_scale * base^(-k/half_dim)`` kernel formula.
    """
    if not (chunk_dim_per_axis >= 2 and chunk_dim_per_axis % 2 == 0):
        raise ValueError(
            f"compute_rope_params: chunk_dim_per_axis must be even and >= 2, got {chunk_dim_per_axis}")
    # nyquist_factor sets lambda_min = nyquist_factor * grid_size:
    #   1 = aliasing floor (one-voxel period) — SUB-Nyquist, do not use;
    #   2 = literal Nyquist (two-voxel period, the principled minimum);
    #   4 = 2x oversample of Nyquist (the historical default).
    if not (nyquist_factor >= 1.0):
        raise ValueError(
            f"compute_rope_params: nyquist_factor must be >= 1 (1 = alias floor; "
            f"2 = literal Nyquist; 4 = 2x oversample), got {nyquist_factor}")
    if not (cell_scaler > nyquist_factor):
        raise ValueError(
            f"compute_rope_params: cell_scaler ({cell_scaler}) must be > nyquist_factor "
            f"({nyquist_factor}) for a meaningful band (lambda_max > lambda_min). A "
            f"cell_scaler <= nyquist_factor means the receptive field is smaller than one "
            f"resolvable wavelength — check the cell geometry / ring reach.")
    if not (grid_size > 0):
        raise ValueError(f"compute_rope_params: grid_size must be positive, got {grid_size}")
    half_dim = chunk_dim_per_axis // 2
    coord_scale = 2.0 * math.pi / (float(nyquist_factor) * float(grid_size))
    if half_dim <= 1:
        # Only one frequency band; base is irrelevant. Pick a sane value.
        return 10000.0, float(coord_scale)
    rope_base = (float(cell_scaler) / float(nyquist_factor)) ** (half_dim / (half_dim - 1))
    return float(rope_base), float(coord_scale)


def compute_msga_rope_params(
    head_dim: int,
    grid_size: float,
    cell_scaler: float,
    num_octaves: int,
) -> Tuple[int, float, float]:
    """Fully-determined 3D-RoPE band for MSGA cell attention — **no knobs**.

    Geometry in (``head_dim``, stage ``grid_size``, ``cell_scaler``, ring
    ``num_octaves``), ``(chunk_dim, rope_base, coord_scale)`` out. There is
    nothing to tune: the per-axis pair count comes from ``head_dim`` and the
    two band edges each sit a **2× safety margin from their failure mode**
    (no knobs):

      * **λ_min = 4·grid_size = 2 × the grid-Nyquist wavelength (2·grid).**
        The finest pair must not ALIAS. The grid's Nyquist wavelength is 2
        pitches; sitting the finest pair *on* that edge corrupts its phase
        under sub-grid coordinate noise (RandomJitter / RandomScale /
        GridSample rounding — the cycle-5b at-Nyquist instability). 2×
        oversampling pulls it off the edge. It still resolves a 1-pitch
        displacement (π/2) and stays monotonic out to 2 pitches — a *wider*
        clean window than the at-Nyquist λ_min=2 (which wraps at 2 pitches).
      * **λ_max = 2·field = 2 × the field extent** (``field = cell_scaler ·
        ring_reach`` grid pitches, the largest displacement two tokens that
        share one cell-attention softmax can be apart). The coarsest pair
        must not WRAP: at λ_max = field it sweeps a full 2π over the field
        and the far edge folds back onto self (cos → +1). 2× puts the field
        edge at π (half a cycle, max-distinct).

    Symmetric 2× margins ⇒ ``λ_max/λ_min = field/2`` ⇒ ``base =
    (field/2)^(half_dim/(half_dim−1))``, ``coord_scale = π/(2·grid_size)``.

    ``ring_reach = 2^(num_octaves-1) + 1`` is the max query↔key token
    displacement in cell widths: a ``cell·2^o_max`` centroid hop (octaves are
    DISJOINT shells [r_{o-1}, r_o), r_o = cell·2^o, so the reach is one ball
    of radius cell·2^o_max, NOT a radial stack) plus ½ query-cell + ½
    neighbour-cell. ``num_octaves == 0`` (pure GridAttention) ⇒ ring_reach =
    1 (own cell only). Hoisted here so PNT, the LitePT MSGAAdapter, and the
    operator share ONE band derivation instead of each re-deriving it.

    Returns ``(chunk_dim, rope_base, coord_scale)`` for the
    ``angle = coord · coord_scale · base^(-k/half_dim)`` kernel formula.
    """
    if not (num_octaves >= 0):
        raise ValueError(f"compute_msga_rope_params: num_octaves must be >= 0, got {num_octaves}")
    chunk_dim = compute_chunk_dim_per_axis(head_dim)
    o_max = max(num_octaves - 1, 0)
    ring_reach = (2 ** o_max + 1) if num_octaves >= 1 else 1
    field = float(cell_scaler) * ring_reach              # max in-softmax displacement, grid pitches
    # Both band edges sit a 2× margin from their failure mode, expressed via the
    # low-level primitive: λ_min = 4·grid (= 2× the grid-Nyquist wavelength
    # 2·grid — anti-alias, nyquist_factor=4) and λ_max = 2·field·grid (pass
    # 2·field as the extent — anti-wrap). base spans λ_max/λ_min = field/2.
    rope_base, coord_scale = compute_rope_params(
        chunk_dim, 2.0 * field, grid_size, nyquist_factor=4.0)
    return chunk_dim, float(rope_base), float(coord_scale)


@triton.jit
def _rope_pair_sincos(c0, c1, c2, base, half_dim: tl.constexpr,
                      PAIRS: tl.constexpr, sign):
    """Per-pair (cos, sin) over the FULL row's pair index space.

    Pair ``p`` covers channels ``(2p, 2p+1)``; in the axis-major layout its
    axis is ``p // half_dim`` (0/1/2; ``>= 3`` ⇒ passthrough) and its
    frequency lane is ``j = p % half_dim``. Passthrough pairs get the
    axis-coordinate 0.0, hence angle 0 ⇒ (cos, sin) = (1, 0) — the identity
    rotation — so ONE uniform rotation expression covers rotated and
    passthrough channels with no per-axis branches. ``inv_freq`` and the
    angle multiply are bit-identical to the legacy per-axis kernel (same
    ``exp(-j/half_dim·log(base))`` lanes, same single ``coord·inv_freq``
    product).
    """
    p = tl.arange(0, PAIRS)
    axis = p // half_dim
    j = p % half_dim
    inv_freq = tl.exp(-j.to(tl.float32) / half_dim * tl.math.log(base))  # [PAIRS]
    c_ax = tl.where(axis[None, :] == 0, c0[:, None],
                    tl.where(axis[None, :] == 1, c1[:, None],
                             tl.where(axis[None, :] == 2, c2[:, None], 0.0)))
    ang = c_ax * inv_freq[None, :]                     # [BLOCK_N, PAIRS]
    return tl.math.cos(ang), tl.math.sin(ang) * sign


@triton.jit
def _rope_rotate_row(x_raw, cosv, sinv, rot_elem,
                     BLOCK_N: tl.constexpr, E_PAD: tl.constexpr):
    """Rotate one contiguous ``[BLOCK_N, E_PAD]`` row tile in registers.

    Even/odd (real/imag) extraction is a register-level ``reshape→split``
    of the contiguous load — no strided global accesses — and the rotated
    row is re-interleaved with ``join→reshape`` for one contiguous store.
    Channels ``>= 3·chunk_dim`` (and the ``E_PAD`` tail) take the RAW loaded
    bits via ``rot_elem`` select: a bitwise passthrough, exactly like the
    legacy kernel's masked copy (and immune to the ``−0.0 → +0.0`` washing
    the identity rotation would apply).
    """
    PAIRS: tl.constexpr = E_PAD // 2
    xp = tl.reshape(x_raw.to(tl.float32), (BLOCK_N, PAIRS, 2))
    xr, xi = tl.split(xp)                              # [BLOCK_N, PAIRS] each
    outr = xr * cosv - xi * sinv
    outi = xr * sinv + xi * cosv
    o = tl.reshape(tl.join(outr, outi), (BLOCK_N, E_PAD))
    return tl.where(rot_elem, o.to(x_raw.dtype), x_raw)


@triton.jit
def _prerotate_3d_rope_kernel(
    X, COORDS, OUT,
    N, H: tl.constexpr, E: tl.constexpr,
    chunk_dim: tl.constexpr, base, coord_scale,
    INVERSE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    E_PAD: tl.constexpr,
):
    """3D-RoPE rotation in axis-major / pair-major layout, block-vectorized.

    Channel partition: ``[0, c) -> x``, ``[c, 2c) -> y``, ``[2c, 3c) -> z``,
    ``[3c, E) -> passthrough`` where ``c = chunk_dim`` (per-axis count).
    Rotated channels live in real/imag pairs at offsets ``(2k, 2k+1)``.

    Transaction shape (the v2 rewrite): each (token-block, head) iteration
    issues ONE contiguous ``[BLOCK_N, E_PAD]`` load and ONE contiguous store
    — full 128B sectors. The previous per-axis form issued 6 stride-2
    masked load/store pairs per head (even/imag split in global memory, 50%
    sector use × ``half_dim/PAIR_PAD`` lane masking) plus a separate
    passthrough copy, and ran ~3.1–3.4× off the bandwidth roofline at
    production shapes (cyc-42 R2 reprofile). The even/odd split now happens
    in registers (``_rope_rotate_row``), and the passthrough channels ride
    the same uniform rotation expression with angle 0 (bitwise via the raw
    select). Numerics are bit-identical to the per-axis form: same
    ``inv_freq`` lanes, same fp32 angle product, same sincos, same
    ``xr·cos − xi·sin`` / ``xr·sin + xi·cos``, same dtype round-trip.

    Each program processes ``BLOCK_N`` tokens × **all H heads**: the
    per-token coords and sin/cos are computed ONCE and reused across heads
    (the cycle-19 contract). Inputs are canonicalized to contiguous
    ``[N, H, E]`` / ``[N, 3]`` by the launcher; ``E_PAD = next_pow2(E)``
    spans the row (tail lanes masked; ``E`` must be even — pairs).

    coord_scale is folded into the angle inside the kernel — callers pass
    raw metres, no host-side _scale_coords pre-pass.

    INVERSE=True applies R(-θ) (used for backward gradient transport).
    """
    pid = tl.program_id(0)
    tok = pid * BLOCK_N + tl.arange(0, BLOCK_N)        # [BLOCK_N]
    m = tok < N
    safe = tl.where(m, tok, 0)                         # clamp OOB lanes for addressing

    half_dim: tl.constexpr = chunk_dim // 2
    rot_dim: tl.constexpr = 3 * chunk_dim
    PAIRS: tl.constexpr = E_PAD // 2

    # Coords (raw metres) folded with coord_scale — once per token, reused
    # across all heads. Contiguous [N,3] ⇒ stride 3.
    c0 = tl.load(COORDS + safe * 3 + 0) * coord_scale  # [BLOCK_N]
    c1 = tl.load(COORDS + safe * 3 + 1) * coord_scale
    c2 = tl.load(COORDS + safe * 3 + 2) * coord_scale

    sign = -1.0 if INVERSE else 1.0
    cosv, sinv = _rope_pair_sincos(c0, c1, c2, base, half_dim, PAIRS, sign)

    e = tl.arange(0, E_PAD)
    lm = m[:, None] & (e < E)[None, :]                 # [BLOCK_N, E_PAD]
    rot_elem = (e < rot_dim)[None, :]
    row = safe[:, None] * (H * E)                      # [BLOCK_N, 1] token base
    for h in range(H):
        ptr = row + h * E + e[None, :]
        x_raw = tl.load(X + ptr, mask=lm, other=0.0)
        tl.store(OUT + ptr,
                 _rope_rotate_row(x_raw, cosv, sinv, rot_elem, BLOCK_N, E_PAD),
                 mask=lm)


@triton.jit
def _prerotate_3d_rope_qk_kernel(
    XQ, XK, COORDS, OUTQ, OUTK,
    N, H: tl.constexpr, E: tl.constexpr,
    chunk_dim: tl.constexpr, base, coord_scale,
    INVERSE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    E_PAD: tl.constexpr,
):
    """Fused Q+K 3D-RoPE: identical math to ``_prerotate_3d_rope_kernel`` run
    on both XQ and XK, but the per-token coords and sin/cos transcendentals are
    loaded/computed ONCE and applied to BOTH tensors. Used when Q and K share
    the same coordinate frame (the MSGA production path, ``prep_kv is prep`` ⇒
    ``coords_s_q == coords_s_kv``): two separate launches would recompute the
    identical sincos twice. Bit-identical to two sequential single-tensor
    launches — same fp32 angle/sincos, same per-element rotation, same dtype
    round-trip — because the stores are independent per tensor and the shared
    sincos is exactly what each separate call would have computed.

    Transaction shape matches the v2 single-tensor kernel: per (token-block,
    head) iteration, ONE contiguous load + ONE contiguous store per tensor
    (register-level even/odd split in ``_rope_rotate_row``), replacing the
    per-axis stride-2 masked accesses.
    """
    pid = tl.program_id(0)
    tok = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    m = tok < N
    safe = tl.where(m, tok, 0)

    half_dim: tl.constexpr = chunk_dim // 2
    rot_dim: tl.constexpr = 3 * chunk_dim
    PAIRS: tl.constexpr = E_PAD // 2

    c0 = tl.load(COORDS + safe * 3 + 0) * coord_scale
    c1 = tl.load(COORDS + safe * 3 + 1) * coord_scale
    c2 = tl.load(COORDS + safe * 3 + 2) * coord_scale

    sign = -1.0 if INVERSE else 1.0
    cosv, sinv = _rope_pair_sincos(c0, c1, c2, base, half_dim, PAIRS, sign)

    e = tl.arange(0, E_PAD)
    lm = m[:, None] & (e < E)[None, :]
    rot_elem = (e < rot_dim)[None, :]
    row = safe[:, None] * (H * E)
    for h in range(H):
        ptr = row + h * E + e[None, :]
        q_raw = tl.load(XQ + ptr, mask=lm, other=0.0)
        tl.store(OUTQ + ptr,
                 _rope_rotate_row(q_raw, cosv, sinv, rot_elem, BLOCK_N, E_PAD),
                 mask=lm)
        k_raw = tl.load(XK + ptr, mask=lm, other=0.0)
        tl.store(OUTK + ptr,
                 _rope_rotate_row(k_raw, cosv, sinv, rot_elem, BLOCK_N, E_PAD),
                 mask=lm)


def _check_chunk_dim(chunk_dim: int, head_dim: int) -> None:
    assert 3 * chunk_dim <= head_dim, (
        f"chunk_dim_per_axis={chunk_dim} too large for head_dim={head_dim}: "
        f"3*chunk_dim={3*chunk_dim} > head_dim. The kernel rotates 3 axes "
        f"of chunk_dim channels each; pick chunk_dim ≤ head_dim/3 (e.g. via "
        f"compute_chunk_dim_per_axis(head_dim))."
    )
    assert head_dim % 2 == 0, (
        f"head_dim={head_dim} must be even: the kernel views the row as "
        f"(real, imag) channel pairs. Every transformer head dim in use "
        f"(16/32/64) satisfies this."
    )


# Block size for the token-blocked launch. 16 was the bandwidth-optimal
# point on RTX 5880 Ada (sm_89) across the cycle-19 sweep (16/32/64/128/256);
# small blocks keep occupancy high while the per-token sincos is amortized
# across all H heads in the inner loop.
_PREROTATE_BLOCK_N = 16


def _prerotate_launch(x, coords, chunk_dim, base, coord_scale, inverse):
    """Shared launcher for forward (inverse=False) and inverse (True).

    Canonicalizes x to contiguous ``[N,H,E]`` and coords to contiguous
    ``[N,3]`` so the kernel uses plain row-major addressing — this also
    makes non-contiguous inputs correct (the legacy kernel hardcoded a
    contiguous coords stride and silently corrupted strided coords views).
    """
    N, H, E = x.shape
    _check_chunk_dim(chunk_dim, E)
    x = x.contiguous()
    coords = coords.contiguous()
    out = torch.empty_like(x)
    grid = (triton.cdiv(N, _PREROTATE_BLOCK_N),)
    wrap_triton(_prerotate_3d_rope_kernel)[grid](
        x, coords, out, N, H, E, chunk_dim, base, float(coord_scale),
        INVERSE=inverse, BLOCK_N=_PREROTATE_BLOCK_N,
        E_PAD=triton.next_power_of_2(E),
    )
    return out


def _prerotate_raw(x, coords, chunk_dim, base, coord_scale=1.0):
    """Raw pre-rotation (no autograd)."""
    return _prerotate_launch(x, coords, chunk_dim, base, coord_scale, False)


def _inverse_prerotate_raw(x, coords, chunk_dim, base, coord_scale=1.0):
    """Raw inverse pre-rotation (no autograd) — applies R(-θ)."""
    return _prerotate_launch(x, coords, chunk_dim, base, coord_scale, True)


@triton_op("pnt_prerotate::prerotate_3d_rope", mutates_args={})
def _prerotate_3d_rope_op(
    x: torch.Tensor, coords: torch.Tensor, chunk_dim: int, base: float,
    coord_scale: float,
) -> torch.Tensor:
    """Compile-safe forward pre-rotation R(+θ). A torch.library.triton_op (not
    autograd.Function) so it traces under torch.compile as a registered leaf;
    register_autograd below applies R(−θ) to the gradient. The kernel launches
    via wrap_triton inside _prerotate_launch."""
    return _prerotate_launch(x, coords, chunk_dim, base, coord_scale, False)


@_prerotate_3d_rope_op.register_fake
def _(x, coords, chunk_dim, base, coord_scale):
    # R(θ) is shape-preserving: out is x's shape/dtype, pure function of inputs.
    return torch.empty_like(x)


def _prerotate_3d_rope_setup_context(ctx, inputs, output):
    x, coords, chunk_dim, base, coord_scale = inputs
    ctx.save_for_backward(coords)
    ctx.chunk_dim = chunk_dim
    ctx.base = base
    ctx.coord_scale = coord_scale


def _prerotate_3d_rope_backward(ctx, grad_output):
    (coords,) = ctx.saved_tensors
    # backward transports the gradient by the inverse rotation R(−θ), launched
    # through the same wrap_triton kernel (compile-safe under double-backward-
    # free training).
    grad_x = _inverse_prerotate_raw(
        grad_output, coords, ctx.chunk_dim, ctx.base, ctx.coord_scale)
    return grad_x, None, None, None, None


_prerotate_3d_rope_op.register_autograd(
    _prerotate_3d_rope_backward,
    setup_context=_prerotate_3d_rope_setup_context)


def prerotate_3d_rope(x, coords, chunk_dim=16, base=10000.0, coord_scale=1.0):
    """Pre-rotate x using absolute 3D coordinates for RoPE.

    Autograd-compatible: backward applies inverse rotation to gradients.
    Compile-safe: routes through the ``pnt_prerotate::prerotate_3d_rope``
    triton_op (register_autograd + register_fake), so it traces under
    ``torch.compile`` as a registered leaf instead of graph-breaking.

    Args:
        x: [N, H, E] — Q or K tensor
        coords: [N, 3] — absolute point coordinates (raw metres)
        chunk_dim: channels per spatial axis (default 16)
        base: RoPE frequency base (default 10000.0)
        coord_scale: scalar multiplier folded into the angle inside the
            kernel; replaces an external coords*coord_scale alloc when set
            to a non-1 value (default 1.0).

    Returns:
        x_rotated: [N, H, E] — same shape, RoPE applied
    """
    return _prerotate_3d_rope_op(x, coords, chunk_dim, base, float(coord_scale))


def _prerotate_qk_launch(xq, xk, coords, chunk_dim, base, coord_scale, inverse):
    """Fused Q+K launcher: one kernel, sincos computed once per token, applied
    to both XQ and XK. xq/xk MUST share shape [N,H,E] and the same coords."""
    assert xq.shape == xk.shape, f"q/k shape mismatch: {xq.shape} vs {xk.shape}"
    N, H, E = xq.shape
    _check_chunk_dim(chunk_dim, E)
    xq = xq.contiguous()
    xk = xk.contiguous()
    coords = coords.contiguous()
    outq = torch.empty_like(xq)
    outk = torch.empty_like(xk)
    grid = (triton.cdiv(N, _PREROTATE_BLOCK_N),)
    wrap_triton(_prerotate_3d_rope_qk_kernel)[grid](
        xq, xk, coords, outq, outk, N, H, E, chunk_dim, base, float(coord_scale),
        INVERSE=inverse, BLOCK_N=_PREROTATE_BLOCK_N,
        E_PAD=triton.next_power_of_2(E),
    )
    return outq, outk


@triton_op("pnt_prerotate::prerotate_3d_rope_qk", mutates_args={})
def _prerotate_3d_rope_qk_op(
    xq: torch.Tensor, xk: torch.Tensor, coords: torch.Tensor,
    chunk_dim: int, base: float, coord_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compile-safe fused Q+K pre-rotation. Forward rotates both by R(+θ)
    sharing one sincos pass; register_autograd below transports both grads by
    R(−θ) in one fused pass. Equivalent to applying the single-tensor op to q
    and k separately. A torch.library.triton_op (not autograd.Function) so it
    traces under torch.compile; the kernel launches via wrap_triton inside
    _prerotate_qk_launch."""
    return _prerotate_qk_launch(xq, xk, coords, chunk_dim, base, coord_scale, False)


@_prerotate_3d_rope_qk_op.register_fake
def _(xq, xk, coords, chunk_dim, base, coord_scale):
    # both outputs are shape/dtype-preserving rotations of their inputs.
    return torch.empty_like(xq), torch.empty_like(xk)


def _prerotate_3d_rope_qk_setup_context(ctx, inputs, output):
    xq, xk, coords, chunk_dim, base, coord_scale = inputs
    ctx.save_for_backward(coords)
    ctx.chunk_dim = chunk_dim
    ctx.base = base
    ctx.coord_scale = coord_scale


def _prerotate_3d_rope_qk_backward(ctx, gq, gk):
    (coords,) = ctx.saved_tensors
    dq, dk = _prerotate_qk_launch(
        gq, gk, coords, ctx.chunk_dim, ctx.base, ctx.coord_scale, True)
    return dq, dk, None, None, None, None


_prerotate_3d_rope_qk_op.register_autograd(
    _prerotate_3d_rope_qk_backward,
    setup_context=_prerotate_3d_rope_qk_setup_context)


def prerotate_3d_rope_qk(q, k, coords, chunk_dim=16, base=10000.0, coord_scale=1.0):
    """Pre-rotate Q and K together in ONE Triton launch when they share a
    coordinate frame — computes the per-token sin/cos once instead of twice.

    Bit-identical to ``prerotate_3d_rope(q, ...), prerotate_3d_rope(k, ...)``
    (same fp32 angles, same per-element rotation, same dtype round-trip); the
    only difference is the shared sincos pass + single launch.

    CONTRACT — this fuses on the assumption that Q and K are rotated by the
    SAME coordinate frame. Use it ONLY when q and k positionally align (the
    MSGA production path, where ``coords_q == coords_k``). If Q and K have
    DIFFERENT coords (e.g. the cyc-28 ``prep_kv`` corruption probe, or any
    cross-attention with distinct query/key positions), this entry point is
    WRONG — call ``prerotate_3d_rope`` separately on each. The guards below
    fail loudly on the structurally-detectable misuse modes; the
    same-coords-frame requirement itself is the caller's responsibility and is
    documented at the call site.

    Args:
        q, k: ``[N, H, E]`` — must share identical shape AND dtype.
        coords: ``[N, 3]`` — the SINGLE coordinate frame for both q and k.
        chunk_dim, base, coord_scale: as in ``prerotate_3d_rope``.

    Returns:
        ``(q_rot, k_rot)`` — each ``[N, H, E]``.
    """
    if q.shape != k.shape:
        raise ValueError(
            f"prerotate_3d_rope_qk requires q.shape == k.shape, got "
            f"{tuple(q.shape)} vs {tuple(k.shape)}. For differently-shaped or "
            f"differently-positioned q/k, call prerotate_3d_rope separately.")
    if q.dtype != k.dtype:
        raise ValueError(
            f"prerotate_3d_rope_qk requires q.dtype == k.dtype, got "
            f"{q.dtype} vs {k.dtype}.")
    if q.dim() != 3:
        raise ValueError(
            f"prerotate_3d_rope_qk expects [N, H, E] tensors, got q.dim()={q.dim()}.")
    if coords.dim() != 2 or coords.shape[0] != q.shape[0] or coords.shape[1] != 3:
        raise ValueError(
            f"prerotate_3d_rope_qk expects coords [N, 3] with N == q.shape[0]"
            f"={q.shape[0]}, got {tuple(coords.shape)}.")
    return _prerotate_3d_rope_qk_op(q, k, coords, chunk_dim, base, float(coord_scale))


# ─── Gather-on-load fused Q+K prerotate (cell-sort gather folded in) ───────────
#
# The MSGA production prerotate path does three separate index_select gathers
# (q, k, coords) into cell-sorted order, then runs `_prerotate_3d_rope_qk_kernel`
# on the pre-sorted tensors. Each gather is a full [N,H,E] / [N,3] HBM
# round-trip (~258 MB each for q/k at waymo production shapes). This variant
# folds those gathers INTO the rotation kernel ("gather-on-load"): the kernel
# loads the SOURCE rows q[perm_q[tok]] / k[perm_k[tok]] / coords[perm_q[tok]]
# directly (indexed loads), rotates, and writes q_rot[tok] / k_rot[tok] in
# SORTED order — eliminating the three gather launches and their round-trips.
#
# Byte-identical to permute_rows(q)+permute_rows(k)+coords.index_select +
# prerotate_3d_rope_qk: index_select is a pure bitwise row copy, so loading
# q[perm_q[tok]] gives the exact same bits as q_s[tok]; the fp32 angle, sincos,
# per-element rotation, and dtype round-trip are unchanged, and each output row
# is computed independently (no cross-row reduction whose order could shift).
#
# perm_q drives BOTH q's feature load AND the coords load (Q-side coordinate
# frame); perm_k drives only k's feature load. In the MSGA production path
# perm_q == perm_k == prep.perm; the cyc-28 corruption probe passes
# perm_k = prep_kv.perm != perm_q so they are kept separate.


@triton.jit
def _prerotate_3d_rope_qk_gathered_kernel(
    XQ, XK, COORDS, PERM_Q, PERM_K, OUTQ, OUTK,
    N, H: tl.constexpr, E: tl.constexpr,
    chunk_dim: tl.constexpr, base, coord_scale,
    INVERSE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    E_PAD: tl.constexpr,
):
    """Gather-on-load fused Q+K 3D-RoPE.

    Identical math to ``_prerotate_3d_rope_qk_kernel`` but the feature/coords
    loads are INDEXED by the cell-sort permutations: for output (sorted) token
    ``tok`` the kernel reads q-source row ``PERM_Q[tok]``, k-source row
    ``PERM_K[tok]``, and coords-source row ``PERM_Q[tok]`` (coords share the
    Q-side frame). Stores ``OUTQ[tok] / OUTK[tok]`` in sorted order. This is the
    composition ``permute_rows + prerotate_3d_rope_qk`` collapsed into one
    launch — bit-for-bit, because ``index_select`` is a pure row copy and the
    fp32 angle / sincos / rotation / dtype round-trip below are byte-for-byte
    the per-row arithmetic the unfused kernel runs on the pre-gathered rows.

    INVERSE=True applies R(-θ) — used by the composed backward's unrotate pass
    (the gather-by-inv_perm half is done in the host backward, NOT here, so the
    forward gather and backward scatter stay separable and bit-exact vs the
    composed reference; see the op's register_autograd).
    """
    pid = tl.program_id(0)
    tok = pid * BLOCK_N + tl.arange(0, BLOCK_N)        # [BLOCK_N] sorted-order rows
    m = tok < N
    safe = tl.where(m, tok, 0)                         # clamp OOB lanes for addressing

    half_dim: tl.constexpr = chunk_dim // 2
    rot_dim: tl.constexpr = 3 * chunk_dim
    PAIRS: tl.constexpr = E_PAD // 2

    # Gather-on-load source rows. PERM_* are int64 [N] permutations; clamp the
    # OOB-lane index to a valid source row (its store is masked off anyway).
    src_q = tl.load(PERM_Q + safe)                     # [BLOCK_N] q + coords source
    src_k = tl.load(PERM_K + safe)                     # [BLOCK_N] k source
    src_q = tl.where(m, src_q, 0)
    src_k = tl.where(m, src_k, 0)

    # Coords (raw metres) from the Q-side source row, folded with coord_scale —
    # once per token, reused across all heads. Contiguous [N,3] ⇒ stride 3.
    c0 = tl.load(COORDS + src_q * 3 + 0) * coord_scale  # [BLOCK_N]
    c1 = tl.load(COORDS + src_q * 3 + 1) * coord_scale
    c2 = tl.load(COORDS + src_q * 3 + 2) * coord_scale

    sign = -1.0 if INVERSE else 1.0
    cosv, sinv = _rope_pair_sincos(c0, c1, c2, base, half_dim, PAIRS, sign)

    e = tl.arange(0, E_PAD)
    lm = m[:, None] & (e < E)[None, :]
    rot_elem = (e < rot_dim)[None, :]
    rowq = src_q[:, None] * (H * E)                     # gathered q/k source bases
    rowk = src_k[:, None] * (H * E)
    rowo = safe[:, None] * (H * E)                      # sorted-order store base
    for h in range(H):
        eh = h * E + e[None, :]
        q_raw = tl.load(XQ + rowq + eh, mask=lm, other=0.0)
        tl.store(OUTQ + rowo + eh,
                 _rope_rotate_row(q_raw, cosv, sinv, rot_elem, BLOCK_N, E_PAD),
                 mask=lm)
        k_raw = tl.load(XK + rowk + eh, mask=lm, other=0.0)
        tl.store(OUTK + rowo + eh,
                 _rope_rotate_row(k_raw, cosv, sinv, rot_elem, BLOCK_N, E_PAD),
                 mask=lm)


def _prerotate_qk_gathered_launch(q, k, coords, perm_q, perm_k,
                                  chunk_dim, base, coord_scale):
    """Forward gather-on-load launcher: rotates q[perm_q]/k[perm_k] with
    coords[perm_q] into sorted order. q/k are the UNSORTED source [N,H,E];
    coords is UNSORTED [N,3]; perm_q/perm_k are int64 [N] permutations.
    Canonicalizes sources to contiguous so the indexed loads use plain
    row-major addressing (matching the reference, which .contiguous()-es the
    pre-gathered tensors)."""
    assert q.shape == k.shape, f"q/k shape mismatch: {q.shape} vs {k.shape}"
    N, H, E = q.shape
    _check_chunk_dim(chunk_dim, E)
    q = q.contiguous()
    k = k.contiguous()
    coords = coords.contiguous()
    perm_q = perm_q.contiguous()
    perm_k = perm_k.contiguous()
    outq = torch.empty_like(q)
    outk = torch.empty_like(k)
    grid = (triton.cdiv(N, _PREROTATE_BLOCK_N),)
    wrap_triton(_prerotate_3d_rope_qk_gathered_kernel)[grid](
        q, k, coords, perm_q, perm_k, outq, outk,
        N, H, E, chunk_dim, base, float(coord_scale),
        INVERSE=False, BLOCK_N=_PREROTATE_BLOCK_N,
        E_PAD=triton.next_power_of_2(E),
    )
    return outq, outk


@triton_op("pnt_prerotate::prerotate_3d_rope_qk_gathered", mutates_args={})
def _prerotate_3d_rope_qk_gathered_op(
    q: torch.Tensor, k: torch.Tensor, coords: torch.Tensor,
    perm_q: torch.Tensor, perm_k: torch.Tensor, inv_perm_q: torch.Tensor,
    inv_perm_k: torch.Tensor, chunk_dim: int, base: float, coord_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compile-safe gather-on-load fused Q+K prerotate. Forward gathers
    q[perm_q]/k[perm_k]/coords[perm_q] and rotates by R(+θ) into sorted order in
    ONE launch. register_autograd transports grads by the COMPOSED reverse of
    permute_rows+prerotate: unrotate R(−θ) in sorted order, then scatter back to
    original order via inv_perm (a coalesced gather, the permute_rows backward).
    inv_perm_q/inv_perm_k are saved for that backward gather. A
    torch.library.triton_op (not autograd.Function) so it traces under
    torch.compile; the kernel launches via wrap_triton inside the launcher."""
    return _prerotate_qk_gathered_launch(
        q, k, coords, perm_q, perm_k, chunk_dim, base, coord_scale)


@_prerotate_3d_rope_qk_gathered_op.register_fake
def _(q, k, coords, perm_q, perm_k, inv_perm_q, inv_perm_k,
      chunk_dim, base, coord_scale):
    # Outputs are [N,H,E] in sorted order (N == perm length == q.shape[0]).
    return torch.empty_like(q), torch.empty_like(k)


def _prerotate_3d_rope_qk_gathered_setup_context(ctx, inputs, output):
    (q, k, coords, perm_q, perm_k, inv_perm_q, inv_perm_k,
     chunk_dim, base, coord_scale) = inputs
    # Backward needs the Q-side coords frame (already in original order, indexed
    # by perm_q in the unrotate) and BOTH inverse perms for the scatter-back.
    ctx.save_for_backward(coords, perm_q, perm_k, inv_perm_q, inv_perm_k)
    ctx.chunk_dim = chunk_dim
    ctx.base = base
    ctx.coord_scale = coord_scale


def _prerotate_3d_rope_qk_gathered_backward(ctx, gq, gk):
    """Composed backward — byte-identical to the reference
    permute_rows(prerotate_3d_rope_qk(...)) graph.

    Reference forward graph (per side):  x -> permute_rows -> prerotate(R(+θ)).
    So its backward is:  g_sorted -> prerotate.backward (R(−θ), sorted order)
                                  -> permute_rows.backward (gather by inv_perm).

    The unrotate R(−θ) reuses the SAME gathered kernel with INVERSE=True so the
    coords are read through perm_q exactly as the forward did (the rotation
    angle per sorted row matches), producing the unrotated gradient in SORTED
    order — bit-identical to what `_prerotate_3d_rope_qk_backward` (the unfused
    op) returns for q_s/k_s. Then permute_rows.backward maps sorted→original via
    `grad.index_select(0, inv_perm)` (a coalesced gather; bit-exact for a true
    permutation, 0+g==g). Composition of two bit-exact steps ⇒ bit-exact total.
    """
    (coords, perm_q, perm_k, inv_perm_q, inv_perm_k) = ctx.saved_tensors
    # Unrotate in sorted order (R(−θ)); coords gathered by perm_q (Q frame),
    # q/k-grads are ALREADY in sorted order so they pass through as identity
    # perms — but to keep the kernel's coords-by-perm_q semantics we feed the
    # gradients with an identity gather on the feature side. Simpler + bit-exact:
    # run the plain (non-gathered) inverse QK kernel on the sorted grads with the
    # sorted-order coords, then scatter by inv_perm. coords in sorted order =
    # coords.index_select(0, perm_q).
    coords_s = coords.index_select(0, perm_q)
    dq_s, dk_s = _prerotate_qk_launch(
        gq, gk, coords_s, ctx.chunk_dim, ctx.base, ctx.coord_scale, True)
    # permute_rows backward: sorted -> original via inverse perm (coalesced).
    dq = dq_s.index_select(0, inv_perm_q)
    dk = dk_s.index_select(0, inv_perm_k)
    return dq, dk, None, None, None, None, None, None, None, None


_prerotate_3d_rope_qk_gathered_op.register_autograd(
    _prerotate_3d_rope_qk_gathered_backward,
    setup_context=_prerotate_3d_rope_qk_gathered_setup_context)


def prerotate_3d_rope_qk_gathered(q, k, coords, perm_q, perm_k,
                                  inv_perm_q, inv_perm_k,
                                  chunk_dim=16, base=10000.0, coord_scale=1.0):
    """Gather-on-load fused Q+K prerotate: fold the cell-sort gather of q, k,
    and coords INTO the 3D-RoPE rotation kernel.

    Byte-identical to the composition::

        q_s = permute_rows(q, perm_q, inv_perm_q)
        k_s = permute_rows(k, perm_k, inv_perm_k)
        coords_s = coords.index_select(0, perm_q)
        q_rot, k_rot = prerotate_3d_rope_qk(q_s, k_s, coords_s, ...)

    forward AND backward, while eliminating the three separate index_select
    launches + their HBM round-trips. ``perm_q`` drives q's feature load and the
    coords load (Q-side frame); ``perm_k`` drives k's feature load. In the MSGA
    production path the two perms are identical; the corruption probe passes
    distinct perms.

    Args:
        q, k: ``[N, H, E]`` UNSORTED source tensors (same shape AND dtype).
        coords: ``[N, 3]`` UNSORTED coords (Q-side frame).
        perm_q, perm_k: ``[N]`` int64 cell-sort permutations.
        inv_perm_q, inv_perm_k: ``[N]`` int64 inverses (for the backward gather).
        chunk_dim, base, coord_scale: as in ``prerotate_3d_rope_qk``.

    Returns:
        ``(q_rot, k_rot)`` — each ``[N, H, E]`` in SORTED (cell) order.
    """
    if q.shape != k.shape:
        raise ValueError(
            f"prerotate_3d_rope_qk_gathered requires q.shape == k.shape, got "
            f"{tuple(q.shape)} vs {tuple(k.shape)}.")
    if q.dtype != k.dtype:
        raise ValueError(
            f"prerotate_3d_rope_qk_gathered requires q.dtype == k.dtype, got "
            f"{q.dtype} vs {k.dtype}.")
    if q.dim() != 3:
        raise ValueError(
            f"prerotate_3d_rope_qk_gathered expects [N, H, E] tensors, got "
            f"q.dim()={q.dim()}.")
    N = q.shape[0]
    if coords.dim() != 2 or coords.shape[0] != N or coords.shape[1] != 3:
        raise ValueError(
            f"prerotate_3d_rope_qk_gathered expects coords [N, 3] with N == "
            f"q.shape[0]={N}, got {tuple(coords.shape)}.")
    for nm, p in (("perm_q", perm_q), ("perm_k", perm_k),
                  ("inv_perm_q", inv_perm_q), ("inv_perm_k", inv_perm_k)):
        if p.dim() != 1 or p.shape[0] != N:
            raise ValueError(
                f"prerotate_3d_rope_qk_gathered expects {nm} to be [N] with "
                f"N == {N}, got {tuple(p.shape)}.")
    return _prerotate_3d_rope_qk_gathered_op(
        q, k, coords, perm_q, perm_k, inv_perm_q, inv_perm_k,
        chunk_dim, base, float(coord_scale))


# ---------------------------------------------------------------------------
# Single-tensor gather-on-load prerotate — the Q-side half of the QK gathered
# op. MSGA V-gather-on-load (#2) needs Q rotated into SORTED order (read
# contiguously by cu_q) while K is rotated IN PLACE (unsorted — the kernel reads
# it through a perm-composed kv_row_indices, alongside raw V). The QK gathered op
# emits BOTH sides sorted, so it cannot serve the unsorted-K case; this op folds
# only the Q gather. Byte-identical to permute_rows(x) -> prerotate_3d_rope.
# ---------------------------------------------------------------------------
@triton.jit
def _prerotate_3d_rope_gathered_kernel(
    X, COORDS, PERM, OUT,
    N, H: tl.constexpr, E: tl.constexpr,
    chunk_dim: tl.constexpr, base, coord_scale,
    INVERSE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    E_PAD: tl.constexpr,
):
    """Gather-on-load single-tensor 3D-RoPE: for sorted-order token ``tok`` read
    source row ``PERM[tok]`` of X and COORDS, rotate by R(±θ), store ``OUT[tok]``
    in sorted order. The single-tensor reduction of
    ``_prerotate_3d_rope_qk_gathered_kernel`` (drops the K loads/stores) —
    bit-for-bit ``permute_rows + prerotate_3d_rope`` (index_select is a pure row
    copy; the fp32 angle/sincos/rotation/dtype round-trip is unchanged)."""
    pid = tl.program_id(0)
    tok = pid * BLOCK_N + tl.arange(0, BLOCK_N)        # [BLOCK_N] sorted-order rows
    m = tok < N
    safe = tl.where(m, tok, 0)                         # clamp OOB lanes

    half_dim: tl.constexpr = chunk_dim // 2
    rot_dim: tl.constexpr = 3 * chunk_dim
    PAIRS: tl.constexpr = E_PAD // 2

    src = tl.load(PERM + safe)                         # [BLOCK_N] x + coords source
    src = tl.where(m, src, 0)

    c0 = tl.load(COORDS + src * 3 + 0) * coord_scale   # [BLOCK_N]
    c1 = tl.load(COORDS + src * 3 + 1) * coord_scale
    c2 = tl.load(COORDS + src * 3 + 2) * coord_scale

    sign = -1.0 if INVERSE else 1.0
    cosv, sinv = _rope_pair_sincos(c0, c1, c2, base, half_dim, PAIRS, sign)

    e = tl.arange(0, E_PAD)
    lm = m[:, None] & (e < E)[None, :]
    rot_elem = (e < rot_dim)[None, :]
    rowi = src[:, None] * (H * E)                       # gathered source base
    rowo = safe[:, None] * (H * E)                      # sorted-order store base
    for h in range(H):
        eh = h * E + e[None, :]
        x_raw = tl.load(X + rowi + eh, mask=lm, other=0.0)
        tl.store(OUT + rowo + eh,
                 _rope_rotate_row(x_raw, cosv, sinv, rot_elem, BLOCK_N, E_PAD),
                 mask=lm)


def _prerotate_gathered_launch(x, coords, perm, chunk_dim, base, coord_scale):
    """Forward gather-on-load launcher: rotates ``x[perm]`` with ``coords[perm]``
    into sorted order. ``x`` is UNSORTED ``[N,H,E]``; ``coords`` UNSORTED
    ``[N,3]``; ``perm`` int64 ``[N]``. Canonicalizes sources to contiguous so the
    indexed loads use plain row-major addressing (matching the reference)."""
    N, H, E = x.shape
    _check_chunk_dim(chunk_dim, E)
    x = x.contiguous()
    coords = coords.contiguous()
    perm = perm.contiguous()
    out = torch.empty_like(x)
    grid = (triton.cdiv(N, _PREROTATE_BLOCK_N),)
    wrap_triton(_prerotate_3d_rope_gathered_kernel)[grid](
        x, coords, perm, out,
        N, H, E, chunk_dim, base, float(coord_scale),
        INVERSE=False, BLOCK_N=_PREROTATE_BLOCK_N,
        E_PAD=triton.next_power_of_2(E),
    )
    return out


@triton_op("pnt_prerotate::prerotate_3d_rope_gathered", mutates_args={})
def _prerotate_3d_rope_gathered_op(
    x: torch.Tensor, coords: torch.Tensor, perm: torch.Tensor,
    inv_perm: torch.Tensor, chunk_dim: int, base: float, coord_scale: float,
) -> torch.Tensor:
    """Compile-safe gather-on-load single-tensor prerotate. Forward gathers
    ``x[perm]``/``coords[perm]`` and rotates by R(+θ) into sorted order in ONE
    launch. register_autograd transports the grad by the COMPOSED reverse of
    permute_rows+prerotate: unrotate R(−θ) in sorted order, then scatter to
    original order via ``inv_perm``. A torch.library.triton_op (not
    autograd.Function) so it traces under torch.compile."""
    return _prerotate_gathered_launch(
        x, coords, perm, chunk_dim, base, coord_scale)


@_prerotate_3d_rope_gathered_op.register_fake
def _(x, coords, perm, inv_perm, chunk_dim, base, coord_scale):
    # Output is [N,H,E] in sorted order (N == perm length == x.shape[0]).
    return torch.empty_like(x)


def _prerotate_3d_rope_gathered_setup_context(ctx, inputs, output):
    (x, coords, perm, inv_perm, chunk_dim, base, coord_scale) = inputs
    # Backward needs the (original-order) coords frame + inv_perm for scatter-back.
    ctx.save_for_backward(coords, perm, inv_perm)
    ctx.chunk_dim = chunk_dim
    ctx.base = base
    ctx.coord_scale = coord_scale


def _prerotate_3d_rope_gathered_backward(ctx, g):
    """Composed backward — byte-identical to permute_rows(prerotate_3d_rope(...)).

    Reference forward graph:  x -> permute_rows -> prerotate(R(+θ)).
    So its backward is:  g_sorted -> prerotate.backward (R(−θ), sorted order)
                                  -> permute_rows.backward (gather by inv_perm).
    coords in sorted order = coords.index_select(0, perm) (the same frame the
    forward rotated through). Both steps are bit-exact ⇒ bit-exact total.
    """
    (coords, perm, inv_perm) = ctx.saved_tensors
    coords_s = coords.index_select(0, perm)
    dx_s = _inverse_prerotate_raw(
        g, coords_s, ctx.chunk_dim, ctx.base, ctx.coord_scale)
    dx = dx_s.index_select(0, inv_perm)
    return dx, None, None, None, None, None, None


_prerotate_3d_rope_gathered_op.register_autograd(
    _prerotate_3d_rope_gathered_backward,
    setup_context=_prerotate_3d_rope_gathered_setup_context)


def prerotate_3d_rope_gathered(x, coords, perm, inv_perm,
                               chunk_dim=16, base=10000.0, coord_scale=1.0):
    """Gather-on-load single-tensor prerotate: fold the cell-sort gather of ``x``
    and ``coords`` INTO the 3D-RoPE rotation kernel.

    Byte-identical to the composition::

        x_s = permute_rows(x, perm, inv_perm)
        coords_s = coords.index_select(0, perm)
        x_rot = prerotate_3d_rope(x_s, coords_s, ...)

    forward AND backward, while eliminating the two separate index_select
    launches + their HBM round-trips. The Q-side half of
    ``prerotate_3d_rope_qk_gathered`` — used by MSGA V-gather-on-load (#2), where
    K is rotated IN PLACE (unsorted) instead of gathered.

    Args:
        x: ``[N, H, E]`` UNSORTED source tensor.
        coords: ``[N, 3]`` UNSORTED coords (same frame as x).
        perm: ``[N]`` int64 cell-sort permutation.
        inv_perm: ``[N]`` int64 inverse (for the backward gather).
        chunk_dim, base, coord_scale: as in ``prerotate_3d_rope``.

    Returns:
        ``x_rot`` — ``[N, H, E]`` in SORTED (cell) order.
    """
    if x.dim() != 3:
        raise ValueError(
            f"prerotate_3d_rope_gathered expects [N, H, E], got x.dim()={x.dim()}.")
    N = x.shape[0]
    if coords.dim() != 2 or coords.shape[0] != N or coords.shape[1] != 3:
        raise ValueError(
            f"prerotate_3d_rope_gathered expects coords [N, 3] with N == "
            f"x.shape[0]={N}, got {tuple(coords.shape)}.")
    for nm, p in (("perm", perm), ("inv_perm", inv_perm)):
        if p.dim() != 1 or p.shape[0] != N:
            raise ValueError(
                f"prerotate_3d_rope_gathered expects {nm} to be [N] with "
                f"N == {N}, got {tuple(p.shape)}.")
    return _prerotate_3d_rope_gathered_op(
        x, coords, perm, inv_perm, chunk_dim, base, float(coord_scale))
