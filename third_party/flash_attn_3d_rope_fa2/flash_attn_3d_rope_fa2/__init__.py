"""flash_attn_3d_rope_fa2 — FA-2 vendored fork (flash_attn 2.8.3 base) with
3D-RoPE-fused varlen FlashAttention. Sibling package: flash_attn_3d_rope_fa3.

Phase 0 status: no-op pass-through. The kernel is currently bit-identical to
upstream `flash_attn_varlen_qkvpacked_func`. Phase 1+ adds 3D-RoPE fusion
behind the same API.

Public API:
    flash_attn_3d_rope_fa2_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen,
                                                  softmax_scale, ...)
"""
from .varlen import flash_attn_3d_rope_fa2_varlen_qkvpacked_func

__version__ = "0.1.0.dev0"
__all__ = ["flash_attn_3d_rope_fa2_varlen_qkvpacked_func"]
