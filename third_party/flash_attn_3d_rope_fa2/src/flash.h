/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include "namespace_config.h"

#include <cuda.h>
#include <vector>

#include <ATen/cuda/CUDAGeneratorImpl.h> // For at::Generator and at::PhiloxCudaState

namespace FLASH_NAMESPACE {
constexpr int TOTAL_DIM = 0;
constexpr int H_DIM = 1;
constexpr int D_DIM = 2;

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Qkv_params {
    using index_t = int64_t;
    // The QKV matrices.
    void *__restrict__ q_ptr;
    void *__restrict__ k_ptr;
    void *__restrict__ v_ptr;

    // The stride between rows of the Q, K and V matrices.
    index_t q_batch_stride;
    index_t k_batch_stride;
    index_t v_batch_stride;
    index_t q_row_stride;
    index_t k_row_stride;
    index_t v_row_stride;
    index_t q_head_stride;
    index_t k_head_stride;
    index_t v_head_stride;

    // The number of heads.
    int h, h_k;
    // In the case of multi-query and grouped-query attention (MQA/GQA), nheads_k could be
    // different from nheads (query).
    int h_h_k_ratio; // precompute h / h_k,
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Flash_fwd_params : public Qkv_params {

    // The O matrix (output).
    void * __restrict__ o_ptr;
    void * __restrict__ oaccum_ptr;

    // The stride between rows of O.
    index_t o_batch_stride;
    index_t o_row_stride;
    index_t o_head_stride;

    // The pointer to the P matrix.
    void * __restrict__ p_ptr;

    // The pointer to the softmax sum.
    void * __restrict__ softmax_lse_ptr;
    void * __restrict__ softmax_lseaccum_ptr;

    // The dimensions.
    int b, seqlen_q, seqlen_k, seqlen_knew, d, seqlen_q_rounded, seqlen_k_rounded, d_rounded, rotary_dim, total_q;

    // The scaling factors for the kernel.
    float scale_softmax;
    float scale_softmax_log2;

    // array of length b+1 holding starting offset of each sequence.
    int * __restrict__ cu_seqlens_q;
    int * __restrict__ cu_seqlens_k;
    int * __restrict__ leftpad_k;

    // If provided, the actual length of each k sequence.
    int * __restrict__ seqused_k;

    int *__restrict__ blockmask;

    // The K_new and V_new matrices.
    void * __restrict__ knew_ptr;
    void * __restrict__ vnew_ptr;

    // The stride between rows of the Q, K and V matrices.
    index_t knew_batch_stride;
    index_t vnew_batch_stride;
    index_t knew_row_stride;
    index_t vnew_row_stride;
    index_t knew_head_stride;
    index_t vnew_head_stride;

    // The cos and sin matrices for rotary embedding.
    void * __restrict__ rotary_cos_ptr;
    void * __restrict__ rotary_sin_ptr;

    // ─── Lever B: 3D-RoPE fused rotation ───────────────────────────────────
    // When coords_3d_ptr is non-null, the inner attention kernel applies
    // 3D rotary position embedding to Q and K in-register before the GEMM.
    // Coords are per-token absolute (x, y, z) world-space positions.
    //
    //   coords_3d_ptr: fp32 [total_q, 3] — same layout as our existing
    //                  `prerotate_3d_rope` Triton kernel input.
    //   chunk_dim_3d:  channels per spatial axis (default 16). Channel
    //                  index e -> axis = e // chunk_dim_3d,
    //                  pair_idx = (e % chunk_dim_3d) // 2.
    //                  Channels with axis >= 3 are NOT rotated (rope_active
    //                  mask).
    //   rope_base_3d:  RoPE frequency base (default 10000.0).
    //
    // Math forward (per channel e of token n, for both Q and K):
    //   inv_freq = exp(-pair_idx / (chunk_dim_3d/2) * log(rope_base_3d))
    //   angle    = coords_3d_ptr[n, axis] * inv_freq
    //   partner  = x[n, h, e ^ 1]
    //   if is_even: x_rot = x*cos(angle) - partner*sin(angle)
    //   else:       x_rot = x*cos(angle) + partner*sin(angle)
    //
    // Backward (applied to dQ and dK at kernel epilogue, per channel):
    //   if is_even: dx_rot = dx*cos(angle) + partner_dx*sin(angle)
    //   else:       dx_rot = dx*cos(angle) - partner_dx*sin(angle)
    //
    // Pyref oracle:
    // tests/unittest/refs/grid_sdpa_pyref.py::absolute_rope_pyref
    void * __restrict__ coords_3d_ptr;
    int chunk_dim_3d;
    float rope_base_3d;
    float coord_scale_3d;

    // ─── MSGA combined indexed×fused: separate K-side coords ───────────────
    // For the unpacked mismatched-cu varlen shape (MSGA: total_k != total_q,
    // K rows are a gathered+duplicated token set) the K rotation needs its
    // own positional coords table, fp32 [total_k, 3] — one row per PACKED
    // varlen K position r (NOT per source row; under indexed-KV the gather
    // index resolves the row, the rotation stays positional). nullptr →
    // fall back to coords_3d_ptr (the packed qkv path where Q and K share
    // one token set; byte-identical legacy behavior).
    void * __restrict__ coords_3d_k_ptr;

    // ─── MSGA indexed fused-KV (Stage A, FA-2 port) ─────────────────────────
    // When non-null: [total_k] int32 row indices, CSR'd by cu_seqlens_k like
    // the packed K rows they replace. k_ptr/v_ptr are then SOURCE buffers
    // [N_src, h_k, d] and the row the kernel would have read at packed
    // position r materializes from k/v[kv_row_indices[r]] inside the gmem
    // loads. Bwd additionally scatters dK/dV through the same index into
    // caller-zeroed fp32 SOURCE-shaped accumulators (see Flash_bwd_params).
    // nullptr (default — set_params_fprop zero-inits) → byte-identical
    // non-indexed behavior.
    int * __restrict__ kv_row_indices;

    // The indices to index into the KV cache.
    int * __restrict__ cache_batch_idx;

    // Paged KV cache
    int * __restrict__ block_table;
    index_t block_table_batch_stride;
    int page_block_size;

    // The dropout probability (probability of keeping an activation).
    float p_dropout;
    // uint32_t p_dropout_in_uint;
    // uint16_t p_dropout_in_uint16_t;
    uint8_t p_dropout_in_uint8_t;

    // Scale factor of 1 / (1 - p_dropout).
    float rp_dropout;
    float scale_softmax_rp_dropout;

    // Local window size
    int window_size_left, window_size_right;
    float softcap;

    // Random state.
    at::PhiloxCudaState philox_args;

    // Pointer to the RNG seed (idx 0) and offset (idx 1).
    uint64_t * rng_state;

    bool is_bf16;
    bool is_causal;

    // If is_seqlens_k_cumulative, then seqlen_k is cu_seqlens_k[bidb + 1] - cu_seqlens_k[bidb].
    // Otherwise it's cu_seqlens_k[bidb], i.e., we use cu_seqlens_k to store the sequence lengths of K.
    bool is_seqlens_k_cumulative;

    bool is_rotary_interleaved;

    int num_splits;  // For split-KV version

    void * __restrict__ alibi_slopes_ptr;
    index_t alibi_slopes_batch_stride;

    bool unpadded_lse;  // For varlen paths: LSE is in [nheads, total_seqlen_q] format instead of [b, nheads, seqlen_q].
    bool seqlenq_ngroups_swapped;  // q has been transposed from (b, 1, (nheads_kv ngroups), d) to (b, ngroups, nheads_kv, d).
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Flash_bwd_params : public Flash_fwd_params {

    // The dO and dQKV matrices.
    void *__restrict__ do_ptr;
    void *__restrict__ dq_ptr;
    void *__restrict__ dk_ptr;
    void *__restrict__ dv_ptr;

    // To accumulate dQ
    void *__restrict__ dq_accum_ptr;
    void *__restrict__ dk_accum_ptr;
    void *__restrict__ dv_accum_ptr;

    // // To accumulate dK and dV in case we're splitting the bwd along seqlen_q
    // dimension void *__restrict__ dk_accum_ptr; void *__restrict__
    // dv_accum_ptr;

    // The stride between rows of the dO, dQ, dK and dV matrices.
    // TD [2022-04-16]: We're using 32-bit indexing to save registers.
    // The code probably won't work for arrays larger than 2GB.
    index_t do_batch_stride;
    index_t do_row_stride;
    index_t do_head_stride;
    index_t dq_batch_stride;
    index_t dk_batch_stride;
    index_t dv_batch_stride;
    index_t dq_row_stride;
    index_t dk_row_stride;
    index_t dv_row_stride;
    index_t dq_head_stride;
    index_t dk_head_stride;
    index_t dv_head_stride;

    // The pointer to the softmax d sum.
    void *__restrict__ dsoftmax_sum;

    bool deterministic;
    index_t dq_accum_split_stride;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, int Headdim, bool Is_causal> void run_mha_fwd_(Flash_fwd_params &params, cudaStream_t stream);
template<typename T, int Headdim, bool Is_causal> void run_mha_fwd_splitkv_dispatch(Flash_fwd_params &params, cudaStream_t stream);

template<typename T, int Headdim, bool Is_causal> void run_mha_bwd_(Flash_bwd_params &params, cudaStream_t stream);

}  // namespace FLASH_NAMESPACE
