/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cute/tensor.hpp>

#include "namespace_config.h"
#include "utils.h"

// Forward declaration for libdevice precise sincos. nvcc bundles
// libdevice (.bc) and links calls to __nv_sincosf to its precise (~1 ULP)
// implementation, regardless of --use_fast_math. The qualified name avoids
// the unqualified `sincosf` → `__sincosf` rewrite that --use_fast_math
// otherwise applies. Equivalent to Triton's `tl.math.cos/sin` precision.
extern "C" __device__ void __nv_sincosf(float, float*, float*);

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace FLASH_NAMESPACE {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool Is_even_K=true, bool Clear_OOB_K=true, bool Inverse=false,
          typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2, typename Engine3, typename Layout3>
__forceinline__ __device__ void copy_rotary_interleaved(Tensor<Engine0, Layout0> const &S,
                                               Tensor<Engine1, Layout1> &D,
                                               Tensor<Engine2, Layout2> const &Cos,
                                               Tensor<Engine2, Layout2> const &Sin,
                                               Tensor<Engine3, Layout3> const &identity_MN,
                                               const int max_MN, const int min_MN,
                                               const int dim, const int rotary_dim) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(Cos));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(Cos));                     // MMA_K
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(Sin));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(Sin));                     // MMA_K
    CUTE_STATIC_ASSERT_V(size<0>(Cos) == size<0>(Sin));                     // MMA_K
    static_assert(decltype(size<0>(S))::value == decltype(size<0>(Cos))::value * 2);
    static_assert(decltype(size<0>(Cos))::value % 2 == 0);  // Since we do fast conversion from fp16/bf16 to fp32
    Tensor rCos = make_fragment_like(Cos);
    Tensor rSin = make_fragment_like(Sin);
    Tensor rS = make_fragment_like(S);
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        if (get<0>(identity_MN(0, m, 0)) >= min_MN && get<0>(identity_MN(0, m, 0)) < max_MN) {
            #pragma unroll
            for (int k = 0; k < size<2>(S); ++k) {
                if (Is_even_K || get<1>(identity_MN(0, 0, k)) < dim) {
                    cute::copy(S(_, m, k), rS(_, m, k));
                    if (get<1>(identity_MN(0, 0, k)) < rotary_dim) {
                        cute::copy(Cos(_, m, k), rCos(_, m, k));
                        cute::copy(Sin(_, m, k), rSin(_, m, k));
                        Tensor S_fp32 = convert_type<float>(rS(_, m, k));
                        Tensor cos_fp32 = convert_type<float>(rCos(_, m, k));
                        Tensor sin_fp32 = convert_type<float>(rSin(_, m, k));
                        #pragma unroll
                        for (int i = 0; i < size<0>(rS) / 2; ++i) {
                            // Inverse rotation: R(-θ) = [cos, sin; -sin, cos] vs [cos, -sin; sin, cos].
                            // Equivalent to negating sin in the same formula.
                            float const sin_i = Inverse ? -sin_fp32(i) : sin_fp32(i);
                            float real = S_fp32(2 * i) * cos_fp32(i) - S_fp32(2 * i + 1) * sin_i;
                            float imag = S_fp32(2 * i) * sin_i + S_fp32(2 * i + 1) * cos_fp32(i);
                            S_fp32(2 * i) = real;
                            S_fp32(2 * i + 1) = imag;
                        }
                        // Idk but I need to copy for the convert_type to work
                        Tensor S_fp32_copy = make_fragment_like(S_fp32);
                        cute::copy(S_fp32, S_fp32_copy);
                        using T = typename Engine0::value_type;
                        Tensor S_og_type = convert_type<T>(S_fp32_copy);
                        cute::copy(S_og_type, rS(_, m, k));
                    }
                    cute::copy(rS(_, m, k), D(_, m, k));
                } else if (Clear_OOB_K) {
                    cute::clear(D(_, m, k));
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool Is_even_K=true, bool Clear_OOB_K=true,
          typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2, typename Engine3, typename Layout3>
__forceinline__ __device__ void copy_rotary_contiguous(Tensor<Engine0, Layout0> const &S,
                                              Tensor<Engine1, Layout1> &D,
                                              Tensor<Engine2, Layout2> const &Cos,
                                              Tensor<Engine2, Layout2> const &Sin,
                                              Tensor<Engine3, Layout3> const &identity_MN,
                                              const int max_MN, const int min_MN,
                                              const int dim, const int rotary_dim) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(Cos));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(Cos));                     // MMA_K
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(Sin));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(Sin));                     // MMA_K
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(Cos));                     // MMA
    CUTE_STATIC_ASSERT_V(size<0>(Cos) == size<0>(Sin));
    static_assert(decltype(size<0>(Cos))::value % 2 == 0);  // Since we do fast conversion from fp16/bf16 to fp32
    Tensor rCos = make_fragment_like(Cos);
    Tensor rSin = make_fragment_like(Sin);
    Tensor rS = make_fragment_like(S);
    Tensor rS_other = make_fragment_like(rS(_, 0, 0));
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        if (get<0>(identity_MN(0, m, 0)) >= min_MN && get<0>(identity_MN(0, m, 0)) < max_MN) {
            #pragma unroll
            for (int k = 0; k < size<2>(S); ++k) {
                if (Is_even_K || get<1>(identity_MN(0, 0, k)) < dim) {
                    cute::copy(S(_, m, k), rS(_, m, k));
                    if (get<1>(identity_MN(0, 0, k)) < rotary_dim) {
                        const bool is_left = get<1>(identity_MN(0, 0, k)) < rotary_dim / 2;
                        Tensor gS_other = make_tensor(S(_, m, k).data() + (is_left ? rotary_dim / 2 : -rotary_dim / 2), S(_, m, k).layout());
                        cute::copy(gS_other, rS_other);
                        // if (cute::thread0()) { print_tensor(rS(_, m, k)); print_tensor(rS_other); }
                        Tensor gCos = make_tensor(Cos(_, m, k).data() + (is_left ? 0 : -rotary_dim / 2), Cos(_, m, k).layout());
                        Tensor gSin = make_tensor(Sin(_, m, k).data() + (is_left ? 0 : -rotary_dim / 2), Sin(_, m, k).layout());
                        cute::copy(gCos, rCos(_, m, k));
                        cute::copy(gSin, rSin(_, m, k));
                        // if (cute::thread0()) { print_tensor(rCos(_, m, k)); print_tensor(rSin(_, m, k)); }
                        Tensor S_fp32 = convert_type<float>(rS(_, m, k));
                        Tensor S_other_fp32 = convert_type<float>(rS_other);
                        Tensor cos_fp32 = convert_type<float>(rCos(_, m, k));
                        Tensor sin_fp32 = convert_type<float>(rSin(_, m, k));
                        #pragma unroll
                        for (int i = 0; i < size<0>(rS); ++i) {
                            S_fp32(i) = S_fp32(i) * cos_fp32(i) + S_other_fp32(i) * (is_left ? -sin_fp32(i) : sin_fp32(i));
                        }
                        // Idk but I need to copy for the convert_type to work
                        Tensor S_fp32_copy = make_fragment_like(S_fp32);
                        cute::copy(S_fp32, S_fp32_copy);
                        using T = typename Engine0::value_type;
                        Tensor S_og_type = convert_type<T>(S_fp32_copy);
                        cute::copy(S_og_type, rS(_, m, k));
                        // if (cute::thread0()) { print_tensor(rS(_, m, k)); }
                    }
                    cute::copy(rS(_, m, k), D(_, m, k));
                } else if (Clear_OOB_K) {
                    cute::clear(D(_, m, k));
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Lever B no-tables variant: 3D-RoPE rotation with cos/sin computed
// inline in registers from per-token coords. Mirrors
// copy_rotary_interleaved's interface and semantics but replaces
// (Cos, Sin) gmem tensors with (coords_ptr, chunk_dim, base,
// coord_scale). Per cycle 5p Lever C analysis: at small rotary_dim the
// gmem traffic of cos/sin tables dominates the cost; computing inline
// in registers from a 3-fp32-per-token coord load is cheaper.
//
// 3D-RoPE channel layout: [0, c) -> x, [c, 2c) -> y, [2c, 3c) -> z,
// [3c, head_dim) -> passthrough where c = chunk_dim. Pairs at offsets
// (2k, 2k+1) within each axis chunk share inv_freq[k] = exp(-k/half_dim
// * log(base)). Identical formula to prerotate_3d_rope_kernel.
template <bool Is_even_K=true, bool Clear_OOB_K=true, bool Inverse=false,
          typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine3, typename Layout3, typename CoordPtr>
__forceinline__ __device__ void apply_rotary_3d_inline_interleaved(
    Tensor<Engine0, Layout0> const &S,
    Tensor<Engine1, Layout1> &D,
    Tensor<Engine3, Layout3> const &identity_MN,
    CoordPtr coords_ptr,            // float* — [N, 3] fp32 gmem
    const int row_base_global,      // global token index of S[m=0]
    const int max_MN, const int min_MN,
    const int dim, const int rotary_dim,
    const int chunk_dim, const float base, const float coord_scale) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K
    static_assert(decltype(size<0>(S))::value % 2 == 0,
                  "interleaved rotary requires even MMA dim");
    Tensor rS = make_fragment_like(S);

    const int half_dim = chunk_dim >> 1;
    const float log_base = logf(base);

    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        const int row_local = get<0>(identity_MN(0, m, 0));
        if (row_local >= min_MN && row_local < max_MN) {
            const int seq_pos = row_base_global + row_local;
            // Load this row's 3 coords once. coord_scale is folded here.
            const float c0 = coords_ptr[seq_pos * 3 + 0] * coord_scale;
            const float c1 = coords_ptr[seq_pos * 3 + 1] * coord_scale;
            const float c2 = coords_ptr[seq_pos * 3 + 2] * coord_scale;
            #pragma unroll
            for (int k = 0; k < size<2>(S); ++k) {
                if (Is_even_K || get<1>(identity_MN(0, 0, k)) < dim) {
                    cute::copy(S(_, m, k), rS(_, m, k));
                    if (get<1>(identity_MN(0, 0, k)) < rotary_dim) {
                        Tensor S_fp32 = convert_type<float>(rS(_, m, k));
                        // Each MMA "fragment" element of S(_, m, k) is one
                        // contiguous head_dim chunk per thread; we walk
                        // pairs (2i, 2i+1) and compute their cos/sin from
                        // (axis, pair_idx) of the global column index.
                        const int col_base = get<1>(identity_MN(0, 0, k));
                        #pragma unroll
                        for (int i = 0; i < size<0>(rS) / 2; ++i) {
                            const int col = col_base + 2 * i;  // global head_dim col of the (real, imag) pair
                            // Per-pair bound check: at hd=16 / chunk_dim=4 the
                            // per-thread fragment may straddle the rotary_dim
                            // boundary (e.g. cols 8, 10 in [0, rotary_dim=12)
                            // share a fragment with cols 12, 14 in [12, head_dim=16)).
                            // Skip rotation for the passthrough cols.
                            if (col >= rotary_dim) { continue; }
                            const int axis = col / chunk_dim;
                            const int pair_idx = (col % chunk_dim) >> 1;
                            const float coord = (axis == 0 ? c0 : axis == 1 ? c1 : c2);
                            // Iter 4: explicit libdevice precise __nv_sincosf — gives
                            // ~1 ULP precision (matches Triton's tl.math.cos/sin) at
                            // single-precision speed. Bypasses --use_fast_math which
                            // only rewrites the unqualified `sincosf`/`__sincosf` calls.
                            // Iter 3 v2 (double-precision sincos) was correct but slow:
                            // production-scale fwd 1.795 ms vs 0.512 ms for table path
                            // (cycle 5q perf check 2026-05-06). Iter 4 restores inline
                            // as the bandwidth-bound winner over tables.
                            const float inv_freq =
                                expf(-((float)pair_idx) / ((float)half_dim) * log_base);
                            const float angle = coord * inv_freq;
                            float sin_a, cos_a;
                            __nv_sincosf(angle, &sin_a, &cos_a);
                            // Inverse rotation: negate sin (R(-θ)).
                            if (Inverse) { sin_a = -sin_a; }
                            const float real = S_fp32(2 * i) * cos_a - S_fp32(2 * i + 1) * sin_a;
                            const float imag = S_fp32(2 * i) * sin_a + S_fp32(2 * i + 1) * cos_a;
                            S_fp32(2 * i) = real;
                            S_fp32(2 * i + 1) = imag;
                        }
                        Tensor S_fp32_copy = make_fragment_like(S_fp32);
                        cute::copy(S_fp32, S_fp32_copy);
                        using T = typename Engine0::value_type;
                        Tensor S_og_type = convert_type<T>(S_fp32_copy);
                        cute::copy(S_og_type, rS(_, m, k));
                    }
                    cute::copy(rS(_, m, k), D(_, m, k));
                } else if (Clear_OOB_K) {
                    cute::clear(D(_, m, k));
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Stage 4 v2: register-side inverse rotation on the GMM accumulator
// (before the smem write). Rationale: the original Stage 4 pattern read
// sdK/sdQ via gmem_tiled_copy_QKV but the smem was WRITTEN via a different
// copy atom (smem_tiled_copy_dKV / smem_tiled_copy_dQ). Partition
// mismatch between the two atoms leaves a subset of channels unrotated,
// causing biased gradients that pass single-step parity (averaged tol)
// but accumulate across training steps into wrong-direction trajectory
// (val mIoU 0.06 vs 0.35 in A/B v2).
//
// This helper bypasses smem entirely: applies R(-θ) directly to the
// register fragment using thr_mma.partition_C(identity) to derive the
// (n, k) coord for each register element. Pairs (real, imag) are at
// adjacent ncol positions in convert_layout_acc_rowcol's output, which
// matches the Sm80 mma_16x8x16 layout (cols 2i, 2i+1 are co-resident).
//
// Usage (FA-2 dK epilogue):
//   apply_acc_inverse_rotary_3d(
//       acc_dk, tiled_mma_dkv, tidx, params,
//       /*kBlock=*/kBlockN, /*kHead=*/kHeadDim,
//       /*row_base_global=*/cell_base_token + n_block * kBlockN,
//       /*max_seq=*/binfo.actual_seqlen_k - n_block * kBlockN);
//   // ... convert_type, smem write, gmem write proceed unchanged ...
//
// Same for dQ in postprocess (kBlock=kBlockM, row_base_global from m_block).
//
// `Element` template parameter is the bf16/fp16 model dtype. The helper
// inserts an explicit `Element` round-trip on the input acc values
// before rotation, matching Triton's bf16-load-fp32-rotate-bf16-store
// arithmetic path (which sees bf16-quantized inputs because dK_rot lives
// in gmem as bf16 between upstream-FA-bwd and Triton-inverse-prerotate).
// Without the round-trip, the fork rotates full-precision fp32 acc
// values, producing systematically different per-element output that
// compounds via chain rule across depth (cycle 5q triangulation:
// fork drifts 3.4× more than Triton from the pyref at depth=14;
// `~/.claude/projects/.../memory/project_stage4_v2_triangulation_results.md`).
// `coords_override` (MSGA combined indexed×fused): when non-null, the inline
// variant reads coords from this table instead of params.coords_3d_ptr. The
// dK epilogue passes the K-side coords (params.coords_3d_k_ptr — the gathered
// per-packed-position table of the unpacked mismatched-cu shape); the dQ
// postprocess passes nothing and keeps params.coords_3d_ptr. Positional
// indexing (n_global = packed varlen position) is unchanged either way.
template <typename Element, typename TensorAcc, typename TiledMma, typename Params>
__forceinline__ __device__ void apply_acc_inverse_rotary_3d(
    TensorAcc &acc,
    TiledMma const &tiled_mma,
    const int tidx,
    Params const &params,
    const int kBlock,
    const int kHead,
    const int row_base_global,
    const int max_seq,
    const float *coords_override = nullptr) {
    if (params.rotary_dim == 0 || !params.is_rotary_interleaved) return;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor c_id = make_identity_tensor(cute::make_shape(kBlock, kHead));
    Tensor thr_coord = thr_mma.partition_C(c_id);
    // Reshape both to rowcol = ((2, MMA_M), (2, MMA_N)) per Sm80 mma layout
    Tensor acc_rc = make_tensor(acc.data(), convert_layout_acc_rowcol(acc.layout()));
    Tensor coord_rc = make_tensor(thr_coord.data(), convert_layout_acc_rowcol(thr_coord.layout()));

    // Note: `Element` is now the template parameter (bf16/fp16 model dtype).
    // The previous local `using Element = typename TensorAcc::value_type` was
    // wrong — TensorAcc is fp32 (the GMM accumulator type), so the table-variant
    // pointer casts at the cos_e/sin_e reads below were silently casting to
    // float* even though the gmem tables are stored as bf16/fp16. The new
    // template-parameter Element is the correct model dtype.
    const int chunk_dim = params.chunk_dim_3d;
    const int rotary_dim = params.rotary_dim;
    const int half_chunk = chunk_dim >> 1;
    const float log_base = logf(params.rope_base_3d);
    const float coord_scale = params.coord_scale_3d;
    const float *coords_eff = coords_override != nullptr
        ? coords_override
        : reinterpret_cast<const float *>(params.coords_3d_ptr);
    const bool inline_mode = (coords_eff != nullptr);

    #pragma unroll
    for (int mi = 0; mi < cute::size<0>(acc_rc); ++mi) {
        const int n_local = cute::get<0>(coord_rc(mi, 0));
        if (n_local >= max_seq) continue;
        const int n_global = row_base_global + n_local;
        // Inline variant: load coords once per row.
        float c0 = 0.f, c1 = 0.f, c2 = 0.f;
        if (inline_mode) {
            c0 = coords_eff[n_global * 3 + 0] * coord_scale;
            c1 = coords_eff[n_global * 3 + 1] * coord_scale;
            c2 = coords_eff[n_global * 3 + 2] * coord_scale;
        }
        #pragma unroll
        for (int ni = 0; ni < cute::size<1>(acc_rc) / 2; ++ni) {
            // Get global head_dim col for the pair (real at 2*ni, imag at 2*ni+1).
            const int col_real = cute::get<1>(coord_rc(mi, 2 * ni));
            const int col_imag = cute::get<1>(coord_rc(mi, 2 * ni + 1));
            // Skip pairs that aren't (col, col+1) or are past rotary_dim.
            if (col_imag != col_real + 1) continue;
            if (col_real >= rotary_dim) continue;
            const int axis = col_real / chunk_dim;
            const int pair_idx = (col_real % chunk_dim) >> 1;
            float cos_a, sin_a;
            if (inline_mode) {
                const float coord = (axis == 0 ? c0 : axis == 1 ? c1 : c2);
                // Iter 4: explicit libdevice precise __nv_sincosf (~1 ULP,
                // matches Triton's tl.math.cos/sin) at float speed. Bypasses
                // --use_fast_math which only rewrites unqualified `sincosf`/
                // `__sincosf`. Iter 3 v2 (double-precision sincos) was correct
                // but slow; this restores inline as the perf winner.
                const float inv_freq =
                    expf(-((float)pair_idx) / ((float)half_chunk) * log_base);
                const float angle = coord * inv_freq;
                __nv_sincosf(angle, &sin_a, &cos_a);
            } else {
                // Table variant: cos/sin tables at row n_global, col pair_idx + axis*half_chunk
                const float *cos_ptr = reinterpret_cast<const float *>(params.rotary_cos_ptr);
                const float *sin_ptr = reinterpret_cast<const float *>(params.rotary_sin_ptr);
                // tables are stored as Element (fp16/bf16), but for register-side we cast on read
                // ... actually rotary_cos_ptr is Element*, not float*. Read via Element pointer.
                const Element *cos_e = reinterpret_cast<const Element *>(params.rotary_cos_ptr);
                const Element *sin_e = reinterpret_cast<const Element *>(params.rotary_sin_ptr);
                const int col_pair_global = pair_idx + axis * half_chunk;
                cos_a = static_cast<float>(cos_e[n_global * (rotary_dim / 2) + col_pair_global]);
                sin_a = static_cast<float>(sin_e[n_global * (rotary_dim / 2) + col_pair_global]);
            }
            // Inverse rotation: R(-θ) negates sin
            sin_a = -sin_a;
            // Round-trip through Element (bf16 or fp16) to match Triton's
            // arithmetic path. Triton's input to rotation comes from gmem
            // already at Element precision; rotating full-precision fp32
            // here would diverge per-call and compound at depth.
            const float a_raw = static_cast<float>(acc_rc(mi, 2 * ni));
            const float b_raw = static_cast<float>(acc_rc(mi, 2 * ni + 1));
            const float a = static_cast<float>(static_cast<Element>(a_raw));
            const float b = static_cast<float>(static_cast<Element>(b_raw));
            acc_rc(mi, 2 * ni)     = a * cos_a - b * sin_a;
            acc_rc(mi, 2 * ni + 1) = a * sin_a + b * cos_a;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace FLASH_NAMESPACE
