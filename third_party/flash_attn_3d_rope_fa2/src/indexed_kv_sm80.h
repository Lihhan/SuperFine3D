/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include "namespace_config.h"

#include <cutlass/cutlass.h>

#include "cute/tensor.hpp"

namespace FLASH_NAMESPACE {

using namespace cute;

// ────────────────────────────────────────────────────────────────────────────
// MSGA indexed fused-KV loader — FA-2 port.
//
// PROVENANCE: copied from the fa3 fork's src/indexed_kv_sm80.h (MSGA Stage A,
// itself adapted from FA-3's PagedKVManager) into this fa2 csrc, per the
// copy-with-provenance convention (no cross-package includes). The loader is
// already cp.async/sm_8x-native — exactly the copy mechanism FA-2's own
// GmemTiledCopyQKV uses — so it ports with two deliberate adaptations:
//
//   1. namespace flash → FLASH_NAMESPACE (this fork's symbol-collision
//      namespace macro, see setup.py NAMESPACE).
//   2. cute::conditional_return<B>(t, f) (template form, newer cute) →
//      cute::conditional_return(bool_constant<B>{}, t, f) (tag-dispatch form;
//      the older cutlass vendored here only has the latter). FA-2 always
//      instantiates kHeadDimV == kHeadDim (one head dim per kernel traits),
//      so the SameHeadDim branch is statically taken.
//
// Thread-geometry note vs the surrounding FA-2 kernel: this manager derives
// its own kGmemThreadsPerRow from kBytePerRow (128B/64B/32B alignment), while
// FA-2's kernel traits use kBlockKSmem/kGmemElemsPerLoad. For the head dims
// this fork builds at ≤64 the two are IDENTICAL (hd16→2, hd32→4, hd64→8
// threads/row); at hd128 they would diverge (16 vs 8 threads/row) — that is
// only a bank-conflict perf nuance on the smem write, not a correctness
// issue, because the manager is self-consistent (it partitions the smem
// destination with its own tiled copy) and the kernel syncs before reading.
//
// Semantics (same as the fa3 original): per-row cp.async loads through a
// flat row index
//     src_row = kv_row_indices[seq_offset_k + n_block*kBlockN + row]
// (`row_indices` below is pre-offset by seq_offset_k = cu_seqlens_k[bidb]),
// addressing SOURCE K/V buffers [N_src, h_k, d]. Each thread of a row-group
// precomputes the row pointers for its share of the tile's rows
// (kRowEntryPerThread entries, strided row mapping), `__shfl_sync` broadcasts
// the pointer to the kGmemThreadsPerRow threads covering that row, and each
// thread issues its 16-byte sub-row copies with the ZFILL copy atom and the
// same seqlen predication as the non-indexed loads. KV_Same_Iter flow:
// load_V recomputes V pointers from the rows stored by the load_K /
// load_row_indices of the same n_block. OOB rows (beyond seqlen_k / beyond
// kBlockN) are predicated exactly as the non-indexed loads: K rows are
// skipped (scores masked downstream) unless ZeroFillOOB (bwd — see load_K),
// V rows are ZFILLed; the guarded index fetch never reads kv_row_indices OOB.
// ────────────────────────────────────────────────────────────────────────────
template <int kBlockN, int kHeadDim, int kHeadDimV, int NumThreads, typename Element>
struct IndexedKVManager {

    static constexpr bool SameHeadDim = (kHeadDim == kHeadDimV);
    static constexpr int kHeadDimGCD = cute::gcd(kHeadDim, kHeadDimV);

    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static_assert(kHeadDimGCD % kGmemElemsPerLoad == 0, "Headdim and HeaddimV must be a multiple of kGmemElemsPerLoad");
    static constexpr int kBytePerRow = kHeadDimGCD * sizeof(Element);
    static constexpr int kBlockKGmem = (kBytePerRow % 128 == 0 ? 128 : (kBytePerRow % 64 == 0 ? 64 : 32)) / sizeof(Element);
    static constexpr int kGmemThreadsPerRow = kBlockKGmem / kGmemElemsPerLoad;
    static_assert(NumThreads % kGmemThreadsPerRow == 0, "NumThreads must be a multiple of kGmemThreadsPerRow");
    // Threads loading the same row must be in the same warp so the row pointer
    // can be shfl-broadcast (same requirement as PagedKVManager).
    static_assert(cutlass::NumThreadsPerWarp % kGmemThreadsPerRow == 0, "kGmemThreadsPerRow must divide NumThreadsPerWarp");
    using GmemCopyAtomCpAsync = cute::Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<uint128_t>, Element>;
    using GmemLayoutAtomKVCpAsync = Layout<Shape <Int<NumThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                           Stride<Int<kGmemThreadsPerRow>, _1>>;
    using GmemTiledCopyKVCpAsync = decltype(
        make_tiled_copy(GmemCopyAtomCpAsync{},
                        GmemLayoutAtomKVCpAsync{},
                        Layout<Shape<_1, Int<kGmemElemsPerLoad>>>{}));  // Val layout, 8 or 16 vals per load

    using GmemThrCopyKVCpAsync = decltype(GmemTiledCopyKVCpAsync{}.get_thread_slice(int(0)));
    using TensortKcK = decltype(GmemTiledCopyKVCpAsync{}.get_thread_slice(int(0)).partition_D(cute::make_identity_tensor(Shape<Int<kBlockN>, Int<kHeadDim>>{})));
    using TensortKpK = decltype(make_tensor<bool>(make_shape(size<1>(TensortKcK{}), size<2>(TensortKcK{})), Stride<_0, _1>{}));
    using TensortVcV = decltype(GmemTiledCopyKVCpAsync{}.get_thread_slice(int(0)).partition_D(cute::make_identity_tensor(Shape<Int<kBlockN>, Int<kHeadDimV>>{})));
    using TensortVpV = decltype(make_tensor<bool>(make_shape(size<1>(TensortVcV{}), size<2>(TensortVcV{})), Stride<_0, _1>{}));

    static_assert(CUTE_STATIC_V(size<1>(TensortKcK{})) == CUTE_STATIC_V(size<1>(TensortVcV{})));
    static constexpr int kRowEntryPerThread = cute::ceil_div(size<1>(TensortKcK{}), kGmemThreadsPerRow);
    using TensorSrcRow = decltype(make_tensor<int>(Shape<Int<kRowEntryPerThread>>{}));
    using TensorKVPtr = decltype(make_tensor<Element const*>(Shape<Int<kRowEntryPerThread>>{}));

    GmemTiledCopyKVCpAsync gmem_tiled_copy_kv;
    int const thread_idx;
    int const seqlen_k;
    int const* const row_indices;        // [seqlen_k] — already offset to this sequence's start
    Element const* const base_K;         // K source base, pre-offset by bidh * head_stride
    Element const* const base_V;
    int64_t const stride_K_row;
    int64_t const stride_V_row;
    GmemThrCopyKVCpAsync const gmem_thr_copy_kv;
    TensortKpK tKpK;
    TensortVpV tVpV;
    TensorSrcRow tPrSrcRow;
    TensorKVPtr tPrVPtr;

    CUTLASS_DEVICE
    IndexedKVManager(int const* const row_indices_seq,  // may be nullptr (manager then unused)
                     Element const* const ptr_K, int64_t const k_row_stride, int64_t const k_head_stride,
                     Element const* const ptr_V, int64_t const v_row_stride, int64_t const v_head_stride,
                     int const d, int const dv,
                     int const bidh, int const thread_idx, int const seqlen_k)
        : thread_idx(thread_idx)
        , seqlen_k(seqlen_k)
        , row_indices(row_indices_seq)
        , base_K(ptr_K + bidh * k_head_stride)
        , base_V(ptr_V + bidh * v_head_stride)
        , stride_K_row(k_row_stride)
        , stride_V_row(v_row_stride)
        , gmem_thr_copy_kv(gmem_tiled_copy_kv.get_thread_slice(thread_idx))
    {
        // Column (head_dim) predicates vs the actual d / dv — mirrors
        // PagedKVManager's tKpK/tVpV construction.
        tKpK = make_tensor<bool>(make_shape(size<1>(TensortKcK{}), size<2>(TensortKcK{})), Stride<_0, _1>{});
        Tensor cK = cute::make_identity_tensor(Shape<Int<kBlockN>, Int<kHeadDim>>{});
        Tensor tKcK = gmem_thr_copy_kv.partition_S(cK);
        #pragma unroll
        for (int k = 0; k < size<1>(tKpK); ++k) { tKpK(_0{}, k) = get<1>(tKcK(_0{}, _0{}, k)) < d; }
        Tensor tVpV_ = make_tensor<bool>(make_shape(size<1>(TensortVcV{}), size<2>(TensortVcV{})), Stride<_0, _1>{});
        Tensor cV = cute::make_identity_tensor(Shape<Int<kBlockN>, Int<kHeadDimV>>{});
        Tensor tVcV = gmem_thr_copy_kv.partition_S(cV);
        #pragma unroll
        for (int k = 0; k < size<1>(tVpV_); ++k) { tVpV_(_0{}, k) = get<1>(tVcV(_0{}, _0{}, k)) < dv; }
        // fa3 original: cute::conditional_return<SameHeadDim>(tKpK, tVpV_);
        // this vendored cutlass only ships the tag-dispatch overload.
        tVpV = cute::conditional_return(cute::bool_constant<SameHeadDim>{}, tKpK, tVpV_);
    };

    // The flat-index analog of PagedKVManager::load_page_table — same strided
    // per-thread row mapping (so the shfl in load_K/load_V picks the right
    // provider lane), guarded so kv_row_indices is never read out of bounds.
    template <bool Seqlenk_mask=false>
    CUTLASS_DEVICE
    void load_row_indices(const int n_block) {
        #pragma unroll
        for (int i = 0; i < kRowEntryPerThread; ++i) {
            int const row = i * NumThreads + (thread_idx % kGmemThreadsPerRow) * (NumThreads / kGmemThreadsPerRow) + (thread_idx / kGmemThreadsPerRow);
            int const row_idx = n_block * kBlockN + row;
            // (i + 1) * NumThreads <= kBlockN is a compile-time upper bound on
            // row, avoiding the branch where possible (same as paged_kv.h).
            bool const in_bounds = ((i + 1) * NumThreads <= kBlockN || row < kBlockN)
                && (!Seqlenk_mask || row_idx < seqlen_k);
            tPrSrcRow[i] = in_bounds ? row_indices[row_idx] : 0;
        }
    };

    CUTLASS_DEVICE
    TensorKVPtr compute_K_ptr() {
        Tensor tPrKPtr = make_tensor<Element const*>(Shape<Int<kRowEntryPerThread>>{});
        #pragma unroll
        for (int i = 0; i < kRowEntryPerThread; ++i) {
            tPrKPtr[i] = base_K + int64_t(tPrSrcRow[i]) * stride_K_row;
        }
        return tPrKPtr;
    };

    CUTLASS_DEVICE
    void compute_V_ptr() {
        #pragma unroll
        for (int i = 0; i < kRowEntryPerThread; ++i) {
            tPrVPtr[i] = base_V + int64_t(tPrSrcRow[i]) * stride_V_row;
        }
    };

    // Body mirrors PagedKVManager::load_K — skip OOB rows (scores are masked
    // downstream), ZFILL OOB head_dim columns.
    //
    // ZeroFillOOB (bwd prologue): the fwd skip-OOB-rows behavior is NOT safe
    // in the bwd — sK feeds the dQ gemm (dQ += dS · K) where the dS
    // coefficient at masked positions is 0, and 0 · garbage-smem (possible
    // Inf/NaN bit patterns) = NaN. The stock bwd prologue therefore ZFILLs OOB
    // K rows (FA-2: flash::copy with Clear_OOB_MN=true); ZeroFillOOB=true
    // reproduces exactly that (copy issued with should_load folded into the
    // predicate, under the same don't-write-smem-past-kBlockN guard structure
    // as load_V).
    template <bool Seqlenk_mask=false, bool ZeroFillOOB=false, typename TensorK>
    CUTLASS_DEVICE
    void load_K(const int n_block, TensorK &&sK) {
        static constexpr bool EvenN = kBlockN % CUTE_STATIC_V(shape<0>(GmemLayoutAtomKVCpAsync{})) == 0;

        Tensor tPrKPtr = compute_K_ptr();

        // Only for index calculation, since all the indices of thread 0 are known at compile time
        auto gmem_thr0_copy_kv = gmem_tiled_copy_kv.get_thread_slice(_0{});
        Tensor tKsK = gmem_thr_copy_kv.partition_D(sK);
        Tensor cK = cute::make_identity_tensor(Shape<Int<kBlockN>, Int<kHeadDim>>{});  // (BLK_N,BLK_K) -> (blk_n,blk_k)
        Tensor tKcK = gmem_thr_copy_kv.partition_S(cK);
        Tensor t0KcK = gmem_thr0_copy_kv.partition_S(cK);

        int const seqlenk_row_limit = -int(get<0>(tKcK(_0{}, _0{}, _0{}))) + (EvenN
            ? seqlen_k - n_block * kBlockN
            : (!Seqlenk_mask ? kBlockN : std::min(seqlen_k - n_block * kBlockN, kBlockN)));
        #pragma unroll
        for (int m = 0; m < size<1>(tKsK); ++m) {
            bool const should_load = EvenN
                ? (!Seqlenk_mask || get<0>(t0KcK(_0{}, m, _0{})) < seqlenk_row_limit)
                : get<0>(t0KcK(_0{}, m, _0{})) < seqlenk_row_limit;
            Element const* k_ptr = reinterpret_cast<Element const*>(__shfl_sync(0xffffffff, reinterpret_cast<uint64_t>(tPrKPtr(m / kGmemThreadsPerRow)), (m % kGmemThreadsPerRow), kGmemThreadsPerRow));
            Tensor mK_cur = make_tensor(make_gmem_ptr(k_ptr), Shape<Int<kHeadDim>>{});
            Tensor mK_cur_copy = cute::tiled_divide(mK_cur, Shape<Int<kGmemElemsPerLoad>>{});
            if constexpr (!ZeroFillOOB) {
                if (should_load) {
                    #pragma unroll
                    for (int k = 0; k < size<2>(tKsK); ++k) {
                        int const ki = get<1>(tKcK(_0{}, _0{}, k)) / kGmemElemsPerLoad;
                        cute::copy(gmem_tiled_copy_kv.with(tKpK(_0{}, k)), mK_cur_copy(_, ki), tKsK(_, m, k));
                    }
                }  // Don't need to clear out the rest of the smem since we'll mask out the scores anyway
            } else {
                // ZFILL OOB rows (predicate folded into the cp.async — writes
                // zeros, never dereferences gmem); same smem-bounds guard
                // structure as load_V for the !EvenN tail.
                if (EvenN || m < size<1>(tKsK) - 1 || get<0>(tKcK(_0{}, m, _0{})) < kBlockN) {
                    #pragma unroll
                    for (int k = 0; k < size<2>(tKsK); ++k) {
                        int const ki = get<1>(tKcK(_0{}, _0{}, k)) / kGmemElemsPerLoad;
                        cute::copy(gmem_tiled_copy_kv.with(tKpK(_0{}, k) && should_load), mK_cur_copy(_, ki), tKsK(_, m, k));
                    }
                }
            }
        }
    };

    // Body mirrors PagedKVManager::load_V (KV_Same_Iter=true flow: V pointers
    // recomputed here from the same rows the matching load_row_indices stored).
    template <bool Seqlenk_mask=false, typename TensorV>
    CUTLASS_DEVICE
    void load_V(const int n_block, TensorV &&sV) {
        static constexpr bool EvenN = kBlockN % CUTE_STATIC_V(shape<0>(GmemLayoutAtomKVCpAsync{})) == 0;

        compute_V_ptr();
        // Only for index calculation, since all the indices of thread 0 are known at compile time
        auto gmem_thr0_copy_kv = gmem_tiled_copy_kv.get_thread_slice(_0{});
        Tensor tVsV = gmem_thr_copy_kv.partition_D(sV);
        Tensor cV = cute::make_identity_tensor(Shape<Int<kBlockN>, Int<kHeadDimV>>{});  // (BLK_N,BLK_K) -> (blk_n,blk_k)
        Tensor tVcV = gmem_thr_copy_kv.partition_S(cV);
        Tensor t0VcV = gmem_thr0_copy_kv.partition_S(cV);

        int const seqlenk_row_limit = seqlen_k - n_block * kBlockN - get<0>(tVcV(_0{}, _0{}, _0{}));
        #pragma unroll
        for (int m = 0; m < size<1>(tVsV); ++m) {
            // OOB rows ZFILL via the predicated cp.async (never dereferenced);
            // we just must not write smem past kBlockN when !EvenN.
            if (EvenN || m < size<1>(tVsV) - 1 || get<0>(tVcV(_0{}, m, _0{})) < kBlockN) {
                bool const should_load = !Seqlenk_mask || get<0>(t0VcV(_0{}, m, _0{})) < seqlenk_row_limit;
                Element const* v_ptr = reinterpret_cast<Element const*>(__shfl_sync(0xffffffff, reinterpret_cast<uint64_t>(tPrVPtr(m / kGmemThreadsPerRow)), m % kGmemThreadsPerRow, kGmemThreadsPerRow));
                Tensor mV_cur = make_tensor(make_gmem_ptr(v_ptr), Shape<Int<kHeadDimV>>{});
                Tensor mV_cur_copy = cute::tiled_divide(mV_cur, Shape<Int<kGmemElemsPerLoad>>{});
                #pragma unroll
                for (int k = 0; k < size<2>(tVsV); ++k) {
                    int const ki = get<1>(tVcV(_0{}, _0{}, k)) / kGmemElemsPerLoad;
                    cute::copy(gmem_tiled_copy_kv.with(tVpV(_0{}, k) && should_load), mV_cur_copy(_, ki), tVsV(_, m, k));
                }
            }
        }
    };

};

} // namespace FLASH_NAMESPACE
