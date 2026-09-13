// Shared pieces of the GQA flash-decoding kernels
// (attn_decode_gqa_f32.cpp, attn_decode_gqa_q8.cpp): the all-DPP
// segment reductions.
//
// `seg_sum_multi<N, LPT>` sums each of N values across the LPT lanes of
// its segment (LPT a power of two, 1..64). Through the compiler's DPP
// builtin so it owns the read-after-write hazards and the schedule — an
// inline-asm version silently broke under register pressure. The
// cross-row steps use row_bcast:15 / row_bcast:31 (GFX9) instead of an
// LDS swizzle and its wait, which was the whole kernel at two waves per
// SIMD. The result is valid in the segment's last lane (dl == LPT-1);
// for LPT <= 16 in every lane of the row. `seg_max_multi` likewise.
#pragma once
#include <hip/hip_runtime.h>

#define DPP_QP_XOR1     0xB1     // quad_perm:[1,0,3,2]
#define DPP_QP_XOR2     0x4E     // quad_perm:[2,3,0,1]
#define DPP_ROW_SHL4    0x104
#define DPP_ROW_SHR4    0x114
#define DPP_ROW_ROR8    0x128
#define DPP_ROW_BCAST15 0x142
#define DPP_ROW_BCAST31 0x143

template <int CTRL, int ROW_MASK, int BANK_MASK>
__device__ __forceinline__ float dpp_f32(float old, float src) {
    return __int_as_float(__builtin_amdgcn_update_dpp(__float_as_int(old), __float_as_int(src),
                                                      CTRL, ROW_MASK, BANK_MASK, false));
}

struct SegAdd { __device__ static __forceinline__ float op(float a, float b) { return a + b; }
                __device__ static __forceinline__ float id() { return 0.0f; } };
struct SegMax { __device__ static __forceinline__ float op(float a, float b) { return fmaxf(a, b); }
                __device__ static __forceinline__ float id() { return -INFINITY; } };

template <class OP, int N, int LPT>
__device__ __forceinline__ void seg_red_multi(float (&x)[N]) {
    #pragma unroll
    for (int g = 0; g < N; g++) {
        float v = x[g];
        if (LPT > 1) v = OP::op(v, dpp_f32<DPP_QP_XOR1, 0xf, 0xf>(OP::id(), v));
        if (LPT > 2) v = OP::op(v, dpp_f32<DPP_QP_XOR2, 0xf, 0xf>(OP::id(), v));
        if (LPT > 4) {
            float t = dpp_f32<DPP_ROW_SHL4, 0xf, 0x5>(OP::id(), v);   // banks 0,2 take lane+4
            t = dpp_f32<DPP_ROW_SHR4, 0xf, 0xa>(t, v);                // banks 1,3 take lane-4
            v = OP::op(v, t);
        }
        if (LPT > 8)  v = OP::op(v, dpp_f32<DPP_ROW_ROR8, 0xf, 0xf>(OP::id(), v));
        if (LPT > 16) v = OP::op(v, dpp_f32<DPP_ROW_BCAST15, 0xa, 0xf>(OP::id(), v));  // rows 1,3 <- lane 15 of rows 0,2
        if (LPT > 32) v = OP::op(v, dpp_f32<DPP_ROW_BCAST31, 0xc, 0xf>(OP::id(), v));  // rows 2,3 <- lane 31
        x[g] = v;
    }
}
template <int N, int LPT> __device__ __forceinline__ void seg_sum_multi(float (&x)[N]) { seg_red_multi<SegAdd, N, LPT>(x); }
template <int N, int LPT> __device__ __forceinline__ void seg_max_multi(float (&x)[N]) { seg_red_multi<SegMax, N, LPT>(x); }
