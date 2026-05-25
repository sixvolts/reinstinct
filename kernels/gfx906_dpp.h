// gfx906 (MI50/MI60) DPP-based warp reductions.
//
// Replaces the cross-lane part of `__shfl_xor(x, off)` with the cheaper
// Vega DPP and LDS-swizzle instructions that bypass the SGPR/LDS
// roundtrip the compiler emits for __shfl_xor.
//
// Cost (per call, one float):
//   xor1, xor2, xor8   → 1 DPP VALU op + one nop slot
//   xor4              → 3 VALU ops (DPP can't do swap-of-4 on Vega)
//   xor16             → 1 ds_swizzle + s_waitcnt
//   xor32             → falls back to __shfl_xor (Vega DPP tops out at 16-lane)
//
// Source: llamacpp-gfx906-furnace (gfx906-common.cuh:73-142), itself
// derived from iacopPBK/llama.cpp-gfx906's Wave64 kernels.
//
// Include this from any *.cpp gfx906 kernel that does a `__shfl_xor`
// reduction; replace the cascade with `wave64_reduce_add_f32(x)` (or
// the max variant). Width-32 / width-16 variants drop the final passes.

#pragma once
#include <hip/hip_runtime.h>

#define REINSTINCT_DEFINE_DPP_F32(name, barrier, dpp_ctrl, vop_instr)      \
    __device__ __forceinline__ float name(float x) {                       \
        float r;                                                            \
        asm volatile(                                                       \
            barrier                                                         \
            vop_instr " %0, %1, %1 " dpp_ctrl " row_mask:0xf bank_mask:0xf" \
            : "=v"(r) : "v"(x) : "memory");                                 \
        return r;                                                           \
    }

REINSTINCT_DEFINE_DPP_F32(rein_add_xor1_f32, "s_nop 4\n", "quad_perm:[1,0,3,2]", "v_add_f32_dpp")
REINSTINCT_DEFINE_DPP_F32(rein_add_xor2_f32, "s_nop 1\n", "quad_perm:[2,3,0,1]", "v_add_f32_dpp")
REINSTINCT_DEFINE_DPP_F32(rein_add_xor8_f32, "s_nop 1\n", "row_ror:8",           "v_add_f32_dpp")
REINSTINCT_DEFINE_DPP_F32(rein_max_xor1_f32, "s_nop 4\n", "quad_perm:[1,0,3,2]", "v_max_f32_dpp")
REINSTINCT_DEFINE_DPP_F32(rein_max_xor2_f32, "s_nop 1\n", "quad_perm:[2,3,0,1]", "v_max_f32_dpp")
REINSTINCT_DEFINE_DPP_F32(rein_max_xor8_f32, "s_nop 1\n", "row_ror:8",           "v_max_f32_dpp")

#undef REINSTINCT_DEFINE_DPP_F32

// xor4 has no single DPP op on Vega — emulated via two masked row shifts.
__device__ __forceinline__ float rein_xfer_xor4_f32(float x) {
    int s = __float_as_int(x);
    int d;
    asm volatile(
        "v_mov_b32 %0, %1\n"
        "s_nop 1\n"
        "v_mov_b32_dpp %0, %1 row_shl:4 row_mask:0xf bank_mask:0x5\n"
        "v_mov_b32_dpp %0, %1 row_shr:4 row_mask:0xf bank_mask:0xa\n"
        : "=v"(d) : "v"(s) : "memory");
    return __int_as_float(d);
}

// xor16 via LDS swizzle; cheaper than the __shfl_xor lowering's full LDS
// roundtrip because ds_swizzle is a single in-place permutation.
__device__ __forceinline__ float rein_xfer_xor16_f32(float x) {
    int s = __float_as_int(x);
    int d;
    asm volatile(
        "ds_swizzle_b32 %0, %1 offset:swizzle(SWAP,16)\n"
        "s_waitcnt lgkmcnt(0)\n"
        : "=v"(d) : "v"(s) : "memory");
    return __int_as_float(d);
}

// Full-width Wave64 sum reduction. Five DPP/swizzle ops + one shfl
// for the cross-half step. Caller's value at lane 0 holds the wave sum.
__device__ __forceinline__ float wave64_reduce_add_f32(float x) {
    x = rein_add_xor1_f32(x);
    x = rein_add_xor2_f32(x);
    x = x + rein_xfer_xor4_f32(x);
    x = rein_add_xor8_f32(x);
    x = x + rein_xfer_xor16_f32(x);
    x = x + __shfl_xor(x, 32);
    return x;
}

// Half-wave (32-lane) sum reduction. Drops the cross-half shfl.
__device__ __forceinline__ float wave32_reduce_add_f32(float x) {
    x = rein_add_xor1_f32(x);
    x = rein_add_xor2_f32(x);
    x = x + rein_xfer_xor4_f32(x);
    x = rein_add_xor8_f32(x);
    x = x + rein_xfer_xor16_f32(x);
    return x;
}

// Full-width Wave64 max reduction. Same shape as the add variant.
__device__ __forceinline__ float wave64_reduce_max_f32(float x) {
    x = rein_max_xor1_f32(x);
    x = rein_max_xor2_f32(x);
    x = fmaxf(x, rein_xfer_xor4_f32(x));
    x = rein_max_xor8_f32(x);
    x = fmaxf(x, rein_xfer_xor16_f32(x));
    x = fmaxf(x, __shfl_xor(x, 32));
    return x;
}

// Hardware reciprocal via v_rcp_f32 — skips the IEEE division pipeline.
// ~24-bit precision (the v_rcp_f32 ULP is one); fine for int8 quantize
// scaling where the downstream output only carries 7-8 bits anyway.
// Source: llamacpp-gfx906-furnace (gfx906-common.cuh:63-71).
__device__ __forceinline__ float fast_rcp_f32(float x) {
    float r;
    asm volatile("v_rcp_f32 %0, %1" : "=v"(r) : "v"(x));
    return r;
}
