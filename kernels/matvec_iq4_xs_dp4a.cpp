// dp4a IQ4_XS matvec — int8-quantized activation × IQ4_XS weight.
//
// Mirrors matvec_q5_k_dp4a_f32's shape: per-row 64-thread workgroup,
// each lane strides sub-blocks of 32 weights. The wave64 fp32 variant
// was 451 µs/call on qwen-3.5-27B decode (~12% of GPU time); switching
// the 32 fp32 multiplies per sub-block to 8 sdot4 ops should bring it
// to ~70 µs in line with the other K-quant dp4a kernels.
//
// IQ4_XS encodes 256 weights per 136-byte super-block:
//   uint16  d                 fp16 super-block scale
//   uint16  scales_h          2 high bits × 8 sub-blocks
//   uint8   scales_l[4]       4 low bits × 8 sub-blocks
//   uint8   qs[128]           8 sub-blocks × 16 nibbles-pairs
//
// Per sub-block ib (32 weights):
//   ls    = scales_l_nib(ib) | (scales_h_pair(ib) << 4)  ∈ [0, 63]
//   dl    = d * (ls - 32)
//   For l in [0, 16):
//     w[l]      = dl * KVALUES_IQ4NL[qs[ib*16 + l] & 0xF]
//     w[l + 16] = dl * KVALUES_IQ4NL[qs[ib*16 + l] >> 4]
//
// KVALUES_IQ4NL is asymmetric (-127..113, NOT centered on 0), so the
// scaling is just `acc += dl * dx * idot` — no Q4_K-style dmin/xsum
// offset needed. dx is the activation block's per-32 scale.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include "gfx906_dpp.h"

struct __attribute__((packed)) BlockIQ4_XS {
    uint16_t d;
    uint16_t scales_h;
    uint8_t  scales_l[4];
    uint8_t  qs[128];
};
static_assert(sizeof(BlockIQ4_XS) == 136, "BlockIQ4_XS must be 136 bytes");

struct __attribute__((packed)) BlockQ8 {
    float  d;
    float  xsum;     // unused for IQ4_XS — kept for layout compat
    int8_t qs[32];
};
static_assert(sizeof(BlockQ8) == 40, "BlockQ8 must be 40 bytes");

__device__ static const int8_t KVALUES_IQ4NL[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
       1,   13,  25,  38,  53,  69,  89, 113,
};

// Same KVALUES_IQ4NL table, packed for v_perm_b32 (__builtin_amdgcn_perm).
// v_perm picks 4 bytes from an 8-byte source pair using a per-byte
// selector. The 16-entry table splits in half: indices 0..7 from
// the "low" pair, 8..15 from the "high" pair. We dispatch on bit 3
// of each nibble and mux the two perm results.
//
// Pre-packed little-endian: byte 0 of low_lo = -127 (=0x81), byte 1
// = -104 (=0x98), ..., byte 0 of high_lo = 1 (=0x01), etc.
__device__ static constexpr uint32_t IQ4NL_LO_LO = 0xBFAD9881u; // -127,-104,-83,-65
__device__ static constexpr uint32_t IQ4NL_LO_HI = 0xF6EADDCFu; // -49,-35,-22,-10
__device__ static constexpr uint32_t IQ4NL_HI_LO = 0x26190D01u; //   1, 13, 25, 38
__device__ static constexpr uint32_t IQ4NL_HI_HI = 0x71594535u; //  53, 69, 89,113

// Look up 4 IQ4NL values in parallel. `nibs` packs 4 nibbles in 4 bytes
// (each byte holds a 4-bit table index, top nibble of byte must be 0).
// Returns the 4 looked-up int8 codebook values packed as 4 bytes in a
// uint32 (same byte order as the input). Two v_perm_b32 + a per-byte
// mux: half the instructions vs the literal byte-by-byte LDS table
// lookup the compiler emits for `KVALUES_IQ4NL[byte]`.
__device__ __forceinline__ uint32_t iq4nl_lookup4(uint32_t nibs)
{
    const uint32_t sel  = nibs & 0x07070707u;           // low 3 bits → source-byte index
    const uint32_t bit3 = (nibs >> 3) & 0x01010101u;
    const uint32_t hi_mask = bit3 * 0xFFu;              // 0xFF in bytes that want the hi half
    const uint32_t lo = __builtin_amdgcn_perm(IQ4NL_LO_HI, IQ4NL_LO_LO, sel);
    const uint32_t hi = __builtin_amdgcn_perm(IQ4NL_HI_HI, IQ4NL_HI_LO, sel);
    return (lo & ~hi_mask) | (hi & hi_mask);
}

extern "C" __global__
void matvec_iq4_xs_dp4a_f32(const BlockIQ4_XS* __restrict__ w_blocks,
                            const BlockQ8*     __restrict__ xq,
                            float*             __restrict__ y,
                            unsigned int in_dim,
                            unsigned int out_dim)
{
    const int row = blockIdx.x;
    if (row >= (int)out_dim) return;
    const int lane = threadIdx.x;

    const unsigned int n_blocks    = in_dim >> 8;   // 256-weight super-blocks
    const unsigned int n_subblocks = in_dim >> 5;   // 32-weight sub-blocks
    const BlockIQ4_XS* row_blocks = w_blocks + (size_t)row * n_blocks;

    float acc = 0.0f;
    for (int sb = lane; sb < (int)n_subblocks; sb += 64) {
        const int blk_idx = sb >> 3;
        const int ib      = sb & 7;

        const BlockIQ4_XS* blk = row_blocks + blk_idx;
        const __half d_h = *reinterpret_cast<const __half*>(&blk->d);
        const float  d   = __half2float(d_h);

        const int ls_lo = (blk->scales_l[ib >> 1] >> (4 * (ib & 1))) & 0x0F;
        const int ls_hi = (blk->scales_h >> (2 * ib)) & 0x3;
        const int ls    = ls_lo | (ls_hi << 4);
        const float dl  = d * (float)(ls - 32);

        const BlockQ8* xb   = xq + sb;
        const int*     xq32 = reinterpret_cast<const int*>(xb->qs);
        const float    dx   = xb->d;

        const uint8_t* qs_base = blk->qs + ib * 16;

        int idot = 0;
        const uint32_t* qs32 = reinterpret_cast<const uint32_t*>(qs_base);
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            // Load 4 packed (lo+hi) nibble bytes in one dword, then look
            // up all 8 nibbles in parallel via v_perm_b32. The dp4a
            // consumes 4 weights at a time; one lookup feeds the lo
            // dot, the other feeds the hi dot.
            const uint32_t bytes    = qs32[j];
            const uint32_t lo_nibs  = bytes & 0x0F0F0F0Fu;
            const uint32_t hi_nibs  = (bytes >> 4) & 0x0F0F0F0Fu;
            const uint32_t lo_packed = iq4nl_lookup4(lo_nibs);
            const uint32_t hi_packed = iq4nl_lookup4(hi_nibs);
            // xq32[0..3] = quants for weight slot 0..15 in this sub-block;
            // xq32[4..7] = quants for weight slot 16..31.
            idot = __builtin_amdgcn_sdot4((int)lo_packed, xq32[j],     idot, false);
            idot = __builtin_amdgcn_sdot4((int)hi_packed, xq32[j + 4], idot, false);
        }
        acc += dl * dx * (float)idot;
    }

    acc = wave64_reduce_add_f32(acc);

    if (lane == 0) y[row] = acc;
}
