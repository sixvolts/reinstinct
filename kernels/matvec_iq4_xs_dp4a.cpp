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
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            // Pack 4 lo nibbles + 4 hi nibbles into 2 int32s via the
            // KVALUES lookup. Each int8 weight maps via the asymmetric
            // KVALUES table.
            uint32_t lo_packed = 0, hi_packed = 0;
            #pragma unroll
            for (int b = 0; b < 4; b++) {
                const int byte = qs_base[j * 4 + b];
                const uint32_t lo = (uint32_t)(uint8_t)KVALUES_IQ4NL[byte & 0xF];
                const uint32_t hi = (uint32_t)(uint8_t)KVALUES_IQ4NL[byte >> 4];
                lo_packed |= lo << (b * 8);
                hi_packed |= hi << (b * 8);
            }
            // xq32[0..3] = quants for weight slot 0..15 in this sub-block;
            // xq32[4..7] = quants for weight slot 16..31. lo_packed dots
            // against the first half, hi_packed against the second.
            idot = __builtin_amdgcn_sdot4((int)lo_packed, xq32[j],     idot, false);
            idot = __builtin_amdgcn_sdot4((int)hi_packed, xq32[j + 4], idot, false);
        }
        acc += dl * dx * (float)idot;
    }

    acc += __shfl_xor(acc, 32);
    acc += __shfl_xor(acc, 16);
    acc += __shfl_xor(acc,  8);
    acc += __shfl_xor(acc,  4);
    acc += __shfl_xor(acc,  2);
    acc += __shfl_xor(acc,  1);

    if (lane == 0) y[row] = acc;
}
