// Wave-cooperative Q6_K matvec for gfx906 (Wave64).
//
// Same dequant math as matvec_q6_k.cpp, but:
//   - One *wave* (64 threads) per 2 output rows (ROWS=2). Each lane
//     reads one sub-block of x once and accumulates a dot for BOTH
//     rows — the activation read is shared across rows, the weight
//     read varies. Halves the launch count vs the 1-row layout,
//     amortising launch overhead on the launch-bound output_proj /
//     lm_head shape (in=1024, out=248320 → 124160 blocks instead
//     of 248320).
//   - Reduction uses DPP across the 64 lanes of the wavefront —
//     no shared memory, no __syncthreads.
//   - Each thread owns ONE sub-block (16 weights) per iteration. For
//     in_dim=1024 there are exactly 64 sub-blocks per row, so the work
//     is one sub-block per lane with no inner stride loop. For larger
//     in_dim the lanes stride.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include "gfx906_dpp.h"

#define ROWS 2

struct __attribute__((packed)) BlockQ6_K {
    uint8_t  ql[128];
    uint8_t  qh[64];
    int8_t   scales[16];
    uint16_t d;
};
static_assert(sizeof(BlockQ6_K) == 210, "BlockQ6_K must be 210 bytes");

extern "C" __global__
void matvec_q6_k_wave64_f32(const BlockQ6_K* __restrict__ w_blocks,
                            const float*     __restrict__ x,
                            float*           __restrict__ y,
                            unsigned int in_dim,
                            unsigned int out_dim)
{
    const int row0 = blockIdx.x * ROWS;
    if (row0 >= (int)out_dim) return;
    const int lane = threadIdx.x;       // 0..63

    const unsigned int n_blocks    = in_dim >> 8;   // /256
    const unsigned int n_subblocks = in_dim >> 4;   // /16

    float acc[ROWS];
    #pragma unroll
    for (int r = 0; r < ROWS; r++) acc[r] = 0.0f;

    for (int sb = lane; sb < (int)n_subblocks; sb += 64) {
        const int blk_idx     = sb >> 4;        // /16 sub-blocks per Q6_K block
        const int sb_in_blk   = sb & 15;
        const int chunk       = sb_in_blk >> 3;
        const int sb_in_chunk = sb_in_blk & 7;
        const int group       = sb_in_chunk >> 1;
        const int is          = sb_in_chunk & 1;
        const int sc_idx      = chunk * 8 + is + group * 2;
        const int l_start     = is * 16;
        const int qh_shift    = group * 2;
        const int ql_off_blk  = chunk * 64 + (group & 1) * 32;
        const int ql_high     = group >> 1;
        const int qh_off_blk  = chunk * 32;

        // Activation is shared across both rows — load once.
        const float* x_base = x + (size_t)sb * 16;
        float xs[16];
        #pragma unroll
        for (int li = 0; li < 16; li++) xs[li] = x_base[li];

        #pragma unroll
        for (int r = 0; r < ROWS; r++) {
            const int row = row0 + r;
            if (row >= (int)out_dim) continue;
            const BlockQ6_K* blk = w_blocks + (size_t)row * n_blocks + blk_idx;
            const __half d_h = *reinterpret_cast<const __half*>(&blk->d);
            const float  d   = __half2float(d_h);
            const float  ds  = d * (float)blk->scales[sc_idx];

            float partial = 0.0f;
            #pragma unroll
            for (int li = 0; li < 16; li++) {
                const int l = l_start + li;
                const uint8_t ql_byte = blk->ql[ql_off_blk + l];
                const uint8_t ql_nib  = ql_high ? (ql_byte >> 4) : (ql_byte & 0x0F);
                const uint8_t qh_byte = blk->qh[qh_off_blk + l];
                const uint8_t qh_pair = (qh_byte >> qh_shift) & 0x3;
                const int q = (int)(ql_nib | (qh_pair << 4)) - 32;
                partial += (float)q * xs[li];
            }
            acc[r] += ds * partial;
        }
    }

    #pragma unroll
    for (int r = 0; r < ROWS; r++) {
        float a = wave64_reduce_add_f32(acc[r]);
        if (lane == 0 && (row0 + r) < (int)out_dim) y[row0 + r] = a;
    }
}
