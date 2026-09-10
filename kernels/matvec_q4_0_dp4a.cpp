// Q4_0 matvec straight from the on-disk 18-byte block layout, for
// weights that are not repacked at load — chiefly a Q4_0 token_embd
// doubling as the tied LM head, which must keep its on-disk form so the
// embedding lookup can gather rows from it.
//
// Two entry points:
//   * `matvec_q4_0_dp4a_f32`   — int8 dp4a against a pre-quantised
//     activation. grid = ceil(out_dim/ROWS); block = 64.
//   * `matvec_q4_0_wave64_f32` — fp32 reference for the
//     REINSTINCT_GEMMA_NO_DP4A A/B path. grid = out_dim; block = 64.
//
// Nibble order matches the repacked kernel: byte k carries weight k in
// its low nibble and weight k+16 in its high, so uint32 j pairs with
// activation group j (low) and j+4 (high). Value is d·(q − 8); the −8
// offset is applied against the summed *quantized* activations rather
// than BlockQ8::xsum, so both terms live in the same quantized domain —
// see matvec_q4_0_repacked.cpp for why that is worth 40× accuracy.
//
// `qs` sits at offset 2 in an 18-byte struct, so it is only 2-byte
// aligned — the quants are assembled from uint16 pairs rather than read
// as uint32, the same dodge matvec_q8_0_dp4a uses for its 34-byte stride.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include "gfx906_dpp.h"

#define ROWS 2

struct __attribute__((packed)) BlockQ4_0 {
    uint16_t d;
    uint8_t  qs[16];
};
static_assert(sizeof(BlockQ4_0) == 18, "BlockQ4_0 must be 18 bytes");

struct __attribute__((packed)) BlockQ8 {
    float  d;
    float  xsum;
    int8_t qs[32];
};
static_assert(sizeof(BlockQ8) == 40, "BlockQ8 must be 40 bytes");

extern "C" __global__
void matvec_q4_0_dp4a_f32(const BlockQ4_0* __restrict__ w_blocks,
                          const BlockQ8*   __restrict__ xq,
                          float*           __restrict__ y,
                          unsigned int in_dim,
                          unsigned int out_dim)
{
    const int row0 = blockIdx.x * ROWS;
    const int lane = threadIdx.x;
    const unsigned int n_blocks = in_dim >> 5;       // 32 weights per block

    float acc[ROWS];
    #pragma unroll
    for (int r = 0; r < ROWS; r++) acc[r] = 0.0f;

    for (int sb = lane; sb < (int)n_blocks; sb += 64) {
        const BlockQ8* xb = xq + sb;
        const float dx   = xb->d;
        const int*  xq32 = reinterpret_cast<const int*>(xb->qs);
        int xqsum = 0;
        #pragma unroll
        for (int g = 0; g < 8; g++)
            xqsum = __builtin_amdgcn_sdot4(0x01010101, xq32[g], xqsum, false);

        #pragma unroll
        for (int r = 0; r < ROWS; r++) {
            const int row = row0 + r;
            if (row >= (int)out_dim) continue;
            const BlockQ4_0* blk = w_blocks + (size_t)row * n_blocks + sb;
            const float dw = __half2float(*reinterpret_cast<const __half*>(&blk->d));
            const uint16_t* qs16 = reinterpret_cast<const uint16_t*>(blk->qs);

            int idot = 0;
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                const uint32_t wq = (uint32_t)qs16[2*j] | ((uint32_t)qs16[2*j+1] << 16);
                idot = __builtin_amdgcn_sdot4((int)( wq       & 0x0F0F0F0Fu),
                                              xq32[j],     idot, false);
                idot = __builtin_amdgcn_sdot4((int)((wq >> 4) & 0x0F0F0F0Fu),
                                              xq32[j + 4], idot, false);
            }
            acc[r] += dw * dx * (float)(idot - 8 * xqsum);
        }
    }

    #pragma unroll
    for (int r = 0; r < ROWS; r++) {
        float a = acc[r];
        a = wave64_reduce_add_f32(a);
        if (lane == 0 && (row0 + r) < (int)out_dim) y[row0 + r] = a;
    }
}

extern "C" __global__
void matvec_q4_0_wave64_f32(const BlockQ4_0* __restrict__ w_blocks,
                            const float*     __restrict__ x,
                            float*           __restrict__ y,
                            unsigned int in_dim,
                            unsigned int out_dim)
{
    const int row = blockIdx.x;
    if (row >= (int)out_dim) return;
    const int lane = threadIdx.x;

    const unsigned int n_blocks = in_dim >> 5;
    const BlockQ4_0* row_blocks = w_blocks + (size_t)row * n_blocks;

    float acc = 0.0f;
    for (int b = lane; b < (int)n_blocks; b += 64) {
        const BlockQ4_0* blk = row_blocks + b;
        const float d = __half2float(*reinterpret_cast<const __half*>(&blk->d));
        const float* xb = x + (size_t)b * 32;
        float partial = 0.0f;
        #pragma unroll
        for (int k = 0; k < 16; k++) {
            const uint8_t byte = blk->qs[k];
            partial += (float)((int)(byte & 0x0F) - 8) * xb[k];
            partial += (float)((int)(byte >> 4)   - 8) * xb[k + 16];
        }
        acc += d * partial;
    }

    acc = wave64_reduce_add_f32(acc);

    if (lane == 0) y[row] = acc;
}
