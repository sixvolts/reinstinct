// Row-blocked Q6_K matvec for gfx906 — same idea as the Q4_K rowblock.
//
// ROWS=2 output rows per wavefront, lane = one 16-weight sub-block.
// Q6_K blocks are 210 bytes, so the stride is only 2-byte aligned —
// the ql / qh bytes are read as uint16 (not uint32) to stay aligned
// while still halving the load-instruction count vs byte-at-a-time.
//
// grid = ceil(out_dim / ROWS); block = 64 (one wavefront).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#define ROWS 2

struct __attribute__((packed)) BlockQ6_K {
    uint8_t  ql[128];
    uint8_t  qh[64];
    int8_t   scales[16];
    uint16_t d;
};
static_assert(sizeof(BlockQ6_K) == 210, "BlockQ6_K must be 210 bytes");

extern "C" __global__
void matvec_q6_k_rowblock_f32(const BlockQ6_K* __restrict__ w_blocks,
                              const float*     __restrict__ x,
                              float*           __restrict__ y,
                              unsigned int in_dim,
                              unsigned int out_dim)
{
    const int row0 = blockIdx.x * ROWS;
    const int lane = threadIdx.x;
    const unsigned int n_blocks    = in_dim >> 8;
    const unsigned int n_subblocks = in_dim >> 4;

    float acc[ROWS];
    #pragma unroll
    for (int r = 0; r < ROWS; r++) acc[r] = 0.0f;

    for (int sb = lane; sb < (int)n_subblocks; sb += 64) {
        const int blk_idx     = sb >> 4;
        const int sb_in_blk   = sb & 15;
        const int chunk       = sb_in_blk >> 3;
        const int sb_in_chunk = sb_in_blk & 7;
        const int group       = sb_in_chunk >> 1;
        const int is          = sb_in_chunk & 1;
        const int sc_idx      = chunk * 8 + is + group * 2;
        const int l_start     = is * 16;
        const int qh_shift    = group * 2;
        const int ql_off      = chunk * 64 + (group & 1) * 32 + l_start;
        const int ql_high     = group >> 1;
        const int qh_off      = chunk * 32 + l_start;
        const float* x_base   = x + (size_t)sb * 16;

        #pragma unroll
        for (int r = 0; r < ROWS; r++) {
            const int row = row0 + r;
            if (row >= (int)out_dim) continue;
            const BlockQ6_K* blk = w_blocks + (size_t)row * n_blocks + blk_idx;

            const float d  = __half2float(*reinterpret_cast<const __half*>(&blk->d));
            const float ds = d * (float)blk->scales[sc_idx];

            // ql/qh sub-block bases are even — read as uint16.
            const uint16_t* ql16 =
                reinterpret_cast<const uint16_t*>(blk->ql + ql_off);
            const uint16_t* qh16 =
                reinterpret_cast<const uint16_t*>(blk->qh + qh_off);

            float partial = 0.0f;
            #pragma unroll
            for (int g = 0; g < 8; g++) {
                const uint32_t uq = ql16[g];
                const uint32_t uh = qh16[g];
                const float2   xv = *reinterpret_cast<const float2*>(x_base + g * 2);
                #pragma unroll
                for (int b = 0; b < 2; b++) {
                    const uint32_t qlb = (uq >> (b * 8)) & 0xFF;
                    const uint32_t qhb = (uh >> (b * 8)) & 0xFF;
                    const uint32_t ql_nib = ql_high ? (qlb >> 4) : (qlb & 0x0F);
                    const uint32_t qh_pair = (qhb >> qh_shift) & 0x3;
                    const int q = (int)(ql_nib | (qh_pair << 4)) - 32;
                    partial += (float)q * (&xv.x)[b];
                }
            }
            acc[r] += ds * partial;
        }
    }

    #pragma unroll
    for (int r = 0; r < ROWS; r++) {
        float a = acc[r];
        a += __shfl_xor(a, 32);
        a += __shfl_xor(a, 16);
        a += __shfl_xor(a,  8);
        a += __shfl_xor(a,  4);
        a += __shfl_xor(a,  2);
        a += __shfl_xor(a,  1);
        if (lane == 0 && (row0 + r) < (int)out_dim) y[row0 + r] = a;
    }
}
