// Wave-cooperative IQ4_XS matvec for gfx906.

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

__device__ static const int8_t KVALUES_IQ4NL_W64[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
       1,   13,  25,  38,  53,  69,  89, 113,
};

extern "C" __global__
void matvec_iq4_xs_wave64_f32(const BlockIQ4_XS* __restrict__ w_blocks,
                              const float*       __restrict__ x,
                              float*             __restrict__ y,
                              unsigned int in_dim,
                              unsigned int out_dim)
{
    const int row = blockIdx.x;
    if (row >= (int)out_dim) return;
    const int lane = threadIdx.x;

    const unsigned int n_blocks    = in_dim >> 8;
    const unsigned int n_subblocks = in_dim >> 5;
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

        const uint8_t* qs_base = blk->qs + ib * 16;
        const float*   x_base  = x + (size_t)sb * 32;

        float partial = 0.0f;
        #pragma unroll
        for (int l = 0; l < 16; l++) {
            const int lo = qs_base[l] & 0x0F;
            const int hi = qs_base[l] >> 4;
            partial += (float)KVALUES_IQ4NL_W64[lo] * x_base[l]
                     + (float)KVALUES_IQ4NL_W64[hi] * x_base[l + 16];
        }
        acc += dl * partial;
    }

    acc += __shfl_xor(acc, 32);
    acc += __shfl_xor(acc, 16);
    acc += __shfl_xor(acc,  8);
    acc += __shfl_xor(acc,  4);
    acc += __shfl_xor(acc,  2);
    acc += __shfl_xor(acc,  1);

    if (lane == 0) y[row] = acc;
}
