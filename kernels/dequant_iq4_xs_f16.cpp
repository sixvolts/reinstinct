// Bulk IQ4_XS → fp16 dequant. One HIP block per 256-weight super-block.

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

__device__ static const int8_t KVALUES_IQ4NL_DQ[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
       1,   13,  25,  38,  53,  69,  89, 113,
};

extern "C" __global__
void dequant_iq4_xs_f16(const BlockIQ4_XS* __restrict__ blocks,
                        __half*            __restrict__ out,
                        unsigned int n_blocks)
{
    const unsigned int blk = blockIdx.x;
    if (blk >= n_blocks) return;
    const int i = (int)threadIdx.x;
    if (i >= 256) return;

    const BlockIQ4_XS* b = blocks + blk;
    const __half d_h = *reinterpret_cast<const __half*>(&b->d);
    const float  d   = __half2float(d_h);

    const int ib      = i >> 5;          // sub-block 0..7
    const int pos     = i & 31;
    const int is_high = pos >> 4;        // 0 = low nibble, 1 = high
    const int l       = pos & 15;

    const int ls_lo = (b->scales_l[ib >> 1] >> (4 * (ib & 1))) & 0x0F;
    const int ls_hi = (b->scales_h >> (2 * ib)) & 0x3;
    const int ls    = ls_lo | (ls_hi << 4);
    const float dl  = d * (float)(ls - 32);

    const uint8_t qs_byte = b->qs[ib * 16 + l];
    const int nib = is_high ? (qs_byte >> 4) : (qs_byte & 0x0F);
    const float w = dl * (float)KVALUES_IQ4NL_DQ[nib];

    out[(size_t)blk * 256 + i] = __float2half(w);
}
