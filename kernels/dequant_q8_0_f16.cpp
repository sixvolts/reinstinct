// Bulk Q8_0 → fp16 dequant. One HIP block per 32-weight block.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

struct __attribute__((packed)) BlockQ8_0 {
    uint16_t d;
    int8_t   qs[32];
};
static_assert(sizeof(BlockQ8_0) == 34, "BlockQ8_0 must be 34 bytes");

extern "C" __global__
void dequant_q8_0_f16(const BlockQ8_0* __restrict__ blocks,
                      __half*          __restrict__ out,
                      unsigned int n_blocks)
{
    const unsigned int blk = blockIdx.x;
    if (blk >= n_blocks) return;
    const int i = (int)threadIdx.x;
    if (i >= 32) return;

    const BlockQ8_0* b = blocks + blk;
    const __half d_h = *reinterpret_cast<const __half*>(&b->d);
    const float  d   = __half2float(d_h);
    out[(size_t)blk * 32 + i] = __float2half(d * (float)b->qs[i]);
}
