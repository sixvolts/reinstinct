// Bulk Q4_0 → fp16 dequant. One HIP block per 32-weight block.
//
// Byte k of the 16-byte nibble payload holds weight k in its low nibble
// and weight k+16 in its high nibble; the value is d·(q − 8).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

struct __attribute__((packed)) BlockQ4_0 {
    uint16_t d;
    uint8_t  qs[16];
};
static_assert(sizeof(BlockQ4_0) == 18, "BlockQ4_0 must be 18 bytes");

extern "C" __global__
void dequant_q4_0_f16(const BlockQ4_0* __restrict__ blocks,
                      __half*          __restrict__ out,
                      unsigned int n_blocks)
{
    const unsigned int blk = blockIdx.x;
    if (blk >= n_blocks) return;
    const int i = (int)threadIdx.x;
    if (i >= 32) return;

    const BlockQ4_0* b = blocks + blk;
    const __half d_h = *reinterpret_cast<const __half*>(&b->d);
    const float  d   = __half2float(d_h);

    const int is_high = i >> 4;
    const int k       = i & 15;
    const uint8_t byte = b->qs[k];
    const int nib = is_high ? (byte >> 4) : (byte & 0x0F);

    out[(size_t)blk * 32 + i] = __float2half(d * (float)(nib - 8));
}
