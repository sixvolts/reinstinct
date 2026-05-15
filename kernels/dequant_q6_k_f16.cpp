// Bulk Q6_K → fp16 dequant. One HIP block per 256-weight super-block.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

struct __attribute__((packed)) BlockQ6_K {
    uint8_t  ql[128];
    uint8_t  qh[64];
    int8_t   scales[16];
    uint16_t d;
};
static_assert(sizeof(BlockQ6_K) == 210, "BlockQ6_K must be 210 bytes");

extern "C" __global__
void dequant_q6_k_f16(const BlockQ6_K* __restrict__ blocks,
                      __half*          __restrict__ out,
                      unsigned int n_blocks)
{
    const unsigned int blk = blockIdx.x;
    if (blk >= n_blocks) return;
    const int i = (int)threadIdx.x;
    if (i >= 256) return;

    const BlockQ6_K* b = blocks + blk;
    const __half d_h = *reinterpret_cast<const __half*>(&b->d);
    const float  d   = __half2float(d_h);

    const int chunk    = i >> 7;
    const int rest     = i - chunk * 128;
    const int group    = rest >> 5;
    const int l        = rest - group * 32;
    const int is       = l >> 4;
    const int sc_idx   = chunk * 8 + is + group * 2;
    const int ql_idx   = chunk * 64 + l + (group & 1) * 32;
    const int ql_high  = group >> 1;
    const int qh_idx   = chunk * 32 + l;
    const int qh_shift = group * 2;

    const uint8_t ql_byte = b->ql[ql_idx];
    const uint8_t ql_nib  = ql_high ? (ql_byte >> 4) : (ql_byte & 0x0F);
    const uint8_t qh_byte = b->qh[qh_idx];
    const uint8_t qh_pair = (qh_byte >> qh_shift) & 0x3;
    const int q = (int)(ql_nib | (qh_pair << 4)) - 32;
    const float w = d * (float)b->scales[sc_idx] * (float)q;

    out[(size_t)blk * 256 + i] = __float2half(w);
}
