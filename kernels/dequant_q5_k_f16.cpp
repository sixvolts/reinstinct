// Bulk Q5_K → fp16 dequant. One HIP block per 256-weight super-block.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

struct __attribute__((packed)) BlockQ5_K {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[12];
    uint8_t  qh[32];
    uint8_t  qs[128];
};
static_assert(sizeof(BlockQ5_K) == 176, "BlockQ5_K must be 176 bytes");

__device__ __forceinline__
void gsm_q5k(int j, const uint8_t* q, uint8_t& sc, uint8_t& m) {
    if (j < 4) { sc = q[j] & 63; m = q[j + 4] & 63; }
    else {
        sc = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        m  = (q[j + 4] >> 4)   | ((q[j]     >> 6) << 4);
    }
}

extern "C" __global__
void dequant_q5_k_f16(const BlockQ5_K* __restrict__ blocks,
                      __half*          __restrict__ out,
                      unsigned int n_blocks)
{
    const unsigned int blk = blockIdx.x;
    if (blk >= n_blocks) return;
    const int i = (int)threadIdx.x;
    if (i >= 256) return;

    const BlockQ5_K* b = blocks + blk;
    const __half d_h  = *reinterpret_cast<const __half*>(&b->d);
    const __half dm_h = *reinterpret_cast<const __half*>(&b->dmin);
    const float  d    = __half2float(d_h);
    const float  dmin = __half2float(dm_h);

    const int chunk   = i >> 6;
    const int pos     = i & 63;
    const int is_high = pos >> 5;
    const int l       = pos & 31;
    const int sub     = chunk * 2 + is_high;
    const int qh_mask = 1 << (chunk * 2 + is_high);

    uint8_t sc, m;
    gsm_q5k(sub, b->scales, sc, m);

    const uint8_t qs_byte = b->qs[chunk * 32 + l];
    const int nib = is_high ? (qs_byte >> 4) : (qs_byte & 0x0F);
    const int q5  = nib + ((b->qh[l] & qh_mask) ? 16 : 0);
    const float w = d * (float)sc * (float)q5 - dmin * (float)m;

    out[(size_t)blk * 256 + i] = __float2half(w);
}
