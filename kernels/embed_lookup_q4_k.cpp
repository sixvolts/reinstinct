// Embedding lookup straight from Q4_K bytes — gathers and dequants one
// 256-weight super-block per HIP block, one thread per output element.
// Same row-gather as embed_lookup_q6_k, with the Q4_K block math from
// dequant_q4_k_f16. Lets token_embd stay resident in its on-disk Q4_K
// form (some GGUFs, e.g. Qwen 3.5 27B, quantize the embedding table).
//
// Layout: table is `[vocab, hidden / 256]` Q4_K blocks. Pick row
// `row_idx` and dequant all `hidden` weights to a contiguous fp32 buffer.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

struct __attribute__((packed)) BlockQ4_K {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[12];
    uint8_t  qs[128];
};
static_assert(sizeof(BlockQ4_K) == 144, "BlockQ4_K must be 144 bytes");

__device__ __forceinline__
void gsm_q4k(int j, const uint8_t* q, uint8_t& sc, uint8_t& m) {
    if (j < 4) { sc = q[j] & 63; m = q[j + 4] & 63; }
    else {
        sc = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        m  = (q[j + 4] >> 4)   | ((q[j]     >> 6) << 4);
    }
}

extern "C" __global__
void embed_lookup_q4_k_f32(const BlockQ4_K* __restrict__ table,
                           float*           __restrict__ out,
                           unsigned int row_idx,
                           unsigned int hidden)
{
    const unsigned int blocks_per_row = hidden >> 8;   // /256
    const unsigned int blk_idx = blockIdx.x;
    if (blk_idx >= blocks_per_row) return;
    const int i = (int)threadIdx.x;        // 0..255 within this block
    if (i >= 256) return;

    const BlockQ4_K* b = table + (size_t)row_idx * blocks_per_row + blk_idx;
    const __half d_h  = *reinterpret_cast<const __half*>(&b->d);
    const __half dm_h = *reinterpret_cast<const __half*>(&b->dmin);
    const float  d    = __half2float(d_h);
    const float  dmin = __half2float(dm_h);

    const int chunk   = i >> 6;
    const int pos     = i & 63;
    const int is_high = pos >> 5;
    const int l       = pos & 31;
    const int sub     = chunk * 2 + is_high;

    uint8_t sc, m;
    gsm_q4k(sub, b->scales, sc, m);

    const uint8_t qs_byte = b->qs[chunk * 32 + l];
    const int nib = is_high ? (qs_byte >> 4) : (qs_byte & 0x0F);
    const float w = d * (float)sc * (float)nib - dmin * (float)m;

    out[blk_idx * 256 + (unsigned int)i] = w;
}
