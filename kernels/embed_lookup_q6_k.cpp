// Embedding lookup straight from Q6_K bytes — gathers and dequants one
// 256-weight super-block per HIP block, one thread per output element.
// Used so token_embd can stay resident in its on-disk Q6_K form
// (~165 MB instead of ~1 GB at fp32 for Qwen 3.5 0.8B).
//
// Reuses the matvec_q6_k dequant math:
//   q6 = ql_nibble | (qh_pair << 4)        ∈ [0, 63]
//   w  = d * scales[sub] * (q6 - 32)
//
// Layout: table is `[vocab, hidden / 256]` Q6_K blocks. We pick the row
// `row_idx` and dequant all `hidden` weights to a contiguous fp32 buffer.

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
void embed_lookup_q6_k_f32(const BlockQ6_K* __restrict__ table,
                           float*           __restrict__ out,
                           unsigned int row_idx,
                           unsigned int hidden)
{
    const unsigned int blocks_per_row = hidden >> 8;   // /256
    const unsigned int blk_idx = blockIdx.x;
    if (blk_idx >= blocks_per_row) return;
    const int i = (int)threadIdx.x;        // 0..255 within this block
    if (i >= 256) return;

    const BlockQ6_K* blk = table + (size_t)row_idx * blocks_per_row + blk_idx;

    // Same per-weight unpack as matvec_q6_k.
    const int chunk    = i >> 7;            // 0 or 1
    const int rest     = i - chunk * 128;
    const int group    = rest >> 5;         // 0..3
    const int l        = rest - group * 32; // 0..31
    const int is       = l >> 4;            // 0 or 1
    const int sc_idx   = chunk * 8 + is + group * 2;
    const int ql_idx   = chunk * 64 + l + (group & 1) * 32;
    const int ql_high  = group >> 1;
    const int qh_idx   = chunk * 32 + l;
    const int qh_shift = group * 2;

    const __half d_h = *reinterpret_cast<const __half*>(&blk->d);
    const float  d   = __half2float(d_h);

    const uint8_t ql_byte = blk->ql[ql_idx];
    const uint8_t ql_nib  = ql_high ? (ql_byte >> 4) : (ql_byte & 0x0F);
    const uint8_t qh_byte = blk->qh[qh_idx];
    const uint8_t qh_pair = (qh_byte >> qh_shift) & 0x3;
    const int q = (int)(ql_nib | (qh_pair << 4)) - 32;

    const unsigned int out_idx = blk_idx * 256 + (unsigned int)i;
    out[out_idx] = d * (float)blk->scales[sc_idx] * (float)q;
}
