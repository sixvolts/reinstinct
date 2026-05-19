// Embedding lookup straight from Q5_K bytes — one HIP block per
// 256-weight super-block, one thread per output element. Gemma 4's
// token_embd is Q5_K; this keeps it resident in on-disk form.
//
// Two entry points: the single-token decode lookup and a batched
// prefill lookup (grid.y = token, ids read from a device array) — the
// batched form replaces a per-token launch+sync loop.

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
void gsm_q5k_emb(int j, const uint8_t* q, uint8_t& sc, uint8_t& m) {
    if (j < 4) { sc = q[j] & 63; m = q[j + 4] & 63; }
    else {
        sc = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        m  = (q[j + 4] >> 4)   | ((q[j]     >> 6) << 4);
    }
}

// Dequantise output element `i` (0..255) of one Q5_K super-block.
__device__ __forceinline__
float q5k_emb_elem(const BlockQ5_K* b, int i) {
    const float d    = __half2float(*reinterpret_cast<const __half*>(&b->d));
    const float dmin = __half2float(*reinterpret_cast<const __half*>(&b->dmin));

    const int chunk   = i >> 6;
    const int pos     = i & 63;
    const int is_high = pos >> 5;
    const int l       = pos & 31;
    const int sub     = chunk * 2 + is_high;
    const int qh_mask = 1 << (chunk * 2 + is_high);

    uint8_t sc, m;
    gsm_q5k_emb(sub, b->scales, sc, m);

    const uint8_t qs_byte = b->qs[chunk * 32 + l];
    const int nib = is_high ? (qs_byte >> 4) : (qs_byte & 0x0F);
    const int q5  = nib + ((b->qh[l] & qh_mask) ? 16 : 0);
    return d * (float)sc * (float)q5 - dmin * (float)m;
}

extern "C" __global__
void embed_lookup_q5_k_f32(const BlockQ5_K*    __restrict__ table,
                           float*              __restrict__ out,
                           const unsigned int* __restrict__ row_idx_ptr,
                           unsigned int hidden)
{
    const unsigned int row_idx = *row_idx_ptr;
    const unsigned int blocks_per_row = hidden >> 8;   // /256
    const unsigned int blk_idx = blockIdx.x;
    if (blk_idx >= blocks_per_row) return;
    const int i = (int)threadIdx.x;
    if (i >= 256) return;

    const BlockQ5_K* b = table + (size_t)row_idx * blocks_per_row + blk_idx;
    out[blk_idx * 256 + (unsigned int)i] = q5k_emb_elem(b, i);
}

// Batched: grid.y selects the token, ids come from a device array, and
// each token's `hidden` outputs are written to row `blockIdx.y`.
extern "C" __global__
void embed_lookup_q5_k_batched_f32(const BlockQ5_K*    __restrict__ table,
                                   float*              __restrict__ out,
                                   const unsigned int* __restrict__ row_idx,
                                   unsigned int hidden)
{
    const unsigned int r = blockIdx.y;
    const unsigned int blocks_per_row = hidden >> 8;
    const unsigned int blk_idx = blockIdx.x;
    if (blk_idx >= blocks_per_row) return;
    const int i = (int)threadIdx.x;
    if (i >= 256) return;

    const BlockQ5_K* b = table + (size_t)row_idx[r] * blocks_per_row + blk_idx;
    out[(size_t)r * hidden + blk_idx * 256 + (unsigned int)i] = q5k_emb_elem(b, i);
}
