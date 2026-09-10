// Embedding lookup straight from Q4_0 bytes — one thread per output
// element. Gemma 4's QAT GGUFs quantize token_embd to Q4_0 like every
// other tensor; this keeps it resident in on-disk form rather than
// widening it at load (756 MB → 1.4 GB on the 31B).
//
// Byte k of a block's 16-byte payload holds weight k in its low nibble
// and weight k+16 in its high nibble; the value is d·(q − 8).
//
// grid = ceil(hidden / 256); block = 256.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

struct __attribute__((packed)) BlockQ4_0 {
    uint16_t d;
    uint8_t  qs[16];
};
static_assert(sizeof(BlockQ4_0) == 18, "BlockQ4_0 must be 18 bytes");

__device__ __forceinline__
float q4_0_weight(const BlockQ4_0* __restrict__ b, unsigned int lane)
{
    const float d = __half2float(*reinterpret_cast<const __half*>(&b->d));
    const unsigned int is_high = lane >> 4;    // 0 for 0..15, 1 for 16..31
    const unsigned int k       = lane & 15;
    const uint8_t byte = b->qs[k];
    const int nib = is_high ? (byte >> 4) : (byte & 0x0F);
    return d * (float)(nib - 8);
}

extern "C" __global__
void embed_lookup_q4_0_f32(const BlockQ4_0*    __restrict__ table,
                           float*              __restrict__ out,
                           const unsigned int* __restrict__ row_idx_ptr,
                           unsigned int hidden)
{
    const unsigned int row_idx = *row_idx_ptr;
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hidden) return;

    const unsigned int blocks_per_row = hidden >> 5;       // /32
    const BlockQ4_0* b = table + (size_t)row_idx * blocks_per_row + (i >> 5);
    out[i] = q4_0_weight(b, i & 31);
}

// Batched: grid.y selects the token, ids come from a device array, and
// each token's `hidden` outputs are written to row `blockIdx.y`.
extern "C" __global__
void embed_lookup_q4_0_batched_f32(const BlockQ4_0*    __restrict__ table,
                                   float*              __restrict__ out,
                                   const unsigned int* __restrict__ row_idx,
                                   unsigned int hidden)
{
    const unsigned int r = blockIdx.y;
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hidden) return;

    const unsigned int blocks_per_row = hidden >> 5;       // /32
    const BlockQ4_0* b = table + (size_t)row_idx[r] * blocks_per_row + (i >> 5);
    out[(size_t)r * hidden + i] = q4_0_weight(b, i & 31);
}
