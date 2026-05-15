// Embedding lookup straight from Q8_0 bytes — one thread per output
// element. Gemma 4 26B-A4B's token_embd is Q8_0; this keeps it resident
// in on-disk form.
//
// grid = ceil(hidden / 256); block = 256.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

struct __attribute__((packed)) BlockQ8_0 {
    uint16_t d;
    int8_t   qs[32];
};
static_assert(sizeof(BlockQ8_0) == 34, "BlockQ8_0 must be 34 bytes");

extern "C" __global__
void embed_lookup_q8_0_f32(const BlockQ8_0*    __restrict__ table,
                           float*              __restrict__ out,
                           const unsigned int* __restrict__ row_idx_ptr,
                           unsigned int hidden)
{
    const unsigned int row_idx = *row_idx_ptr;
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hidden) return;

    const unsigned int blocks_per_row = hidden >> 5;       // /32
    const unsigned int blk_idx = i >> 5;
    const unsigned int lane    = i & 31;

    const BlockQ8_0* b = table + (size_t)row_idx * blocks_per_row + blk_idx;
    const float d = __half2float(*reinterpret_cast<const __half*>(&b->d));
    out[i] = d * (float)b->qs[lane];
}
