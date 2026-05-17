// Embedding lookup from a Q8_0 token-embedding table — value-form row
// index (the qwen35 runtime passes the token by value; the gemma path
// uses the device-pointer `embed_lookup_q8_0` for its captured graph).
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
void embed_lookup_q8_0_v_f32(const BlockQ8_0* __restrict__ table,
                             float*           __restrict__ out,
                             unsigned int row_idx,
                             unsigned int hidden)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hidden) return;
    const unsigned int blocks_per_row = hidden >> 5;       // /32
    const BlockQ8_0* b = table + (size_t)row_idx * blocks_per_row + (i >> 5);
    const float d = __half2float(*reinterpret_cast<const __half*>(&b->d));
    out[i] = d * (float)b->qs[i & 31];
}
