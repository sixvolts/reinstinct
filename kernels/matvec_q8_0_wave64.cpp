// Wave-cooperative Q8_0 matvec for gfx906.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include "gfx906_dpp.h"

struct __attribute__((packed)) BlockQ8_0 {
    uint16_t d;
    int8_t   qs[32];
};
static_assert(sizeof(BlockQ8_0) == 34, "BlockQ8_0 must be 34 bytes");

extern "C" __global__
void matvec_q8_0_wave64_f32(const BlockQ8_0* __restrict__ w_blocks,
                            const float*     __restrict__ x,
                            float*           __restrict__ y,
                            unsigned int in_dim,
                            unsigned int out_dim)
{
    const int row = blockIdx.x;
    if (row >= (int)out_dim) return;
    const int lane = threadIdx.x;

    const unsigned int n_blocks = in_dim >> 5;  // /32 (Q8_0 block = 32 weights)
    const BlockQ8_0* row_blocks = w_blocks + (size_t)row * n_blocks;

    float acc = 0.0f;
    for (int b = lane; b < (int)n_blocks; b += 64) {
        const BlockQ8_0* blk = row_blocks + b;
        const __half d_h = *reinterpret_cast<const __half*>(&blk->d);
        const float  d   = __half2float(d_h);
        const float* xb = x + (size_t)b * 32;
        float partial = 0.0f;
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            partial += (float)blk->qs[i] * xb[i];
        }
        acc += d * partial;
    }

    acc = wave64_reduce_add_f32(acc);

    if (lane == 0) y[row] = acc;
}
