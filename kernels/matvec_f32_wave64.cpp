// Wave-cooperative fp32 matvec for gfx906.
//
// One 64-thread wavefront per output row. Each lane reads in_dim/64
// elements with a stride of 64, then a single-wave reduction via
// __shfl_xor produces the dot. No shared memory.

#include <hip/hip_runtime.h>
#include "gfx906_dpp.h"

extern "C" __global__
void matvec_f32_wave64(const float* __restrict__ w,
                       const float* __restrict__ x,
                       float*       __restrict__ y,
                       unsigned int in_dim,
                       unsigned int out_dim)
{
    const int row = blockIdx.x;
    if (row >= (int)out_dim) return;
    const int lane = threadIdx.x;        // 0..63

    const float* wrow = w + (size_t)row * (size_t)in_dim;
    float acc = 0.0f;
    for (int i = lane; i < (int)in_dim; i += 64) {
        acc += wrow[i] * x[i];
    }

    acc = wave64_reduce_add_f32(acc);

    if (lane == 0) y[row] = acc;
}
