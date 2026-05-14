// In-place vector add: x[i] += y[i].
//
// Used for the residual sum in transformer blocks
// (`hidden ← hidden + sublayer_output`).

#include <hip/hip_runtime.h>

extern "C" __global__
void add_inplace_f32(float*       __restrict__ x,    // [n] in/out
                     const float* __restrict__ y,    // [n]
                     unsigned int n)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] += y[i];
}
