// In-place SiLU: x[i] = x[i] * sigmoid(x[i]) = x[i] / (1 + exp(-x[i])).
//
// Used after the GDN block's Conv1D and inside RMSNormGated.

#include <hip/hip_runtime.h>

extern "C" __global__
void silu_inplace_f32(float* __restrict__ x, unsigned int n)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = x[i];
    x[i] = v / (1.0f + __expf(-v));
}
