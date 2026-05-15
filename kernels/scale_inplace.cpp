// In-place scalar multiply: x[i] *= s.
//
// Gemma 4 uses this twice: scaling the embedding by √hidden, and
// applying the per-layer `layer_output_scale` scalar.

#include <hip/hip_runtime.h>

extern "C" __global__
void scale_inplace_f32(float* __restrict__ x, unsigned int n, float s)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] *= s;
}
