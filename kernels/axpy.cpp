// Scaled accumulate / store. Used to sum the routed-expert outputs of a
// MoE layer (a = routing weight · per-expert scale). The first expert
// uses scaled_set (overwrites garbage in the fresh accumulator); the
// rest use axpy.

#include <hip/hip_runtime.h>

extern "C" __global__
void axpy_f32(float* __restrict__ y, const float* __restrict__ x,
              float a, unsigned int n)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += a * x[i];
}

extern "C" __global__
void scaled_set_f32(float* __restrict__ y, const float* __restrict__ x,
                    float a, unsigned int n)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i];
}
