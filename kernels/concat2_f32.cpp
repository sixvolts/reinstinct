// Concatenate two f32 vectors of width N into a 2N-wide output.
// out[0..N) = a; out[N..2N) = b.
//
// One thread per output element; grid = ceil(2N / 256), block = 256.

#include <hip/hip_runtime.h>

extern "C" __global__
void concat2_f32(const float* __restrict__ a,
                 const float* __restrict__ b,
                 float*       __restrict__ out,
                 unsigned int n)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 2u * n) return;
    out[i] = (i < n) ? a[i] : b[i - n];
}
