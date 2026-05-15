// Final-logit soft-cap, in place:  y[i] = cap · tanh(y[i] / cap)
//
// Gemma 4 bounds its output logits to ±cap (cap = 30). Applied once to
// the vocab-length logit vector after the output projection.

#include <hip/hip_runtime.h>

extern "C" __global__
void logit_softcap_f32(float* __restrict__ y, unsigned int n, float cap)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = cap * tanhf(y[i] / cap);
}
