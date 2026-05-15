// GeGLU activation: out[i] = gelu(gate[i]) * up[i]
//   gelu (tanh approx) = 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))
//
// Gemma 4's FFN gate fusion — same shape as swiglu.cpp but GELU
// instead of SiLU.

#include <hip/hip_runtime.h>

extern "C" __global__
void geglu_mul_f32(const float* __restrict__ gate,
                   const float* __restrict__ up,
                   float*       __restrict__ out,
                   unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float x = gate[i];
    const float k = 0.7978845608f;   // sqrt(2/pi)
    const float inner = k * (x + 0.044715f * x * x * x);
    const float g = 0.5f * x * (1.0f + tanhf(inner));
    out[i] = g * up[i];
}
