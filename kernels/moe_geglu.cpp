// Batched GeGLU for the MoE expert FFN. `gu` is the fused gate_up
// output [n_slot, 2·ff_exp] (gate = first half, up = second); writes
// act[n_slot, ff_exp] = gelu(gate) · up. One launch over all slots.
//
// grid = ceil(n_slot·ff_exp / 256); block = 256.

#include <hip/hip_runtime.h>

__device__ __forceinline__ float gelu_tanh(float x) {
    const float k = 0.7978845608028654f;          // sqrt(2/pi)
    const float c = x + 0.044715f * x * x * x;
    return 0.5f * x * (1.0f + tanhf(k * c));
}

extern "C" __global__
void moe_geglu_f32(const float* __restrict__ gu,
                   float*       __restrict__ act,
                   unsigned int ff_exp,
                   unsigned int n_slot)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_slot * ff_exp) return;
    const unsigned int slot = i / ff_exp;
    const unsigned int k    = i % ff_exp;
    const float* g = gu + (size_t)slot * 2u * ff_exp;
    act[i] = gelu_tanh(g[k]) * g[ff_exp + k];
}
