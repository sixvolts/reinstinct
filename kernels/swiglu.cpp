// SwiGLU activation: out[i] = silu(gate[i]) * up[i]
//   silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
//
// Element-wise; one thread per element. The gate and up vectors come
// out of the FFN's gate_proj and up_proj matvecs respectively; together
// with down_proj this forms a Qwen 3.5 (Llama-style) FFN block.

#include <hip/hip_runtime.h>

extern "C" __global__
void swiglu_mul_f32(const float* __restrict__ gate,
                    const float* __restrict__ up,
                    float*       __restrict__ out,
                    unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = gate[i];
    const float s = g / (1.0f + __expf(-g));   // silu(g)
    out[i] = s * up[i];
}
