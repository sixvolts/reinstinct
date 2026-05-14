// In-place output gating: x[i] *= sigmoid(gate[i]).
//
// Used at the end of Qwen 3.5's full-attention block, where the per-head
// `gate` channel pulled out of the QKV-style projection modulates the
// attention output before the final o_proj.
//
//   sigmoid(g) = 1 / (1 + exp(-g))

#include <hip/hip_runtime.h>

extern "C" __global__
void sigmoid_mul_inplace_f32(float*       __restrict__ x,        // [n] in/out
                             const float* __restrict__ gate,     // [n]
                             unsigned int n)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = gate[i];
    const float s = 1.0f / (1.0f + __expf(-g));
    x[i] *= s;
}
