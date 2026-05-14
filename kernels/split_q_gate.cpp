// Split the QKV-style interleaved Q+gate buffer into separate Q and gate.
//
//   Input  q_raw:  [n_heads, 2 * head_dim] flat
//                  q_raw[h * 2 * head_dim + d]               = Q[h, d]
//                  q_raw[h * 2 * head_dim + head_dim + d]    = gate[h, d]
//   Output q:      [n_heads, head_dim]
//   Output gate:   [n_heads, head_dim]
//
// One thread per output element.

#include <hip/hip_runtime.h>

extern "C" __global__
void split_q_gate_f32(const float* __restrict__ q_raw,
                      float*       __restrict__ q_out,
                      float*       __restrict__ gate_out,
                      unsigned int n_heads,
                      unsigned int head_dim)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int total = n_heads * head_dim;
    if (i >= total) return;
    const unsigned int h = i / head_dim;
    const unsigned int d = i % head_dim;
    const unsigned int src = h * 2 * head_dim + d;
    q_out[i]    = q_raw[src];
    gate_out[i] = q_raw[src + head_dim];
}
