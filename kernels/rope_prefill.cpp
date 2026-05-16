// Batched RoPE for prefill — applies the rotary embedding to all P
// prompt tokens in one launch. Token p sits at position p (a prefill
// always starts from position 0), so the position is just blockIdx.z;
// no device position buffer is needed.
//
//   x : [P, n_heads, head_dim]   (in place)
//   grid = (ceil(rotary_dim/2 / block), n_heads, P)
//
// Half-split convention, matching rope_dpos.cpp.

#include <hip/hip_runtime.h>

extern "C" __global__
void rope_prefill_f32(float*       __restrict__ x,
                      const float* __restrict__ cos,    // [max_seq, rotary_dim]
                      const float* __restrict__ sin,
                      unsigned int head_dim,
                      unsigned int rotary_dim,
                      unsigned int n_heads)
{
    const unsigned int half = rotary_dim >> 1;
    const unsigned int h = blockIdx.y;
    const unsigned int p = blockIdx.z;          // token = position
    if (h >= n_heads) return;
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= half) return;

    float* head     = x + ((size_t)p * n_heads + h) * head_dim;
    const float* cr = cos + (size_t)p * rotary_dim;
    const float* sr = sin + (size_t)p * rotary_dim;

    const float a = head[i];
    const float b = head[i + half];
    head[i]        = a * cr[i]        - b * sr[i];
    head[i + half] = b * cr[i + half] + a * sr[i + half];
}
