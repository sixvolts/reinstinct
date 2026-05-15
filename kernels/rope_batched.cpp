// Partial RoPE over a batch of N positions.
//
// Same half-split rotation as rope.cpp, but with an extra grid axis for
// the batch row. Row r holds the token at sequence position
// base_pos + r, so each row rotates with a different (cos, sin) slice.
//
//   Input layout:  x[(r * n_heads + h) * head_dim + i]
//   grid: (ceil(rotary_dim/2 / block), n_heads, n_rows)

#include <hip/hip_runtime.h>

extern "C" __global__
void rope_apply_batched_f32(float*       __restrict__ x,    // [n_rows, n_heads, head_dim]
                            const float* __restrict__ cos,  // [max_seq, rotary_dim]
                            const float* __restrict__ sin,
                            unsigned int head_dim,
                            unsigned int rotary_dim,
                            unsigned int n_heads,
                            unsigned int base_pos)
{
    const unsigned int half = rotary_dim >> 1;
    const unsigned int row  = blockIdx.z;
    const unsigned int h    = blockIdx.y;
    if (h >= n_heads) return;
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= half) return;

    const unsigned int pos = base_pos + row;
    float* head     = x + ((size_t)row * n_heads + h) * head_dim;
    const float* cr = cos + (size_t)pos * rotary_dim;
    const float* sr = sin + (size_t)pos * rotary_dim;

    const float a = head[i];
    const float b = head[i + half];
    head[i]        = a * cr[i]        - b * sr[i];
    head[i + half] = b * cr[i + half] + a * sr[i + half];
}
