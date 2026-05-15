// Partial RoPE (rotary position embedding), in-place across multiple heads.
//
//   Input layout:  x[h * head_dim + i]  for h in [0, n_heads), i in [0, head_dim)
//   Rotation: only the first rotary_dim of each head are rotated; the
//   remaining (head_dim - rotary_dim) elements pass through unchanged
//   (Qwen 3.5 0.8B rotates 64 of 256).
//
// Half-split convention (matches Llama / HF):
//   half = rotary_dim / 2
//   y[i]        = x[i]        * cos[i]        - x[i + half] * sin[i]
//   y[i + half] = x[i + half] * cos[i + half] + x[i]        * sin[i + half]
//
// Cos/Sin tables are precomputed for all positions:
//   cos[pos * rotary_dim + i], same shape for sin.
//
// Each thread handles one (head, lane) pair where lane in [0, half).
// It reads both x[i] and x[i+half] into registers and writes both back,
// so there's no cross-thread data hazard.

#include <hip/hip_runtime.h>

extern "C" __global__
void rope_apply_f32(float*       __restrict__ x,        // [n_heads, head_dim]
                    const float* __restrict__ cos,      // [max_seq, rotary_dim]
                    const float* __restrict__ sin,      // [max_seq, rotary_dim]
                    unsigned int head_dim,
                    unsigned int rotary_dim,
                    unsigned int n_heads,
                    const unsigned int* __restrict__ pos_ptr)
{
    const unsigned int half = rotary_dim >> 1;
    const unsigned int h    = blockIdx.y;
    if (h >= n_heads) return;
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= half) return;

    float* head      = x + (size_t)h * head_dim;
    const unsigned int pos = *pos_ptr;
    const float* cr  = cos + (size_t)pos * rotary_dim;
    const float* sr  = sin + (size_t)pos * rotary_dim;

    const float a = head[i];
    const float b = head[i + half];
    head[i]        = a * cr[i]        - b * sr[i];
    head[i + half] = b * cr[i + half] + a * sr[i + half];
}
