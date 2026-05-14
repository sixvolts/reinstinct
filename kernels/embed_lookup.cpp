// Embedding lookup: out[i] = table[row_idx * hidden + i] for i in [0, hidden).
//
// Single-token gather. Used at the start of every forward pass. For prefill
// of multiple tokens we'll launch a 2D grid (one row per blockIdx.y).

#include <hip/hip_runtime.h>

extern "C" __global__
void embed_lookup_f32(const float* __restrict__ table,
                      float*       __restrict__ out,
                      unsigned int row_idx,
                      unsigned int hidden)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hidden) return;
    // size_t cast: vocab*hidden can exceed 2^31 floats for big vocabs.
    out[i] = table[(size_t)row_idx * hidden + i];
}
