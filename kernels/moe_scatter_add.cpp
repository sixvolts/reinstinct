// Scatter-add the routed-expert outputs of a grouped MoE prefill back
// into the per-token accumulator.
//
//   dst[idx[j]] += weights[j] · down_exps_s[expert_id] · src[j]
//
// src is [n_tok, hidden] — expert `expert_id`'s down outputs for the
// n_tok tokens routed to it; idx[j] is the global token index. Each
// expert's launch touches a disjoint token set, and the per-expert
// launches are stream-ordered, so a plain += needs no atomics.
//
// grid = (ceil(hidden/256), n_tok); block = 256.

#include <hip/hip_runtime.h>

extern "C" __global__
void moe_scatter_add_f32(const float* __restrict__ src,        // [n_tok, hidden]
                         const int*   __restrict__ idx,        // [n_tok]
                         const float* __restrict__ weights,    // [n_tok]
                         const float* __restrict__ down_exps_s,
                         unsigned int expert_id,
                         float*       __restrict__ dst,        // [P, hidden]
                         unsigned int hidden,
                         unsigned int n_tok)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int j = blockIdx.y;
    if (i >= hidden || j >= n_tok) return;
    const float s = weights[j] * down_exps_s[expert_id];
    dst[(size_t)idx[j] * hidden + i] += s * src[(size_t)j * hidden + i];
}
