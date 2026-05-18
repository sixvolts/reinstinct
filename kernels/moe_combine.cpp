// Combine the routed-expert outputs of a MoE layer:
//
//   out[i] = sum_j  weights[j] · down_exps_s[ids[j]] · experts[j·h + i]
//
// `experts` is [n_tok, n_used, hidden]; `weights` / `ids` are
// [n_tok, n_used]; `down_exps_s` the per-expert down-output scalar,
// indexed by the device-resident expert id. One thread per hidden
// element — no host round-trip anywhere in the MoE path.
//
// grid = (ceil(hidden / 256), n_tok); block = 256. grid.y batches
// prefill tokens; decode launches grid.y = 1.

#include <hip/hip_runtime.h>

extern "C" __global__
void moe_combine_f32(const float* __restrict__ experts,
                     const int*   __restrict__ ids,
                     const float* __restrict__ weights,
                     const float* __restrict__ down_exps_s,
                     float*       __restrict__ out,
                     unsigned int hidden,
                     unsigned int n_used)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hidden) return;
    // One grid.y slice per token.
    const size_t tok = blockIdx.y;
    experts += tok * n_used * hidden;
    ids     += tok * n_used;
    weights += tok * n_used;
    out     += tok * hidden;
    float acc = 0.0f;
    for (unsigned int j = 0; j < n_used; j++) {
        const float s = weights[j] * down_exps_s[ids[j]];
        acc += s * experts[(size_t)j * hidden + i];
    }
    out[i] = acc;
}
