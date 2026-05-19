// Merge the partial (m, l, o) outputs of the split-K decode attention
// (attn_partial_q8.cpp / attn_partial_f32.cpp) into the final per-head
// output. One workgroup per Q head.
//
//   global_m = max_s m_partial[s]
//   denom    = Σ_s exp(m_partial[s] - global_m)·l_partial[s]
//   out[d]   = (Σ_s exp(m_partial[s] - global_m)·o_partial[s][d]) / denom
//
// Empty splits carry m = -INF, l = 0, o = 0, so exp(m - global_m) = 0
// and they contribute nothing.

#include <hip/hip_runtime.h>

extern "C" __global__
void attn_merge_f32(const float* __restrict__ o_partial,  // [n_heads, n_splits, head_dim]
                    const float* __restrict__ m_partial,  // [n_heads, n_splits]
                    const float* __restrict__ l_partial,  // [n_heads, n_splits]
                    float*       __restrict__ out,        // [n_heads, head_dim]
                    unsigned int head_dim,
                    unsigned int n_splits)
{
    const int h   = blockIdx.x;
    const int tid = threadIdx.x;
    const int bs  = blockDim.x;
    const float* mp = m_partial + (size_t)h * n_splits;
    const float* lp = l_partial + (size_t)h * n_splits;
    const float* op = o_partial + (size_t)h * n_splits * head_dim;

    // global max + softmax denominator (cheap, computed per thread).
    float gm = -INFINITY;
    for (int s = 0; s < (int)n_splits; s++) gm = fmaxf(gm, mp[s]);
    float gl = 0.0f;
    for (int s = 0; s < (int)n_splits; s++) gl += __expf(mp[s] - gm) * lp[s];
    const float inv = gl > 0.0f ? 1.0f / gl : 0.0f;

    for (int d = tid; d < (int)head_dim; d += bs) {
        float acc = 0.0f;
        for (int s = 0; s < (int)n_splits; s++)
            acc += __expf(mp[s] - gm) * op[(size_t)s * head_dim + d];
        out[(size_t)h * head_dim + d] = acc * inv;
    }
}
