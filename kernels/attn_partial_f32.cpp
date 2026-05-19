// FlashDecoding-style split-K decode attention over an f32 KV cache
// (the Qwen full-attention path). f32 analogue of attn_partial_q8.cpp:
// no int8 quantisation, no per-token K/V scales, no sliding window.
//
// grid = (n_heads, n_splits). One workgroup owns one Q head and a
// contiguous slice of [0, total_len); writes a partial (m, l, o). The
// merge kernel (attn_merge.cpp, shared with the q8 path) combines them.
//
// LDS (dynamic): qf[head_dim f32] | scores[chunk f32] | tmp[bs]

#include <hip/hip_runtime.h>

extern "C" __global__
void attn_partial_f32(const float* __restrict__ q,         // [n_heads, head_dim]
                      const float* __restrict__ k_cache,   // [max_seq, n_kv, head_dim]
                      const float* __restrict__ v_cache,
                      float*       __restrict__ o_partial,  // [n_heads, n_splits, head_dim]
                      float*       __restrict__ m_partial,  // [n_heads, n_splits]
                      float*       __restrict__ l_partial,  // [n_heads, n_splits]
                      unsigned int n_heads,
                      unsigned int n_kv_heads,
                      unsigned int head_dim,
                      const unsigned int* __restrict__ pos_ptr,  // decode pos (device)
                      float        scaling,
                      unsigned int n_splits)
{
    extern __shared__ float lds[];
    const int h   = blockIdx.x;
    const int sp  = blockIdx.y;
    if (h >= (int)n_heads) return;
    const int tid = threadIdx.x;
    const int bs  = blockDim.x;
    const int groups = (int)(n_heads / n_kv_heads);
    const int kv_h   = h / groups;
    const int total_len = (int)(*pos_ptr) + 1;

    const int chunk = ((int)total_len + (int)n_splits - 1) / (int)n_splits;
    const int slice_start = sp * chunk;
    int slice_end = sp * chunk + chunk;
    if (slice_end > (int)total_len) slice_end = (int)total_len;
    const int slice_len = slice_end - slice_start;

    float* qf     = lds;                  // [head_dim]
    float* scores = qf + head_dim;        // [chunk]
    float* tmp    = scores + chunk;       // [bs]

    if (slice_len <= 0) {
        for (int d = tid; d < (int)head_dim; d += bs)
            o_partial[((size_t)h * n_splits + sp) * head_dim + d] = 0.0f;
        if (tid == 0) {
            m_partial[(size_t)h * n_splits + sp] = -INFINITY;
            l_partial[(size_t)h * n_splits + sp] = 0.0f;
        }
        return;
    }

    // --- load this head's Q row into LDS ---
    for (int i = tid; i < (int)head_dim; i += bs)
        qf[i] = q[(size_t)h * head_dim + i];
    __syncthreads();

    const size_t kv_row = (size_t)n_kv_heads * head_dim;

    // --- scores: Q·Kᵀ over the slice ---
    for (int i = tid; i < slice_len; i += bs) {
        const int t = slice_start + i;
        const float* k_t = k_cache + (size_t)t * kv_row + (size_t)kv_h * head_dim;
        float acc = 0.0f;
        for (int d = 0; d < (int)head_dim; d++) acc += qf[d] * k_t[d];
        scores[i] = acc * scaling;
    }
    __syncthreads();

    // --- stable softmax over the slice: m, l, exp in place ---
    float m = -INFINITY;
    for (int i = tid; i < slice_len; i += bs) m = fmaxf(m, scores[i]);
    tmp[tid] = m;
    __syncthreads();
    for (int r = bs >> 1; r > 0; r >>= 1) {
        if (tid < r) tmp[tid] = fmaxf(tmp[tid], tmp[tid + r]);
        __syncthreads();
    }
    const float mx = tmp[0];
    __syncthreads();
    float sum = 0.0f;
    for (int i = tid; i < slice_len; i += bs) {
        float e = __expf(scores[i] - mx);
        scores[i] = e;
        sum += e;
    }
    tmp[tid] = sum;
    __syncthreads();
    for (int r = bs >> 1; r > 0; r >>= 1) {
        if (tid < r) tmp[tid] += tmp[tid + r];
        __syncthreads();
    }
    const float l = tmp[0];
    __syncthreads();

    // --- P·V: o[d] = Σ_i scores[i]·V[t][d] ---
    for (int d = tid; d < (int)head_dim; d += bs) {
        float acc = 0.0f;
        for (int i = 0; i < slice_len; i++) {
            const int t = slice_start + i;
            acc += scores[i] * v_cache[(size_t)t * kv_row
                                       + (size_t)kv_h * head_dim + d];
        }
        o_partial[((size_t)h * n_splits + sp) * head_dim + d] = acc;
    }
    if (tid == 0) {
        m_partial[(size_t)h * n_splits + sp] = mx;
        l_partial[(size_t)h * n_splits + sp] = l;
    }
}
