// Decode-step GQA attention with an optional sliding window.
//
// Same structure as attn_step.cpp — one block per Q head, load q to
// LDS, parallel scores, stable softmax, parallel PV — but the attended
// range is [lo, total_len) where
//   lo = window > 0 ? max(0, total_len - window) : 0
//
// `window == 0` means full causal attention. Gemma 4's sliding layers
// pass window = 1024; its full layers pass 0.
//
// Dynamic LDS: q_lds[head_dim] | scores[window_len] | tmp[bs].
// The launcher sizes `scores` for the worst case (min(total_len,
// window) when window > 0, else total_len).

#include <hip/hip_runtime.h>

extern "C" __global__
void attn_step_window_f32(const float* __restrict__ q,         // [n_heads, head_dim]
                          const float* __restrict__ k_cache,   // [max_seq, n_kv_heads, head_dim]
                          const float* __restrict__ v_cache,
                          float*       __restrict__ out,       // [n_heads, head_dim]
                          unsigned int n_heads,
                          unsigned int n_kv_heads,
                          unsigned int head_dim,
                          unsigned int total_len,
                          unsigned int window,                  // 0 = full causal
                          float        scaling)
{
    extern __shared__ float lds[];
    const int h = blockIdx.x;
    if (h >= (int)n_heads) return;
    const int groups = n_heads / n_kv_heads;
    const int kv_h   = h / groups;
    const int tid    = threadIdx.x;
    const int bs     = blockDim.x;

    const int lo = (window > 0 && total_len > window) ? (int)(total_len - window) : 0;
    const int win_len = (int)total_len - lo;

    float* q_lds  = lds;
    float* scores = q_lds  + head_dim;     // [win_len]
    float* tmp    = scores + win_len;      // [bs]

    for (int i = tid; i < (int)head_dim; i += bs) {
        q_lds[i] = q[(size_t)h * head_dim + i];
    }
    __syncthreads();

    const size_t kv_row_stride = (size_t)n_kv_heads * head_dim;
    for (int s = tid; s < win_len; s += bs) {
        const int t = lo + s;
        const float* k_t = k_cache + (size_t)t * kv_row_stride
                                   + (size_t)kv_h * head_dim;
        float acc = 0.0f;
        for (int d = 0; d < (int)head_dim; d++) {
            acc += q_lds[d] * k_t[d];
        }
        scores[s] = acc * scaling;
    }
    __syncthreads();

    // parallel max
    {
        float m = -INFINITY;
        for (int s = tid; s < win_len; s += bs) {
            float v = scores[s];
            if (v > m) m = v;
        }
        tmp[tid] = m;
        __syncthreads();
        for (int r = bs / 2; r > 0; r >>= 1) {
            if (tid < r) { float a = tmp[tid], b = tmp[tid + r]; tmp[tid] = a > b ? a : b; }
            __syncthreads();
        }
    }
    const float max_v = tmp[0];
    __syncthreads();

    // exp + parallel sum
    {
        float my_sum = 0.0f;
        for (int s = tid; s < win_len; s += bs) {
            float e = __expf(scores[s] - max_v);
            scores[s] = e;
            my_sum += e;
        }
        tmp[tid] = my_sum;
        __syncthreads();
        for (int r = bs / 2; r > 0; r >>= 1) {
            if (tid < r) tmp[tid] += tmp[tid + r];
            __syncthreads();
        }
    }
    const float inv_sum = 1.0f / tmp[0];
    __syncthreads();

    for (int s = tid; s < win_len; s += bs) {
        scores[s] *= inv_sum;
    }
    __syncthreads();

    for (int d = tid; d < (int)head_dim; d += bs) {
        float acc = 0.0f;
        for (int s = 0; s < win_len; s++) {
            const int t = lo + s;
            const float v_td = v_cache[(size_t)t * kv_row_stride
                                       + (size_t)kv_h * head_dim + d];
            acc += scores[s] * v_td;
        }
        out[(size_t)h * head_dim + d] = acc;
    }
}
