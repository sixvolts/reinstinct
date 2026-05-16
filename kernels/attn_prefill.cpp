// Batched causal attention for prefill — P query tokens in one launch.
//
// Query token p attends to keys [lo, p] (causal), where
//   lo = window > 0 ? max(0, p+1-window) : 0   (Gemma sliding window).
// Q/K/V are this layer's freshly-projected f32 activations for the P
// prompt tokens — no KV cache involved; prefill computes the whole
// triangle at once.
//
//   Q : [P, n_heads,    head_dim]
//   K : [P, n_kv_heads, head_dim]
//   V : [P, n_kv_heads, head_dim]
//   out: [P, n_heads,   head_dim]
//
// One workgroup per (head, query). The score row [lo,p] fits in LDS, so
// it's a plain three-pass softmax (no flash/online-softmax needed).
//
// LDS: q_lds[head_dim] | scores[P] | tmp[bs].  grid = (n_heads, P).

#include <hip/hip_runtime.h>

extern "C" __global__
void attn_prefill_f32(const float* __restrict__ q,    // [P, n_heads, head_dim]
                      const float* __restrict__ k,    // [P, n_kv,    head_dim]
                      const float* __restrict__ v,
                      float*       __restrict__ out,
                      unsigned int n_heads,
                      unsigned int n_kv_heads,
                      unsigned int head_dim,
                      unsigned int window,            // 0 = full causal
                      float        scaling)
{
    extern __shared__ float lds[];
    const int h = blockIdx.x;
    const int p = blockIdx.y;                  // this query token
    if (h >= (int)n_heads) return;
    const int groups = n_heads / n_kv_heads;
    const int kv_h   = h / groups;
    const int tid    = threadIdx.x;
    const int bs     = blockDim.x;

    const int total = p + 1;
    const int lo = (window > 0 && total > (int)window) ? (total - (int)window) : 0;
    const int win_len = total - lo;

    float* q_lds  = lds;
    float* scores = q_lds  + head_dim;         // [win_len]
    float* tmp    = scores + win_len;          // [bs]

    for (int i = tid; i < (int)head_dim; i += bs)
        q_lds[i] = q[((size_t)p * n_heads + h) * head_dim + i];
    __syncthreads();

    const size_t kv_row = (size_t)n_kv_heads * head_dim;
    for (int s = tid; s < win_len; s += bs) {
        const int t = lo + s;
        const float* k_t = k + (size_t)t * kv_row + (size_t)kv_h * head_dim;
        float acc = 0.0f;
        for (int d = 0; d < (int)head_dim; d++) acc += q_lds[d] * k_t[d];
        scores[s] = acc * scaling;
    }
    __syncthreads();

    {
        float m = -INFINITY;
        for (int s = tid; s < win_len; s += bs) m = fmaxf(m, scores[s]);
        tmp[tid] = m;
        __syncthreads();
        for (int r = bs >> 1; r > 0; r >>= 1) {
            if (tid < r) tmp[tid] = fmaxf(tmp[tid], tmp[tid + r]);
            __syncthreads();
        }
    }
    const float max_v = tmp[0];
    __syncthreads();
    {
        float sum = 0.0f;
        for (int s = tid; s < win_len; s += bs) {
            float e = __expf(scores[s] - max_v);
            scores[s] = e;
            sum += e;
        }
        tmp[tid] = sum;
        __syncthreads();
        for (int r = bs >> 1; r > 0; r >>= 1) {
            if (tid < r) tmp[tid] += tmp[tid + r];
            __syncthreads();
        }
    }
    const float inv_sum = 1.0f / tmp[0];
    __syncthreads();

    for (int d = tid; d < (int)head_dim; d += bs) {
        float acc = 0.0f;
        for (int s = 0; s < win_len; s++) {
            const int t = lo + s;
            const float v_td = v[(size_t)t * kv_row + (size_t)kv_h * head_dim + d];
            acc += scores[s] * v_td;
        }
        out[((size_t)p * n_heads + h) * head_dim + d] = acc * inv_sum;
    }
}
