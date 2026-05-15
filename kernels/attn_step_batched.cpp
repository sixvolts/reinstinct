// Batched causal GQA attention for prefill.
//
// Processes N query rows in one launch. Query row r is the token at
// sequence position base_pos + r, and attends causally to cache
// positions [0, base_pos + r] — i.e. total_len = base_pos + r + 1.
//
// The K/V cache must already hold all base_pos + N entries (caller
// pushes the N new K/V rows before launching).
//
// grid: (n_heads, n_rows). One block per (q-head, query-row); within
// the block the same load-q-to-LDS / parallel-scores / stable-softmax /
// parallel-PV structure as attn_step.cpp.
//
// Dynamic LDS: q_lds[head_dim] | scores[max_total_len] | tmp[bs].
// The launcher sizes scores for the largest row (base_pos + n_rows).

#include <hip/hip_runtime.h>

extern "C" __global__
void attn_step_batched_f32(const float* __restrict__ q,         // [n_rows, n_heads, head_dim]
                           const float* __restrict__ k_cache,   // [max_seq, n_kv_heads, head_dim]
                           const float* __restrict__ v_cache,
                           float*       __restrict__ out,       // [n_rows, n_heads, head_dim]
                           unsigned int n_heads,
                           unsigned int n_kv_heads,
                           unsigned int head_dim,
                           unsigned int base_pos,
                           unsigned int n_rows,
                           float        scaling)
{
    extern __shared__ float lds[];
    const int h   = blockIdx.x;
    const int row = blockIdx.y;
    if (h >= (int)n_heads || row >= (int)n_rows) return;
    const int groups = n_heads / n_kv_heads;
    const int kv_h   = h / groups;
    const int tid    = threadIdx.x;
    const int bs     = blockDim.x;

    const int total_len = (int)base_pos + row + 1;   // causal horizon

    float* q_lds  = lds;
    float* scores = q_lds  + head_dim;
    float* tmp    = scores + total_len;

    const float* q_head = q + ((size_t)row * n_heads + h) * head_dim;
    for (int i = tid; i < (int)head_dim; i += bs) {
        q_lds[i] = q_head[i];
    }
    __syncthreads();

    const size_t kv_row_stride = (size_t)n_kv_heads * head_dim;
    for (int t = tid; t < total_len; t += bs) {
        const float* k_t = k_cache + (size_t)t * kv_row_stride
                                   + (size_t)kv_h * head_dim;
        float acc = 0.0f;
        for (int d = 0; d < (int)head_dim; d++) {
            acc += q_lds[d] * k_t[d];
        }
        scores[t] = acc * scaling;
    }
    __syncthreads();

    // parallel max
    {
        float m = -INFINITY;
        for (int t = tid; t < total_len; t += bs) {
            float v = scores[t];
            if (v > m) m = v;
        }
        tmp[tid] = m;
        __syncthreads();
        for (int s = bs / 2; s > 0; s >>= 1) {
            if (tid < s) { float a = tmp[tid], b = tmp[tid + s]; tmp[tid] = a > b ? a : b; }
            __syncthreads();
        }
    }
    const float max_v = tmp[0];
    __syncthreads();

    // exp + parallel sum
    {
        float my_sum = 0.0f;
        for (int t = tid; t < total_len; t += bs) {
            float e = __expf(scores[t] - max_v);
            scores[t] = e;
            my_sum += e;
        }
        tmp[tid] = my_sum;
        __syncthreads();
        for (int s = bs / 2; s > 0; s >>= 1) {
            if (tid < s) tmp[tid] += tmp[tid + s];
            __syncthreads();
        }
    }
    const float inv_sum = 1.0f / tmp[0];
    __syncthreads();

    for (int t = tid; t < total_len; t += bs) {
        scores[t] *= inv_sum;
    }
    __syncthreads();

    float* out_head = out + ((size_t)row * n_heads + h) * head_dim;
    for (int d = tid; d < (int)head_dim; d += bs) {
        float acc = 0.0f;
        for (int t = 0; t < total_len; t++) {
            const float v_td = v_cache[(size_t)t * kv_row_stride
                                       + (size_t)kv_h * head_dim + d];
            acc += scores[t] * v_td;
        }
        out_head[d] = acc;
    }
}
