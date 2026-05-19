// Decode-step GQA attention kernel.
//
//   For each Q head h:
//     kv_h = h / (n_heads / n_kv_heads)
//     scores[t]  = (q_h · K[t, kv_h]) * scaling     for t in [0, total_len)
//     scores      = softmax(scores)
//     out_h[d]    = sum_t scores[t] * V[t, kv_h, d]
//
// One block per Q head. Threads cooperate on:
//   1. loading q into LDS (so each score uses LDS reads, not gmem reads)
//   2. computing scores in parallel across t
//   3. parallel max + exp + sum reduction (online-stable softmax)
//   4. weighted PV sum, parallel across head_dim
//
// Dynamic shared memory layout:
//   q_lds[head_dim] | scores[total_len] | tmp[bs]
// Caller picks `block_size` to match wave-friendly dims (256 here).

#include <hip/hip_runtime.h>

extern "C" __global__
void attn_step_f32(const float* __restrict__ q,         // [n_heads, head_dim]
                   const float* __restrict__ k_cache,   // [max_seq, n_kv_heads, head_dim]
                   const float* __restrict__ v_cache,   // [max_seq, n_kv_heads, head_dim]
                   float*       __restrict__ out,       // [n_heads, head_dim]
                   unsigned int n_heads,
                   unsigned int n_kv_heads,
                   unsigned int head_dim,
                   const unsigned int* __restrict__ pos_ptr,  // decode pos (device)
                   float        scaling)
{
    extern __shared__ float lds[];
    const int h = blockIdx.x;
    if (h >= (int)n_heads) return;
    const int groups = n_heads / n_kv_heads;
    const int kv_h   = h / groups;
    const int tid    = threadIdx.x;
    const int bs     = blockDim.x;
    const unsigned int total_len = *pos_ptr + 1;  // cache positions to attend over

    float* q_lds  = lds;                               // [head_dim]
    float* scores = q_lds  + head_dim;                 // [total_len]
    float* tmp    = scores + total_len;                // [bs]

    // ---- 0. Load q for this head into LDS. ----------------------------------
    for (int i = tid; i < (int)head_dim; i += bs) {
        q_lds[i] = q[(size_t)h * head_dim + i];
    }
    __syncthreads();

    // ---- 1. scores[t] = (q · K[t, kv_h]) * scaling --------------------------
    const size_t kv_row_stride = (size_t)n_kv_heads * head_dim;
    for (int t = tid; t < (int)total_len; t += bs) {
        const float* k_t = k_cache + (size_t)t * kv_row_stride
                                   + (size_t)kv_h * head_dim;
        float acc = 0.0f;
        for (int d = 0; d < (int)head_dim; d++) {
            acc += q_lds[d] * k_t[d];
        }
        scores[t] = acc * scaling;
    }
    __syncthreads();

    // ---- 2a. parallel max over scores ---------------------------------------
    {
        float m = -INFINITY;
        for (int t = tid; t < (int)total_len; t += bs) {
            float v = scores[t];
            if (v > m) m = v;
        }
        tmp[tid] = m;
        __syncthreads();
        for (int s = bs / 2; s > 0; s >>= 1) {
            if (tid < s) {
                float a = tmp[tid], b = tmp[tid + s];
                tmp[tid] = a > b ? a : b;
            }
            __syncthreads();
        }
    }
    const float max_v = tmp[0];
    __syncthreads();

    // ---- 2b. exp + parallel sum --------------------------------------------
    {
        float my_sum = 0.0f;
        for (int t = tid; t < (int)total_len; t += bs) {
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
    const float sum_exp = tmp[0];
    const float inv_sum = 1.0f / sum_exp;
    __syncthreads();

    // ---- 2c. normalize ------------------------------------------------------
    for (int t = tid; t < (int)total_len; t += bs) {
        scores[t] *= inv_sum;
    }
    __syncthreads();

    // ---- 3. PV: out[d] = sum_t scores[t] * V[t, kv_h, d] --------------------
    for (int d = tid; d < (int)head_dim; d += bs) {
        float acc = 0.0f;
        for (int t = 0; t < (int)total_len; t++) {
            const float v_td = v_cache[(size_t)t * kv_row_stride
                                       + (size_t)kv_h * head_dim + d];
            acc += scores[t] * v_td;
        }
        out[(size_t)h * head_dim + d] = acc;
    }
}
