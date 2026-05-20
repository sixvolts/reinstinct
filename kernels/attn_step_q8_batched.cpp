// Batched-decode GQA attention over an int8 KV cache — K queries in
// one launch, each with its own causal range.
//
// This is the batched analogue of attn_step_q8: instead of one query
// per launch (decode-step), it processes K queries at positions
// [base_pos, base_pos+n_q_rows), each attending over its OWN total_len
// = base_pos + q_row + 1 (causal). Used by the MTP spec-decode verify
// to score the K drafted candidate tokens in a single forward.
//
//   Q : f32 [n_q_rows, n_heads, head_dim]
//   K : int8 cache [max_seq, n_kv, head_dim] populated up through slot
//                                            base_pos+n_q_rows-1.
//   V : same shape as K, with per-(slot, head) f32 scales.
//   out: f32 [n_q_rows, n_heads, head_dim].
//
//   For each query row q_row and head h:
//     total_len = base_pos + q_row + 1
//     lo        = window>0 ? max(0,total_len-window) : 0
//     scores[t] = dq · dk_t · <Qi · Ki_t> · scaling   for t in [lo, total_len)
//     scores    = softmax(scores)
//     out_h[d]  = Σ scores[t] · dv_t · V[t, kv_h, d]
//
// One workgroup per (head, q_row). Body mirrors attn_step_q8 — same
// quantise→dp4a-scores→stable-softmax→PV-accumulate pattern; only the
// indexing differs (per-query Q row, per-query total_len, per-query out
// row). LDS = qi[head_dim int8] | scores[max_win f32] | tmp[bs f32].
//
// grid = (n_heads, n_q_rows); block = 256.

#include <hip/hip_runtime.h>
#include <stdint.h>

__device__ __forceinline__
void attn_step_q8_batched_body(const float*       __restrict__ q,
                               const signed char* __restrict__ k_cache,
                               const float*       __restrict__ k_scale,
                               const signed char* __restrict__ v_cache,
                               const float*       __restrict__ v_scale,
                               float*             __restrict__ out,
                               unsigned int n_heads,
                               unsigned int n_kv_heads,
                               unsigned int head_dim,
                               unsigned int base_pos,
                               unsigned int n_q_rows,
                               unsigned int window,
                               float        scaling)
{
    extern __shared__ float lds[];
    const int h     = blockIdx.x;
    const int q_row = blockIdx.y;
    if (h >= (int)n_heads || q_row >= (int)n_q_rows) return;
    const int groups = n_heads / n_kv_heads;
    const int kv_h   = h / groups;
    const int tid    = threadIdx.x;
    const int bs     = blockDim.x;

    const unsigned int total_len = base_pos + (unsigned int)q_row + 1u;
    const int lo = (window > 0 && total_len > window) ? (int)(total_len - window) : 0;
    const int win_len = (int)total_len - lo;

    // LDS layout: qi (int8, head_dim) then the f32 regions.
    signed char* qi    = reinterpret_cast<signed char*>(lds);
    float*       scores = reinterpret_cast<float*>(qi + head_dim);
    float*       tmp    = scores + win_len;

    // --- quantise THIS (q_row, h) row of Q to int8 in LDS ---
    const float* qh = q + ((size_t)q_row * n_heads + (size_t)h) * head_dim;
    float amax = 0.0f;
    for (int i = tid; i < (int)head_dim; i += bs) amax = fmaxf(amax, fabsf(qh[i]));
    tmp[tid] = amax;
    __syncthreads();
    for (int r = bs >> 1; r > 0; r >>= 1) {
        if (tid < r) tmp[tid] = fmaxf(tmp[tid], tmp[tid + r]);
        __syncthreads();
    }
    const float q_amax = tmp[0];
    const float dq     = q_amax > 0.0f ? q_amax / 127.0f : 1.0f;
    const float q_inv  = q_amax > 0.0f ? 127.0f / q_amax : 0.0f;
    __syncthreads();
    for (int i = tid; i < (int)head_dim; i += bs) {
        int v = (int)rintf(qh[i] * q_inv);
        qi[i] = (signed char)max(-127, min(127, v));
    }
    __syncthreads();

    const int*   qi32 = reinterpret_cast<const int*>(qi);
    const int    n4   = head_dim >> 2;
    const size_t kv_row = (size_t)n_kv_heads * head_dim;

    // --- scores: int8 dp4a Q·Kᵀ over the causal window ---
    for (int s = tid; s < win_len; s += bs) {
        const int t = lo + s;
        const signed char* k_t = k_cache + (size_t)t * kv_row + (size_t)kv_h * head_dim;
        const int* k32 = reinterpret_cast<const int*>(k_t);
        int idot = 0;
        for (int g = 0; g < n4; g++)
            idot = __builtin_amdgcn_sdot4(qi32[g], k32[g], idot, false);
        const float dk = k_scale[(size_t)t * n_kv_heads + kv_h];
        scores[s] = dq * dk * (float)idot * scaling;
    }
    __syncthreads();

    // --- stable softmax ---
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
    // fold 1/sum and the per-token V scale into the probability.
    for (int s = tid; s < win_len; s += bs) {
        const int t = lo + s;
        scores[s] *= inv_sum * v_scale[(size_t)t * n_kv_heads + kv_h];
    }
    __syncthreads();

    // --- P·V: f32 accumulate over dequantised int8 V ---
    for (int d = tid; d < (int)head_dim; d += bs) {
        float acc = 0.0f;
        for (int s = 0; s < win_len; s++) {
            const int t = lo + s;
            const signed char v_td =
                v_cache[(size_t)t * kv_row + (size_t)kv_h * head_dim + d];
            acc += scores[s] * (float)v_td;
        }
        out[((size_t)q_row * n_heads + (size_t)h) * head_dim + d] = acc;
    }
}

extern "C" __global__
void attn_step_q8_batched_f32(const float*       __restrict__ q,
                              const signed char* __restrict__ k_cache,
                              const float*       __restrict__ k_scale,
                              const signed char* __restrict__ v_cache,
                              const float*       __restrict__ v_scale,
                              float*             __restrict__ out,
                              unsigned int n_heads,
                              unsigned int n_kv_heads,
                              unsigned int head_dim,
                              unsigned int base_pos,
                              unsigned int n_q_rows,
                              unsigned int window,
                              float        scaling)
{
    attn_step_q8_batched_body(q, k_cache, k_scale, v_cache, v_scale, out,
                              n_heads, n_kv_heads, head_dim, base_pos,
                              n_q_rows, window, scaling);
}

// Variant that reads `base_pos` from a device-resident uint32 — used
// by verify_forward when captured into a HIP graph (see the comment in
// kv_quant_prefill_offset_f32).
extern "C" __global__
void attn_step_q8_batched_offset_f32(const float*       __restrict__ q,
                                     const signed char* __restrict__ k_cache,
                                     const float*       __restrict__ k_scale,
                                     const signed char* __restrict__ v_cache,
                                     const float*       __restrict__ v_scale,
                                     float*             __restrict__ out,
                                     unsigned int n_heads,
                                     unsigned int n_kv_heads,
                                     unsigned int head_dim,
                                     const unsigned int* __restrict__ base_pos_ptr,
                                     unsigned int n_q_rows,
                                     unsigned int window,
                                     float        scaling)
{
    attn_step_q8_batched_body(q, k_cache, k_scale, v_cache, v_scale, out,
                              n_heads, n_kv_heads, head_dim, *base_pos_ptr,
                              n_q_rows, window, scaling);
}
