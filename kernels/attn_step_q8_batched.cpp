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
// One workgroup per (head, q_row). Same quantise -> dp4a-scores ->
// softmax -> PV pattern as attn_step_q8, but the keys stream through in
// chunks of blockDim.x with an online softmax (running max m, running
// sum l, accumulators rescaled by exp(m_old - m_new) per chunk), so
// LDS is fixed at qi[head_dim] | p[bs] | red[4] whatever the context
// length. (It used to hold every score of the window: a graph captured
// at one position overran LDS when replayed at a later one, and full
// layers capped out at ~15.8K positions in 64 KB.)
//
// grid = (n_heads, n_q_rows); block = 256 (4 wave64s); head_dim <= 512
// and a multiple of 4.

#include <hip/hip_runtime.h>
#include <stdint.h>
#include "gfx906_dpp.h"

#define AQB_BS 256
#define AQB_WAVES (AQB_BS / 64)
#define AQB_DPT 2              // head_dim / block, rounded up: <= 512 dims

// Block-wide max / sum over AQB_BS threads; every thread gets the result.
__device__ __forceinline__ float aqb_block_max(float v, float* red) {
    v = wave64_reduce_max_f32(v);
    const int w = threadIdx.x >> 6;
    if ((threadIdx.x & 63) == 0) red[w] = v;
    __syncthreads();
    float r = red[0];
    #pragma unroll
    for (int i = 1; i < AQB_WAVES; i++) r = fmaxf(r, red[i]);
    __syncthreads();
    return r;
}
__device__ __forceinline__ float aqb_block_sum(float v, float* red) {
    v = wave64_reduce_add_f32(v);
    const int w = threadIdx.x >> 6;
    if ((threadIdx.x & 63) == 0) red[w] = v;
    __syncthreads();
    float r = red[0];
    #pragma unroll
    for (int i = 1; i < AQB_WAVES; i++) r += red[i];
    __syncthreads();
    return r;
}

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
    __shared__ int   qi32[512 / 4];
    __shared__ float p[AQB_BS];
    __shared__ float red[AQB_WAVES];
    const int h     = blockIdx.x;
    const int q_row = blockIdx.y;
    if (h >= (int)n_heads || q_row >= (int)n_q_rows) return;
    const int groups = n_heads / n_kv_heads;
    const int kv_h   = h / groups;
    const int tid    = threadIdx.x;

    const int total_len = (int)(base_pos + (unsigned int)q_row + 1u);
    const int lo = (window > 0 && total_len > (int)window) ? total_len - (int)window : 0;

    // --- quantise THIS (q_row, h) row of Q to int8 in LDS ---
    const float* qh = q + ((size_t)q_row * n_heads + (size_t)h) * head_dim;
    float amax = 0.0f;
    for (int i = tid; i < (int)head_dim; i += AQB_BS) amax = fmaxf(amax, fabsf(qh[i]));
    const float q_amax = aqb_block_max(amax, red);
    const float dq     = q_amax > 0.0f ? q_amax / 127.0f : 1.0f;
    const float q_inv  = q_amax > 0.0f ? 127.0f / q_amax : 0.0f;
    signed char* qi = reinterpret_cast<signed char*>(qi32);
    for (int i = tid; i < (int)head_dim; i += AQB_BS) {
        int v = (int)rintf(qh[i] * q_inv);
        qi[i] = (signed char)max(-127, min(127, v));
    }
    __syncthreads();

    const int    n4     = head_dim >> 2;
    const size_t kv_row = (size_t)n_kv_heads * head_dim;
    const float  sdq    = dq * scaling;

    float m = -INFINITY, l = 0.0f;
    float acc[AQB_DPT];
    #pragma unroll
    for (int j = 0; j < AQB_DPT; j++) acc[j] = 0.0f;

    for (int c0 = lo; c0 < total_len; c0 += AQB_BS) {
        const int n = min(AQB_BS, total_len - c0);
        // --- this chunk's scores: one key per thread ---
        float sc = -INFINITY;
        if (tid < n) {
            const int t = c0 + tid;
            const int* k32 = reinterpret_cast<const int*>(
                k_cache + (size_t)t * kv_row + (size_t)kv_h * head_dim);
            int idot = 0;
            for (int g = 0; g < n4; g++)
                idot = __builtin_amdgcn_sdot4(qi32[g], k32[g], idot, false);
            sc = sdq * k_scale[(size_t)t * n_kv_heads + kv_h] * (float)idot;
        }
        // --- online softmax update ---
        const float m_new = fmaxf(m, aqb_block_max(sc, red));
        const float alpha = __expf(m - m_new);          // 0 on the first chunk
        const float e     = tid < n ? __expf(sc - m_new) : 0.0f;
        // P carries the per-token V scale; l sums the plain weights.
        p[tid] = tid < n ? e * v_scale[(size_t)(c0 + tid) * n_kv_heads + kv_h] : 0.0f;
        l = l * alpha + aqb_block_sum(e, red);          // its barrier publishes p[]
        m = m_new;
        // --- P.V over the chunk, dims tid, tid + bs ---
        #pragma unroll
        for (int j = 0; j < AQB_DPT; j++) {
            const int d = tid + j * AQB_BS;
            if (d < (int)head_dim) {
                const signed char* vp = v_cache + (size_t)c0 * kv_row + (size_t)kv_h * head_dim + d;
                float a = 0.0f;
                for (int s = 0; s < n; s++) a += p[s] * (float)vp[(size_t)s * kv_row];
                acc[j] = acc[j] * alpha + a;
            }
        }
        __syncthreads();                                // p[] is rewritten next chunk
    }

    const float inv_l = 1.0f / l;
    #pragma unroll
    for (int j = 0; j < AQB_DPT; j++) {
        const int d = tid + j * AQB_BS;
        if (d < (int)head_dim)
            out[((size_t)q_row * n_heads + (size_t)h) * head_dim + d] = acc[j] * inv_l;
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
