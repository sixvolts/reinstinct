// Decode-step GQA attention over an int8 KV cache — SageAttention-style
// without flash/online-softmax: the score row fits in LDS, so it's a
// plain three-pass softmax. The Q·Kᵀ scores are an int8 dp4a GEMM
// (v_dot4_i32_i8); the P·V is f32 accumulate over dequantised V.
//
//   Q : f32, quantised per head to int8 in LDS (scale dq)
//   K : int8 cache + per-(token,head) f32 scale
//   score_t = dq · dk_t · <Qi · Ki_t> · scaling
//   V : int8 cache + per-(token,head) f32 scale; out = Σ p_t·dv_t·Vi_t
//
// One workgroup per Q head. Sliding window: attended range [lo,
// total_len), lo = window>0 ? max(0,total_len-window) : 0.
//
// LDS: qi[head_dim int8, 4-aligned] | scores[win_len f32] | tmp[bs f32].
// grid = n_heads; block = 256.

#include <hip/hip_runtime.h>
#include <stdint.h>

extern "C" __global__
void attn_step_q8_f32(const float*       __restrict__ q,        // [n_heads, head_dim]
                      const signed char* __restrict__ k_cache,  // [max_seq, n_kv, head_dim]
                      const float*       __restrict__ k_scale,  // [max_seq, n_kv]
                      const signed char* __restrict__ v_cache,
                      const float*       __restrict__ v_scale,
                      float*             __restrict__ out,      // [n_heads, head_dim]
                      unsigned int n_heads,
                      unsigned int n_kv_heads,
                      unsigned int head_dim,
                      const unsigned int* __restrict__ pos_ptr,
                      unsigned int window,
                      float        scaling)
{
    extern __shared__ float lds[];
    const unsigned int total_len = *pos_ptr + 1;
    const int h = blockIdx.x;
    if (h >= (int)n_heads) return;
    const int groups = n_heads / n_kv_heads;
    const int kv_h   = h / groups;
    const int tid    = threadIdx.x;
    const int bs     = blockDim.x;

    const int lo = (window > 0 && total_len > window) ? (int)(total_len - window) : 0;
    const int win_len = (int)total_len - lo;

    // LDS layout: qi (int8, head_dim) then the f32 regions.
    signed char* qi    = reinterpret_cast<signed char*>(lds);
    float*       scores = reinterpret_cast<float*>(qi + head_dim);   // [win_len]
    float*       tmp    = scores + win_len;                          // [bs]

    // --- quantise this head's Q row to int8 (per-head scale) ---
    const float* qh = q + (size_t)h * head_dim;
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

    const int*   qi32 = reinterpret_cast<const int*>(qi);   // head_dim/4 int32
    const int    n4   = head_dim >> 2;
    const size_t kv_row = (size_t)n_kv_heads * head_dim;

    // --- scores: int8 dp4a Q·Kᵀ ---
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
    // fold the 1/sum and the per-token V scale into the probability.
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
        out[(size_t)h * head_dim + d] = acc;
    }
}
