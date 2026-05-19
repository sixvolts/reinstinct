// FlashDecoding-style split-K decode attention over an int8 KV cache.
//
// grid = (n_heads, n_splits). One workgroup owns one Q head and a
// contiguous slice of the attended context: it runs a stable softmax
// over its slice and writes a partial (m, l, o). The merge kernel
// (attn_merge.cpp) combines the per-head splits.
//
// Why split the context across blocks: the old one-block-per-head
// kernel (attn_step_q8.cpp) left most CUs idle at depth and ran the P·V
// accumulation serially over the whole context. Splitting restores
// occupancy and shortens the serial scan ~n_splits×.
//
//   partial m = max score over the slice
//   partial l = Σ exp(score - m)            (softmax denominator)
//   partial o = Σ exp(score - m)·dv·V       (unnormalised)
//
// LDS (dynamic): qi[head_dim int8] | scores[chunk f32] | tmp[bs]

#include <hip/hip_runtime.h>
#include <stdint.h>

extern "C" __global__
void attn_partial_q8_f32(const float*       __restrict__ q,        // [n_heads, head_dim]
                         const signed char* __restrict__ k_cache,  // [max_seq, n_kv, head_dim]
                         const float*       __restrict__ k_scale,  // [max_seq, n_kv]
                         const signed char* __restrict__ v_cache,
                         const float*       __restrict__ v_scale,
                         float*             __restrict__ o_partial, // [n_heads, n_splits, head_dim]
                         float*             __restrict__ m_partial, // [n_heads, n_splits]
                         float*             __restrict__ l_partial, // [n_heads, n_splits]
                         unsigned int n_heads,
                         unsigned int n_kv_heads,
                         unsigned int head_dim,
                         const unsigned int* __restrict__ pos_ptr,
                         unsigned int window,
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
    const unsigned int total_len = *pos_ptr + 1;

    // Attended range [lo, total_len); this block owns split `sp`.
    const int lo = (window > 0 && total_len > window) ? (int)(total_len - window) : 0;
    const int win_len = (int)total_len - lo;
    const int chunk = (win_len + (int)n_splits - 1) / (int)n_splits;
    const int slice_start = lo + sp * chunk;
    int slice_end = lo + win_len;
    if (slice_start + chunk < slice_end) slice_end = slice_start + chunk;
    const int slice_len = slice_end - slice_start;

    signed char* qi = reinterpret_cast<signed char*>(lds);
    float* scores   = reinterpret_cast<float*>(qi + head_dim);   // [chunk]
    float* tmp      = scores + chunk;                            // [bs]

    // Empty split (n_splits > win_len): neutral partials.
    if (slice_len <= 0) {
        for (int d = tid; d < (int)head_dim; d += bs)
            o_partial[((size_t)h * n_splits + sp) * head_dim + d] = 0.0f;
        if (tid == 0) {
            m_partial[(size_t)h * n_splits + sp] = -INFINITY;
            l_partial[(size_t)h * n_splits + sp] = 0.0f;
        }
        return;
    }

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

    const int* qi32 = reinterpret_cast<const int*>(qi);
    const int  n4   = (int)head_dim >> 2;
    const size_t kv_row = (size_t)n_kv_heads * head_dim;

    // --- scores: int8 dp4a Q·Kᵀ over the slice ---
    for (int i = tid; i < slice_len; i += bs) {
        const int t = slice_start + i;
        const int* k32 = reinterpret_cast<const int*>(
            k_cache + (size_t)t * kv_row + (size_t)kv_h * head_dim);
        int idot = 0;
        for (int c = 0; c < n4; c++)
            idot = __builtin_amdgcn_sdot4(qi32[c], k32[c], idot, false);
        const float dk = k_scale[(size_t)t * n_kv_heads + kv_h];
        scores[i] = dq * dk * (float)idot * scaling;
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
    // fold the per-token V scale into the (unnormalised) probabilities.
    for (int i = tid; i < slice_len; i += bs)
        scores[i] *= v_scale[(size_t)(slice_start + i) * n_kv_heads + kv_h];
    __syncthreads();

    // --- P·V: o[d] = Σ_i scores[i]·V[t][d] ---
    for (int d = tid; d < (int)head_dim; d += bs) {
        float acc = 0.0f;
        for (int i = 0; i < slice_len; i++) {
            const int t = slice_start + i;
            acc += scores[i] * (float)v_cache[(size_t)t * kv_row
                                              + (size_t)kv_h * head_dim + d];
        }
        o_partial[((size_t)h * n_splits + sp) * head_dim + d] = acc;
    }
    if (tid == 0) {
        m_partial[(size_t)h * n_splits + sp] = mx;
        l_partial[(size_t)h * n_splits + sp] = l;
    }
}
