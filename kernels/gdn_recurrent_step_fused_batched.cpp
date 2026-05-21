// Batched GDN recurrent step + decay/beta fusion. Same per-row math as
// gdn_recurrent_step_fused.cpp, but the outer prefill loop
// `for r in 0..n_rows` runs INSIDE the kernel — each workgroup loops
// over rows in order, threading the persistent `state` matrix through
// the iterations (the recurrence's data dependency lives in `state`,
// which is read+written by every iteration).
//
// Workgroup geometry is unchanged: one workgroup per (head, vv-tile)
// pair, 64 threads, COLS=16 columns × 4 kk-quarters per workgroup. The
// per-row inputs (q, k, v, a, b) are re-fetched per iteration; q/k go
// through LDS as before, but the LDS is rewritten at the top of each
// row iteration (with a __syncthreads before AND after the rewrite so
// the previous row's reads finish before the writes, and the writes
// finish before the new row reads).

#include <hip/hip_runtime.h>

#define COLS 16

__device__ __forceinline__ float softplus_stable_r(float x) {
    return (x > 0.0f) ? x + __logf(1.0f + __expf(-x))
                      :     __logf(1.0f + __expf(x));
}

extern "C" __global__
void gdn_recurrent_step_fused_batched_f32(
    const float* __restrict__ q_in_batch,    // [n_rows, n_k_heads, head_dim]
    const float* __restrict__ k_in_batch,    // [n_rows, n_k_heads, head_dim]
    const float* __restrict__ v_in_batch,    // [n_rows, n_heads,   head_dim]
    const float* __restrict__ a_in_batch,    // [n_rows, n_heads]
    const float* __restrict__ b_in_batch,    // [n_rows, n_heads]
    const float* __restrict__ ssm_a,         // [n_heads]
    const float* __restrict__ dt_bias,       // [n_heads]
    float*       __restrict__ state,         // [n_heads, head_dim, head_dim]
    float*       __restrict__ out_batch,     // [n_rows, n_heads, head_dim]
    unsigned int n_heads,
    unsigned int head_dim,
    unsigned int n_k_heads,
    unsigned int n_rows,
    unsigned int qk_row_stride,              // floats per row in q/k_in (= n_k_heads * head_dim)
    unsigned int v_row_stride,               // floats per row in v_in   (= n_heads   * head_dim)
    unsigned int ab_row_stride,              // floats per row in a/b    (= n_heads)
    unsigned int out_row_stride)             // floats per row in out    (= n_heads * head_dim)
{
    extern __shared__ float lds[];           // q_lds | k_lds, head_dim each
    const int h = blockIdx.x;
    if (h >= (int)n_heads) return;
    const int kh  = h % (int)n_k_heads;
    const int tid = threadIdx.x;
    const int grp = tid >> 4;
    const unsigned int vv = blockIdx.y * COLS + (tid & 15);

    float* q_lds = lds;
    float* k_lds = lds + head_dim;

    const int quarter = (int)head_dim >> 2;
    const int kk0  = grp * quarter;
    const int kk1  = kk0 + quarter;
    const size_t hd = head_dim;
    // Per-head state column — same column across all rows; the recurrence
    // mutates it in place.
    float* col = state + (size_t)h * head_dim * head_dim + vv;

    // Per-head constants (decay's `ssm_a` and `dt_bias` are row-invariant).
    const float ssm_a_h   = ssm_a[h];
    const float dt_bias_h = dt_bias[h];

    for (unsigned int r = 0; r < n_rows; r++) {
        // Sync before rewriting LDS so the previous row's reads finish.
        __syncthreads();

        // Load this row's q, k into LDS (every thread participates).
        for (int i = tid; i < (int)head_dim; i += 64) {
            q_lds[i] = q_in_batch[(size_t)r * qk_row_stride + (size_t)kh * head_dim + i];
            k_lds[i] = k_in_batch[(size_t)r * qk_row_stride + (size_t)kh * head_dim + i];
        }
        __syncthreads();
        if (vv >= head_dim) continue;

        // Per-row decay, beta, v scalar.
        const float dec = __expf(ssm_a_h *
            softplus_stable_r(a_in_batch[(size_t)r * ab_row_stride + h] + dt_bias_h));
        const float bet = 1.0f / (1.0f + __expf(
            -b_in_batch[(size_t)r * ab_row_stride + h]));
        const float vval = v_in_batch[(size_t)r * v_row_stride
                                    + (size_t)h * head_dim + vv];

        // decay this quarter of the column, accumulate partial stateᵀ·k
        float pkv = 0.0f;
        for (int kk = kk0; kk < kk1; kk++) {
            float s = col[(size_t)kk * hd] * dec;
            col[(size_t)kk * hd] = s;
            pkv += s * k_lds[kk];
        }
        pkv += __shfl_xor(pkv, 16);
        const float kv = pkv + __shfl_xor(pkv, 32);
        const float delta = (vval - kv) * bet;

        // rank-1 update of this quarter, accumulate partial stateᵀ·q
        float pout = 0.0f;
        for (int kk = kk0; kk < kk1; kk++) {
            float s = col[(size_t)kk * hd] + k_lds[kk] * delta;
            col[(size_t)kk * hd] = s;
            pout += s * q_lds[kk];
        }
        pout += __shfl_xor(pout, 16);
        const float acc = pout + __shfl_xor(pout, 32);
        if (grp == 0) {
            out_batch[(size_t)r * out_row_stride + (size_t)h * head_dim + vv] = acc;
        }
    }
}
