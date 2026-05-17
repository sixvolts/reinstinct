// GDN recurrent (gated delta-rule) update with decay/beta fused in.
//
//   beta  = sigmoid(b[h])
//   decay = exp(ssm_a[h] * softplus(a[h] + dt_bias[h]))
//   state *= decay
//   delta  = (v - stateᵀ·k) * beta
//   state += k ⊗ delta
//   out    = stateᵀ·q
//
// The recurrence is independent per value-dim column vv. Four threads
// cooperate on one column, each owning a quarter of the kk range: each
// does the decay + rank-1 update for its quarter and a partial kv_mem /
// output dot, combined with two __shfl_xor. Splitting kk gives 4× the
// wavefront count of one-thread-per-column — this kernel is
// occupancy-bound on the state traffic, not compute-bound.
//
// block = 64 (one wavefront): COLS=16 columns × 4 kk-quarters. Lanes of
// one quarter map to consecutive vv, so the strided state accesses
// coalesce. grid = (n_heads, head_dim / COLS).
//
// GQA: value head h pairs with key head h % n_k_heads (Qwen3.5 tiling).

#include <hip/hip_runtime.h>

#define COLS 16

__device__ __forceinline__ float softplus_stable_r(float x) {
    return (x > 0.0f) ? x + __logf(1.0f + __expf(-x))
                      :     __logf(1.0f + __expf(x));
}

extern "C" __global__
void gdn_recurrent_step_fused_f32(const float* __restrict__ q_in,    // [n_k_heads, head_dim]
                                  const float* __restrict__ k_in,    // [n_k_heads, head_dim]
                                  const float* __restrict__ v_in,    // [n_heads,   head_dim]
                                  const float* __restrict__ a_in,    // [n_heads] ssm_alpha proj
                                  const float* __restrict__ b_in,    // [n_heads] ssm_beta proj
                                  const float* __restrict__ ssm_a,   // [n_heads] -exp(A_log)
                                  const float* __restrict__ dt_bias, // [n_heads]
                                  float*       __restrict__ state,   // [n_heads, head_dim, head_dim]
                                  float*       __restrict__ out,     // [n_heads, head_dim]
                                  unsigned int n_heads,
                                  unsigned int head_dim,
                                  unsigned int n_k_heads)
{
    extern __shared__ float lds[];                  // q_lds | k_lds, head_dim each
    const int h = blockIdx.x;                       // value head
    if (h >= (int)n_heads) return;
    const int kh  = h % (int)n_k_heads;             // key/query head
    const int tid = threadIdx.x;                    // 0..63
    const int grp = tid >> 4;                       // kk quarter (0..3)
    const unsigned int vv = blockIdx.y * COLS + (tid & 15);   // this column

    float* q_lds = lds;
    float* k_lds = lds + head_dim;
    for (int i = tid; i < (int)head_dim; i += 64) {
        q_lds[i] = q_in[(size_t)kh * head_dim + i];
        k_lds[i] = k_in[(size_t)kh * head_dim + i];
    }
    __syncthreads();
    if (vv >= head_dim) return;

    const float dec = __expf(ssm_a[h] * softplus_stable_r(a_in[h] + dt_bias[h]));
    const float bet = 1.0f / (1.0f + __expf(-b_in[h]));
    const float vval = v_in[(size_t)h * head_dim + vv];

    const int quarter = (int)head_dim >> 2;
    const int kk0  = grp * quarter;
    const int kk1  = kk0 + quarter;
    float* col = state + (size_t)h * head_dim * head_dim + vv;   // s[kk] at col[kk*head_dim]
    const size_t hd = head_dim;

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
    if (grp == 0) out[(size_t)h * head_dim + vv] = acc;
}
