// GDN recurrent (gated delta-rule) update with decay/beta fused in.
//
//   beta  = sigmoid(b[h])
//   decay = exp(ssm_a[h] * softplus(a[h] + dt_bias[h]))
//   state *= decay
//   delta  = (v - stateᵀ·k) * beta
//   state += k ⊗ delta
//   out    = stateᵀ·q
//
// The recurrence is independent per value-dim column vv: every column's
// state slice s[:,vv] is decayed, used, updated and read back using only
// that column plus the shared k/q vectors. So one thread owns one column
// and runs the whole recurrence for it — no inter-phase __syncthreads
// (the old kernel had four), and consecutive threads touch consecutive
// vv, so the strided state accesses coalesce.
//
// grid = (n_heads, head_dim / COLS); block = COLS.
//
// GQA: `n_heads` value heads, `n_k_heads` key/query heads; value head h
// pairs with key head h % n_k_heads (Qwen3.5 tiles them).

#include <hip/hip_runtime.h>

#define COLS 64

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
    const int h  = blockIdx.x;                      // value head
    if (h >= (int)n_heads) return;
    const int kh = h % (int)n_k_heads;              // key/query head
    const unsigned int vv = blockIdx.y * COLS + threadIdx.x;   // this thread's column

    float* q_lds = lds;
    float* k_lds = lds + head_dim;
    for (int i = threadIdx.x; i < (int)head_dim; i += COLS) {
        q_lds[i] = q_in[(size_t)kh * head_dim + i];
        k_lds[i] = k_in[(size_t)kh * head_dim + i];
    }
    __syncthreads();
    if (vv >= head_dim) return;

    const float dec = __expf(ssm_a[h] * softplus_stable_r(a_in[h] + dt_bias[h]));
    const float bet = 1.0f / (1.0f + __expf(-b_in[h]));
    const float vval = v_in[(size_t)h * head_dim + vv];

    float* col = state + (size_t)h * head_dim * head_dim + vv;   // s[kk] at col[kk*head_dim]
    const size_t hd = head_dim;

    // decay the column, accumulate kv_mem = stateᵀ·k
    float kv = 0.0f;
    for (int kk = 0; kk < (int)head_dim; kk++) {
        float s = col[(size_t)kk * hd] * dec;
        col[(size_t)kk * hd] = s;
        kv += s * k_lds[kk];
    }
    const float delta = (vval - kv) * bet;

    // rank-1 update, then out = stateᵀ·q
    float acc = 0.0f;
    for (int kk = 0; kk < (int)head_dim; kk++) {
        float s = col[(size_t)kk * hd] + k_lds[kk] * delta;
        col[(size_t)kk * hd] = s;
        acc += s * q_lds[kk];
    }
    out[(size_t)h * head_dim + vv] = acc;
}
