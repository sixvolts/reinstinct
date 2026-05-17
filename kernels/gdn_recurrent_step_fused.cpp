// GDN recurrent update with decay/beta computation fused in.
//
// Identical to gdn_recurrent_step.cpp except it takes the raw
// projection outputs (a, b) plus the per-head parameters (ssm_a,
// dt_bias) and computes decay/beta inline instead of receiving them
// from a separate gdn_decay_beta kernel:
//
//   beta  = sigmoid(b[h])
//   decay = exp(ssm_a[h] * softplus(a[h] + dt_bias[h]))
//
// Every thread recomputes the two per-head scalars — that's a handful
// of flops, far cheaper than a separate kernel launch.

#include <hip/hip_runtime.h>

__device__ __forceinline__ float softplus_stable_r(float x) {
    return (x > 0.0f) ? x + __logf(1.0f + __expf(-x))
                      :     __logf(1.0f + __expf(x));
}

// GQA: `n_heads` value heads, `n_k_heads` key/query heads. Value head h
// reads its q/k from key head `h / (n_heads / n_k_heads)`. With
// n_k_heads == n_heads this reduces to the uniform-head case.
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
    extern __shared__ float lds[];
    const int h = blockIdx.x;                              // value head
    if (h >= (int)n_heads) return;
    const int kh = h / ((int)n_heads / (int)n_k_heads);    // key/query head
    const int tid = threadIdx.x;
    const int bs  = blockDim.x;

    float* q_lds = lds;
    float* k_lds = lds +     head_dim;
    float* v_lds = lds + 2 * head_dim;
    float* delta = lds + 3 * head_dim;

    for (int i = tid; i < (int)head_dim; i += bs) {
        q_lds[i] = q_in[(size_t)kh * head_dim + i];
        k_lds[i] = k_in[(size_t)kh * head_dim + i];
        v_lds[i] = v_in[(size_t)h  * head_dim + i];
    }
    __syncthreads();

    // Fused decay/beta — every thread computes the same two scalars.
    const float ax  = a_in[h] + dt_bias[h];
    const float dec = __expf(ssm_a[h] * softplus_stable_r(ax));
    const float bet = 1.0f / (1.0f + __expf(-b_in[h]));

    float* s = state + (size_t)h * head_dim * head_dim;
    const size_t hd = head_dim;

    // 1. state *= decay
    const int n_state = (int)hd * (int)hd;
    for (int i = tid; i < n_state; i += bs) {
        s[i] *= dec;
    }
    __syncthreads();

    // 2 & 3. delta = (v - state^T·k) * beta
    for (int vv = tid; vv < (int)head_dim; vv += bs) {
        float kv_mem = 0.0f;
        for (int kk = 0; kk < (int)head_dim; kk++) {
            kv_mem += s[(size_t)kk * hd + vv] * k_lds[kk];
        }
        delta[vv] = (v_lds[vv] - kv_mem) * bet;
    }
    __syncthreads();

    // 4. state += k ⊗ delta
    for (int kk = tid; kk < (int)head_dim; kk += bs) {
        const float kk_v = k_lds[kk];
        float* row = s + (size_t)kk * hd;
        for (int vv = 0; vv < (int)head_dim; vv++) {
            row[vv] += kk_v * delta[vv];
        }
    }
    __syncthreads();

    // 5. out = state^T · q
    for (int vv = tid; vv < (int)head_dim; vv += bs) {
        float acc = 0.0f;
        for (int kk = 0; kk < (int)head_dim; kk++) {
            acc += s[(size_t)kk * hd + vv] * q_lds[kk];
        }
        out[(size_t)h * head_dim + vv] = acc;
    }
}
