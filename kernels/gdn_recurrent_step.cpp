// Per-head Gated DeltaNet recurrent update (one decode step).
//
// For each head h with state matrix S[h] of shape [head_dim, head_dim],
// row-major (row index = key dim, col index = value dim):
//
//   1. S      *= decay[h]
//   2. kv_mem  = S^T · k_h          (read columns of S)
//   3. delta   = (v_h - kv_mem) * beta[h]
//   4. S      += k_h ⊗ delta        (rank-1 outer-product update)
//   5. out    = S^T · q_h           (read columns of S)
//
// Threading: one block per head, blockDim.x = head_dim. Each thread owns
// one column index `vv` of the state matrix (and writes one element of
// `out`). For `head_dim = 128`, that's 128 threads = 2 wavefronts on
// gfx906 — one block fits in two 64-thread waves with no idle threads.
//
// LDS layout (4 × head_dim floats):
//   q_lds | k_lds | v_lds | delta

#include <hip/hip_runtime.h>

extern "C" __global__
void gdn_recurrent_step_f32(const float* __restrict__ q_in,    // [n_heads, head_dim]
                            const float* __restrict__ k_in,    // [n_heads, head_dim]
                            const float* __restrict__ v_in,    // [n_heads, head_dim]
                            const float* __restrict__ decay,   // [n_heads]
                            const float* __restrict__ beta,    // [n_heads]
                            float*       __restrict__ state,   // [n_heads, head_dim, head_dim]
                            float*       __restrict__ out,     // [n_heads, head_dim]
                            unsigned int n_heads,
                            unsigned int head_dim)
{
    extern __shared__ float lds[];
    const int h = blockIdx.x;
    if (h >= (int)n_heads) return;
    const int tid = threadIdx.x;
    const int bs  = blockDim.x;

    float* q_lds = lds;
    float* k_lds = lds +     head_dim;
    float* v_lds = lds + 2 * head_dim;
    float* delta = lds + 3 * head_dim;

    // Load q, k, v for this head into LDS.
    for (int i = tid; i < (int)head_dim; i += bs) {
        q_lds[i] = q_in[(size_t)h * head_dim + i];
        k_lds[i] = k_in[(size_t)h * head_dim + i];
        v_lds[i] = v_in[(size_t)h * head_dim + i];
    }
    __syncthreads();

    const float dec = decay[h];
    const float bet = beta[h];
    float* s = state + (size_t)h * head_dim * head_dim;
    const size_t hd = head_dim;

    // 1. state *= decay
    const int n_state = (int)hd * (int)hd;
    for (int i = tid; i < n_state; i += bs) {
        s[i] *= dec;
    }
    __syncthreads();

    // 2 & 3. delta[vv] = (v[vv] - Σ_kk s[kk, vv] * k[kk]) * beta
    //        Each thread handles one (or more) vv columns of s.
    for (int vv = tid; vv < (int)head_dim; vv += bs) {
        float kv_mem = 0.0f;
        for (int kk = 0; kk < (int)head_dim; kk++) {
            kv_mem += s[(size_t)kk * hd + vv] * k_lds[kk];
        }
        delta[vv] = (v_lds[vv] - kv_mem) * bet;
    }
    __syncthreads();

    // 4. state[kk, vv] += k[kk] * delta[vv]
    //    Partition by row index kk so every kk's row is touched by exactly
    //    one thread (no atomics needed).
    for (int kk = tid; kk < (int)head_dim; kk += bs) {
        const float kk_v = k_lds[kk];
        float* row = s + (size_t)kk * hd;
        for (int vv = 0; vv < (int)head_dim; vv++) {
            row[vv] += kk_v * delta[vv];
        }
    }
    __syncthreads();

    // 5. out[h, vv] = Σ_kk s[kk, vv] * q[kk]
    for (int vv = tid; vv < (int)head_dim; vv += bs) {
        float acc = 0.0f;
        for (int kk = 0; kk < (int)head_dim; kk++) {
            acc += s[(size_t)kk * hd + vv] * q_lds[kk];
        }
        out[(size_t)h * head_dim + vv] = acc;
    }
}
