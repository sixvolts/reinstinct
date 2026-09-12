// GDN gated delta-rule recurrent step, decode (one token).
//
//   S = S * dec;  kv = S^T k;  delta = (v - kv) * beta;  S += k delta^T;
//   out = S^T q          — per value head, S is [head_dim x head_dim].
//
// Layout of the work: one workgroup per (head, 32-column slab). 256
// threads = 8 row groups x 32 columns; thread (g, c) owns column c of
// the slab for rows g*R .. g*R+R-1 (R = head_dim / 8) and keeps those R
// state values in registers between the decay pass and the update
// pass, so the state is read once and written once per token (the
// previous kernel re-read and re-wrote it: four passes over 3 MB per
// block on the 27B). A 32-column slab is one 128-byte cache line per
// row, so every load instruction is a full line; the previous 16-column
// slab used half lines. The kv / out column sums span the 8 row groups
// (different waves) and go through LDS.
#include <hip/hip_runtime.h>

#define COLS   32
#define GROUPS 8
#define RMAX   32          // head_dim <= 256

__device__ __forceinline__ float softplus_stable_r(float x) {
    return (x > 0.0f) ? x + __logf(1.0f + __expf(-x))
                      :     __logf(1.0f + __expf(x));
}

extern "C" __global__ __launch_bounds__(256)
void gdn_recurrent_step_v2_f32(const float* __restrict__ q_in,    // [n_k_heads, head_dim]
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
                               unsigned int n_k_heads,
                               unsigned int n_part,      // a_in/b_in are n_part partial sums,
                               unsigned int part_stride) // part p at + p*part_stride
{
    extern __shared__ float lds[];                  // q | k (head_dim each) | red[GROUPS][COLS]
    const int h   = blockIdx.x;
    const int kh  = h % (int)n_k_heads;
    const int tid = threadIdx.x;
    const int g   = tid >> 5;                       // row group 0..7
    const int c   = tid & 31;                       // column within the slab
    const unsigned int vv = blockIdx.y * COLS + c;
    float* q_lds = lds;
    float* k_lds = lds + head_dim;
    float* red   = lds + 2 * head_dim;              // [GROUPS][COLS]
    for (int i = tid; i < (int)head_dim; i += 256) {
        q_lds[i] = q_in[(size_t)kh * head_dim + i];
        k_lds[i] = k_in[(size_t)kh * head_dim + i];
    }
    float a_h = 0.0f, b_h = 0.0f;
    for (unsigned int p = 0; p < n_part; p++) {
        a_h += a_in[(size_t)p * part_stride + h];
        b_h += b_in[(size_t)p * part_stride + h];
    }
    const float dec = __expf(ssm_a[h] * softplus_stable_r(a_h + dt_bias[h]));
    const float bet = 1.0f / (1.0f + __expf(-b_h));
    const bool active = vv < head_dim;
    const float vval = active ? v_in[(size_t)h * head_dim + vv] : 0.0f;
    const int R   = (int)head_dim / GROUPS;
    const int kk0 = g * R;
    float* col = state + (size_t)h * head_dim * head_dim + (active ? vv : 0);
    const size_t hd = head_dim;
    __syncthreads();

    // Decay pass: load this thread's R rows once, scale, and dot with k.
    float s[RMAX];
    float pkv = 0.0f;
    #pragma unroll
    for (int i = 0; i < RMAX; i++) {
        if (i < R) {
            const int kk = kk0 + i;
            s[i] = col[(size_t)kk * hd] * dec;
            pkv += s[i] * k_lds[kk];
        }
    }
    red[g * COLS + c] = pkv;
    __syncthreads();
    float kv = 0.0f;
    #pragma unroll
    for (int gg = 0; gg < GROUPS; gg++) kv += red[gg * COLS + c];
    const float delta = (vval - kv) * bet;
    __syncthreads();                                // red is reused below

    // Update pass from registers: S += k delta^T, write once, dot with q.
    float pout = 0.0f;
    #pragma unroll
    for (int i = 0; i < RMAX; i++) {
        if (i < R) {
            const int kk = kk0 + i;
            const float sn = s[i] + k_lds[kk] * delta;
            if (active) col[(size_t)kk * hd] = sn;
            pout += sn * q_lds[kk];
        }
    }
    red[g * COLS + c] = pout;
    __syncthreads();
    if (g == 0 && active) {
        float acc = 0.0f;
        #pragma unroll
        for (int gg = 0; gg < GROUPS; gg++) acc += red[gg * COLS + c];
        out[(size_t)h * head_dim + vv] = acc;
    }
}
