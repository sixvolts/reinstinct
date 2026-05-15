// LDS-staged GDN recurrent update.
//
// Same math as gdn_recurrent_step.cpp; only the data placement changes.
// The per-head state matrix [head_dim, head_dim] is loaded into LDS
// once at the top of the kernel and written back once at the bottom,
// so steps 2/4/5 (which all touch the state heavily) become LDS-bound
// instead of HBM-bound.
//
// LDS budget on gfx906 is 64 KB per CU. For head_dim = 128 the state
// is exactly 128 * 128 * 4 = 65536 bytes — uses the full LDS, which
// caps occupancy at one block per CU. That's fine here: we only launch
// 16 blocks per call (n_heads = 16) and there are 60 CUs available,
// so per-CU occupancy was never the bottleneck.
//
// q, k, v, and delta no longer live in LDS — they're tiny (≤ 512 B
// each per head) and L1-cached on every wave. delta is written through
// a per-head scratch buffer (caller's `delta` arg).
//
// HBM traffic per head per call drops from ~5× state reads/writes to
// 1× state read + 1× state write, plus tiny q/k/v/delta/out traffic.

#include <hip/hip_runtime.h>

extern "C" __global__
void gdn_recurrent_step_lds_f32(const float* __restrict__ q_in,    // [n_heads, head_dim]
                                const float* __restrict__ k_in,    // [n_heads, head_dim]
                                const float* __restrict__ v_in,    // [n_heads, head_dim]
                                const float* __restrict__ decay,   // [n_heads]
                                const float* __restrict__ beta,    // [n_heads]
                                float*       __restrict__ state,   // [n_heads, head_dim, head_dim]
                                float*       __restrict__ out,     // [n_heads, head_dim]
                                float*       __restrict__ delta_scratch, // [n_heads, head_dim]
                                unsigned int n_heads,
                                unsigned int head_dim)
{
    extern __shared__ float s_lds[];   // sized to head_dim * head_dim by the launcher

    const int h = blockIdx.x;
    if (h >= (int)n_heads) return;
    const int tid = threadIdx.x;
    const int bs  = blockDim.x;
    const size_t hd = head_dim;

    const float* q     = q_in + (size_t)h * head_dim;
    const float* k     = k_in + (size_t)h * head_dim;
    const float* v     = v_in + (size_t)h * head_dim;
    const float dec    = decay[h];
    const float bet    = beta[h];
    float*       s_g   = state + (size_t)h * head_dim * head_dim;
    float*       delta = delta_scratch + (size_t)h * head_dim;

    // Load state from HBM to LDS, applying decay in flight (fuses step 1).
    const int n_state = (int)hd * (int)hd;
    for (int i = tid; i < n_state; i += bs) {
        s_lds[i] = s_g[i] * dec;
    }
    __syncthreads();

    // Steps 2 & 3: each thread vv computes its delta from the LDS state
    // and writes it to a per-head scratch in HBM (so all threads can
    // read it back in step 4 — cross-wavefront so LDS would be ideal,
    // but our LDS budget is fully spent on the state).
    if (tid < (int)head_dim) {
        const int vv = tid;
        float kv_mem = 0.0f;
        for (int kk = 0; kk < (int)head_dim; kk++) {
            kv_mem += s_lds[(size_t)kk * hd + vv] * k[kk];
        }
        delta[vv] = (v[vv] - kv_mem) * bet;
    }
    __syncthreads();

    // Step 4: rank-1 outer product update, in LDS. Partition by row
    // (tid == kk) so each row is touched by one thread — no atomics.
    if (tid < (int)head_dim) {
        const int kk = tid;
        const float kk_v = k[kk];
        float* row = s_lds + (size_t)kk * hd;
        for (int vv = 0; vv < (int)head_dim; vv++) {
            row[vv] += kk_v * delta[vv];
        }
    }
    __syncthreads();

    // Step 5: out[vv] = Σ_kk s[kk, vv] * q[kk] from LDS.
    if (tid < (int)head_dim) {
        const int vv = tid;
        float acc = 0.0f;
        for (int kk = 0; kk < (int)head_dim; kk++) {
            acc += s_lds[(size_t)kk * hd + vv] * q[kk];
        }
        out[(size_t)h * head_dim + vv] = acc;
    }
    __syncthreads();

    // Write the updated state back to HBM (LDS → HBM).
    for (int i = tid; i < n_state; i += bs) {
        s_g[i] = s_lds[i];
    }
}
