// Flash-attention-style batched causal attention for prefill.
//
// The plain attn_prefill kernel uses one workgroup per (head, query) and
// reads each query's K/V rows straight from global — so K/V global
// traffic is O(P²). This kernel processes a tile of BQ queries per
// workgroup (one wavefront each) and streams the keys in BK-key tiles,
// each tile staged once into LDS and reused by all BQ queries — K/V
// traffic drops to O(P²/BQ). Softmax is the online (flash) form, so the
// full [BQ, P] score matrix is never materialised.
//
//   Q : [P, n_heads,    head_dim]   K/V : [P, n_kv_heads, head_dim]
//   out: [P, n_heads,   head_dim]
//
// A wavefront owns one query; lane l holds query elements l, l+64, …
// (head_dim must be a multiple of 64 — 256 / 512 here). grid =
// (n_heads, ceil(P/BQ)); block = 64·BQ. Dynamic LDS = 2·BK·head_dim f32.

#include <hip/hip_runtime.h>

#define BQ 8          // queries (= wavefronts) per workgroup
#define BK 8          // keys per key-tile
#define DPL_MAX 8     // head_dim/64 upper bound (head_dim ≤ 512)

extern "C" __global__
void attn_prefill_flash_f32(const float* __restrict__ q,
                            const float* __restrict__ k,
                            const float* __restrict__ v,
                            float*       __restrict__ out,
                            unsigned int n_heads,
                            unsigned int n_kv_heads,
                            unsigned int head_dim,
                            unsigned int window,        // 0 = full causal
                            float        scaling,
                            unsigned int p_rows)
{
    extern __shared__ float lds[];
    float* kt = lds;                       // [BK, head_dim]
    float* vt = lds + BK * head_dim;       // [BK, head_dim]

    const int wf   = threadIdx.x >> 6;     // 0..BQ-1  — wavefront = query
    const int lane = threadIdx.x & 63;
    const unsigned int h     = blockIdx.x;
    const unsigned int qbase = blockIdx.y * BQ;
    const unsigned int q_pos = qbase + wf;
    const bool active = q_pos < p_rows;

    const int groups = n_heads / n_kv_heads;
    const unsigned int kv_h = h / groups;
    const unsigned int dpl  = head_dim >> 6;       // dims per lane
    const size_t kv_row = (size_t)n_kv_heads * head_dim;

    // This query's Q — lane l holds elements l, l+64, …
    float qreg[DPL_MAX];
    #pragma unroll
    for (int i = 0; i < DPL_MAX; i++) {
        qreg[i] = (active && (unsigned)i < dpl)
            ? q[((size_t)q_pos * n_heads + h) * head_dim + lane + i * 64] : 0.0f;
    }

    // Online-softmax running state.
    float m = -INFINITY, lsum = 0.0f;
    float acc[DPL_MAX];
    #pragma unroll
    for (int i = 0; i < DPL_MAX; i++) acc[i] = 0.0f;

    // Causal/window key range. The workgroup loops the union over its BQ
    // queries: from the smallest query's `lo` to the largest query pos.
    const int total = (int)q_pos + 1;
    const int lo = (window > 0 && total > (int)window) ? (total - (int)window) : 0;
    const int q0t   = (int)qbase + 1;
    const int lo_min = (window > 0 && q0t > (int)window) ? (q0t - (int)window) : 0;
    const int kt_start = (lo_min / BK) * BK;
    int kt_end = (int)qbase + BQ;
    if (kt_end > (int)p_rows) kt_end = (int)p_rows;

    for (int kt0 = kt_start; kt0 < kt_end; kt0 += BK) {
        // Cooperative load of the K/V tile — coalesced over head_dim.
        for (unsigned int e = threadIdx.x; e < BK * head_dim; e += blockDim.x) {
            const unsigned int kk = e / head_dim, dd = e % head_dim;
            const unsigned int kpos = kt0 + kk;
            if (kpos < p_rows) {
                const size_t o = (size_t)kpos * kv_row + (size_t)kv_h * head_dim + dd;
                kt[e] = k[o];
                vt[e] = v[o];
            } else {
                kt[e] = 0.0f; vt[e] = 0.0f;
            }
        }
        __syncthreads();

        if (active) {
            #pragma unroll
            for (int kk = 0; kk < BK; kk++) {
                const int kpos = kt0 + kk;
                const bool valid = (kpos >= lo) && (kpos <= (int)q_pos);
                float score;
                if (!valid) {
                    score = -INFINITY;
                } else {
                    float partial = 0.0f;
                    #pragma unroll
                    for (int i = 0; i < DPL_MAX; i++)
                        if ((unsigned)i < dpl)
                            partial += qreg[i] * kt[kk * head_dim + lane + i * 64];
                    #pragma unroll
                    for (int o = 32; o > 0; o >>= 1) partial += __shfl_xor(partial, o);
                    score = partial * scaling;
                }
                const float m_new = fmaxf(m, score);
                const float corr = (m == -INFINITY)     ? 0.0f : __expf(m - m_new);
                const float pw   = (score == -INFINITY) ? 0.0f : __expf(score - m_new);
                lsum = lsum * corr + pw;
                #pragma unroll
                for (int i = 0; i < DPL_MAX; i++)
                    if ((unsigned)i < dpl)
                        acc[i] = acc[i] * corr + pw * vt[kk * head_dim + lane + i * 64];
                m = m_new;
            }
        }
        __syncthreads();
    }

    if (active) {
        const float inv = 1.0f / lsum;
        #pragma unroll
        for (int i = 0; i < DPL_MAX; i++)
            if ((unsigned)i < dpl)
                out[((size_t)q_pos * n_heads + h) * head_dim + lane + i * 64]
                    = acc[i] * inv;
    }
}
