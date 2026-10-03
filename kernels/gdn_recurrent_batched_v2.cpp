// GDN gated delta-rule recurrence over a batch of rows (prefill), v2.
//
// Same algorithm and geometry as gdn_recurrent_step_fused_batched_lds128
// — one workgroup per (head, 16-column slab), 64 threads = 4 kk-groups
// x 16 columns, the 8 KB state slice resident in LDS across all rows —
// with the two things that made that kernel 3x slower in the model than
// in its microbench fixed:
//
//   * it staged every row's a / b scalar in LDS (2 x n_rows floats): at
//     3971 rows that is 32 KB, one workgroup per CU, and the 384
//     workgroups of the 27B ran in ~6 sequential rounds of the whole
//     row loop. Here the LDS footprint is fixed at 9 KB (state slice
//     8 KB, no pad, plus q / k), so 7 workgroups fit a CU and all 384
//     are co-resident: one round.
//   * per row the q / k / v / a / b loads for the row were issued after
//     the barrier and waited on before compute. Here row r+1's values
//     are prefetched into registers while row r computes.
//
// The state slice is XOR-swizzled instead of +1-padded: element (vv, kk)
// lives at vv*HD + (kk ^ (2*vv)), so the 16 columns of a group hit 16
// distinct banks at any kk (the pad had the same effect at 8.3 KB, which
// rounds to a 19th LDS granule and loses the 7th workgroup).
//
// Registers: the per-row loops address the slice through 8 per-lane
// bases plus immediate offsets (see the row loop) instead of 32 hoisted
// swizzled addresses, and the checkpoint store has its own entry. That
// took the kernel from 157 VGPRs (one wave per SIMD: 256 workgroups on
// 240 SIMDs ran in two rounds) to 111; Qwen 3.6-35B shape 1.96 -> 1.24 ms
// per layer at 512 rows.
//
// gdn_recurrent_batched_v2_ckpt_f32 also writes the state after each of
// the first ckpt_rows rows to ckpt (spec-decode verify); the plain entry
// takes the same arguments and ignores those two.
//
// Compiled with `#define GDN_HEAD_DIM <head_dim>` (a multiple of 64).
#include <hip/hip_runtime.h>

#ifndef GDN_HEAD_DIM
#define GDN_HEAD_DIM 128
#endif
#define HD      GDN_HEAD_DIM
#define COLS    16
#define QUARTER (HD / 4)
#define PER_T   (HD / 64)           // q / k elements each thread stages

static_assert(HD % 64 == 0 && HD <= 256, "head_dim must be a multiple of 64, <= 256");

// The workgroup is exactly one wave64 and a wave's LDS operations
// complete in order, so ordering the compiler is enough. __syncthreads()
// also drains outstanding global loads (its fence waits on vmcnt), which
// stalled every row on the very prefetch meant to hide its latency.
// (From the llama fork's gated_delta_net_lds_wave64, crossport L8b.)
__device__ __forceinline__ void gdn_wave_sync() {
    __builtin_amdgcn_wave_barrier();
    __atomic_signal_fence(__ATOMIC_SEQ_CST);
}

__device__ __forceinline__ float softplus_stable_r(float x) {
    return (x > 0.0f) ? x + __logf(1.0f + __expf(-x))
                      :     __logf(1.0f + __expf(x));
}

#define GDN_V2_PARAMS                                                     \
    const float* __restrict__ q_in_batch,                                 \
    const float* __restrict__ k_in_batch,                                 \
    const float* __restrict__ v_in_batch,                                 \
    const float* __restrict__ a_in_batch,                                 \
    const float* __restrict__ b_in_batch,                                 \
    const float* __restrict__ ssm_a,                                      \
    const float* __restrict__ dt_bias,                                    \
    float*       __restrict__ state,                                      \
    float*       __restrict__ out_batch,                                  \
    unsigned int n_heads,                                                 \
    unsigned int head_dim,  /* == GDN_HEAD_DIM; ABI parity */             \
    unsigned int n_k_heads,                                               \
    unsigned int n_rows,                                                  \
    unsigned int qk_row_stride,                                           \
    unsigned int v_row_stride,                                            \
    unsigned int ab_row_stride,                                           \
    unsigned int out_row_stride,                                          \
    float*       __restrict__ ckpt,  /* [ckpt_rows][n_heads*HD*HD] */     \
    unsigned int ckpt_rows
#define GDN_V2_ARGS q_in_batch, k_in_batch, v_in_batch, a_in_batch, b_in_batch, ssm_a, \
    dt_bias, state, out_batch, n_heads, head_dim, n_k_heads, n_rows, qk_row_stride,   \
    v_row_stride, ab_row_stride, out_row_stride, ckpt, ckpt_rows

template <bool CKPT>
__device__ __forceinline__ void gdn_batched_v2(GDN_V2_PARAMS)
{
    (void)head_dim;
    __shared__ float state_lds[COLS * HD];
    __shared__ float q_lds[HD];
    __shared__ float k_lds[HD];

    const int h   = blockIdx.x;
    const int kh  = h % (int)n_k_heads;
    const int tid = threadIdx.x;
    const int grp = tid >> 4;
    const int lvv = tid & 15;
    const unsigned int tile_base = blockIdx.y * COLS;
    const unsigned int vv = tile_base + lvv;
    const size_t head_base = (size_t)h * HD * HD;

    // Stage the slice: kk-major in HBM (consecutive vv coalesce), swizzled here.
    for (int i = tid; i < COLS * HD; i += 64) {
        const int kk = i >> 4, c = i & 15;
        state_lds[c * HD + (kk ^ (2 * c))] = state[head_base + (size_t)kk * HD + tile_base + c];
    }
    const float ssm_a_h   = ssm_a[h];
    const float dt_bias_h = dt_bias[h];
    const float* qk_base_q = q_in_batch + (size_t)kh * HD;
    const float* qk_base_k = k_in_batch + (size_t)kh * HD;
    const float* v_base    = v_in_batch + (size_t)h * HD + vv;
    const float* a_base    = a_in_batch + h;
    const float* b_base    = b_in_batch + h;

    // Row-0 prefetch.
    float pq[PER_T], pk[PER_T];
    #pragma unroll
    for (int i = 0; i < PER_T; i++) {
        pq[i] = qk_base_q[tid + i * 64];
        pk[i] = qk_base_k[tid + i * 64];
    }
    float pv = v_base[0];
    float pa = a_base[0];
    float pb = b_base[0];

    float* lds_vv = state_lds + lvv * HD;
    const int swz = 2 * lvv;

    for (unsigned int r = 0; r < n_rows; r++) {
        gdn_wave_sync();                       // row r-1 done reading q/k
        #pragma unroll
        for (int i = 0; i < PER_T; i++) { q_lds[tid + i * 64] = pq[i]; k_lds[tid + i * 64] = pk[i]; }
        const float vval = pv, a_r = pa, b_r = pb;
        // Prefetch row r+1 (clamped: the last row re-reads itself).
        {
            const size_t rn = (size_t)min(r + 1, n_rows - 1);
            #pragma unroll
            for (int i = 0; i < PER_T; i++) {
                pq[i] = qk_base_q[rn * qk_row_stride + tid + i * 64];
                pk[i] = qk_base_k[rn * qk_row_stride + tid + i * 64];
            }
            pv = v_base[rn * v_row_stride];
            pa = a_base[rn * ab_row_stride];
            pb = b_base[rn * ab_row_stride];
        }
        gdn_wave_sync();

        const float dec = __expf(ssm_a_h * softplus_stable_r(a_r + dt_bias_h));
        const float bet = 1.0f / (1.0f + __expf(-b_r));

        // kk = local*4 + grp and the swizzle xors 2*lvv into it, which only
        // touches the low 3 bits of `local`: with local = 8*hi + lo the
        // physical word is sw_lo[lo] + 32*hi, so 8 per-lane bases plus
        // immediate offsets cover all 32 (not 32 hoisted addresses).
        float pkv = 0.0f;
        float s_arr[QUARTER];
        #pragma unroll
        for (int lo = 0; lo < 8; lo++) {
            const float* sp = lds_vv + ((((lo ^ (lvv >> 1)) << 2)) | ((grp ^ swz) & 3));
            const float* kp = k_lds + lo * 4 + grp;
            #pragma unroll
            for (int hi = 0; hi < QUARTER / 8; hi++) { const float sv = sp[32 * hi] * dec; s_arr[lo * 4 + hi] = sv; pkv += sv * kp[32 * hi]; }
        }
        pkv += __shfl_xor(pkv, 16);
        const float kv = pkv + __shfl_xor(pkv, 32);
        const float delta = (vval - kv) * bet;

        float pout = 0.0f;
        #pragma unroll
        for (int lo = 0; lo < 8; lo++) {
            float* sp = lds_vv + ((((lo ^ (lvv >> 1)) << 2)) | ((grp ^ swz) & 3));
            const float* kp = k_lds + lo * 4 + grp;
            const float* qp = q_lds + lo * 4 + grp;
            #pragma unroll
            for (int hi = 0; hi < QUARTER / 8; hi++) {
                const float sv = s_arr[lo * 4 + hi] + kp[32 * hi] * delta;
                sp[32 * hi] = sv;
                pout += sv * qp[32 * hi];
            }
        }
        // Spec-decode verify: the state after row r, in `state`'s layout,
        // so a partial accept restores it with one copy. Outside the loop
        // above (a branch in it cost prefill 2x); each lane re-reads the
        // LDS words it just wrote.
        if (CKPT && r < ckpt_rows) {
            float* ck = ckpt + (size_t)r * n_heads * HD * HD + head_base + vv;
            #pragma unroll 1
            for (int lo = 0; lo < 8; lo++) {
                const float* sp = lds_vv + ((((lo ^ (lvv >> 1)) << 2)) | ((grp ^ swz) & 3));
                float* cp = ck + (size_t)(lo * 4 + grp) * HD;
                #pragma unroll
                for (int hi = 0; hi < QUARTER / 8; hi++) cp[(size_t)32 * hi * HD] = sp[32 * hi];
            }
        }
        pout += __shfl_xor(pout, 16);
        const float acc = pout + __shfl_xor(pout, 32);
        if (grp == 0)
            out_batch[(size_t)r * out_row_stride + (size_t)h * HD + vv] = acc;
    }

    __syncthreads();
    for (int i = tid; i < COLS * HD; i += 64) {
        const int kk = i >> 4, c = i & 15;
        state[head_base + (size_t)kk * HD + tile_base + c] = state_lds[c * HD + (kk ^ (2 * c))];
    }
}

// Prefill: no checkpoints (ckpt / ckpt_rows ignored). The checkpoint
// store costs ~10% even when skipped (registers and code around the row
// loop), so it gets its own entry.
extern "C" __global__ __launch_bounds__(64)
void gdn_recurrent_batched_v2_f32(GDN_V2_PARAMS)
{ gdn_batched_v2<false>(GDN_V2_ARGS); }

// Spec-decode verify: also writes the state after each of the first
// ckpt_rows rows to ckpt.
extern "C" __global__ __launch_bounds__(64)
void gdn_recurrent_batched_v2_ckpt_f32(GDN_V2_PARAMS)
{ gdn_batched_v2<true>(GDN_V2_ARGS); }
