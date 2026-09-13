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

__device__ __forceinline__ float softplus_stable_r(float x) {
    return (x > 0.0f) ? x + __logf(1.0f + __expf(-x))
                      :     __logf(1.0f + __expf(x));
}

extern "C" __global__ __launch_bounds__(64)
void gdn_recurrent_batched_v2_f32(
    const float* __restrict__ q_in_batch,
    const float* __restrict__ k_in_batch,
    const float* __restrict__ v_in_batch,
    const float* __restrict__ a_in_batch,
    const float* __restrict__ b_in_batch,
    const float* __restrict__ ssm_a,
    const float* __restrict__ dt_bias,
    float*       __restrict__ state,
    float*       __restrict__ out_batch,
    unsigned int n_heads,
    unsigned int head_dim,      // == GDN_HEAD_DIM; ABI parity with the general kernel
    unsigned int n_k_heads,
    unsigned int n_rows,
    unsigned int qk_row_stride,
    unsigned int v_row_stride,
    unsigned int ab_row_stride,
    unsigned int out_row_stride)
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
        __syncthreads();                       // row r-1 done reading q/k
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
        __syncthreads();

        const float dec = __expf(ssm_a_h * softplus_stable_r(a_r + dt_bias_h));
        const float bet = 1.0f / (1.0f + __expf(-b_r));

        float s_arr[QUARTER];
        float pkv = 0.0f;
        #pragma unroll
        for (int local = 0; local < QUARTER; local++) {
            const int kk = local * 4 + grp;
            const float s = lds_vv[kk ^ swz] * dec;
            s_arr[local] = s;
            pkv += s * k_lds[kk];
        }
        pkv += __shfl_xor(pkv, 16);
        const float kv = pkv + __shfl_xor(pkv, 32);
        const float delta = (vval - kv) * bet;

        float pout = 0.0f;
        #pragma unroll
        for (int local = 0; local < QUARTER; local++) {
            const int kk = local * 4 + grp;
            const float s = s_arr[local] + k_lds[kk] * delta;
            lds_vv[kk ^ swz] = s;
            pout += s * q_lds[kk];
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
