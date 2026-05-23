// LDS-resident-state variant of gdn_recurrent_step_fused_batched for
// the qwen 3.5/3.6 head_dim=128 case.
//
// The general kernel's hot path is HBM-latency-bound: per row, every
// thread does 2*quarter (=64) strided reads + writes to the state matrix
// in global memory. With sequential row dependency, those latencies
// can't pipeline across rows. At n_rows=504 that's ~30k HBM round-trips
// per workgroup per call — the kernel measures 8.57 ms/call.
//
// Each WG only needs an 8 KB slice of state (HEAD_DIM × COLS floats =
// 128 × 16 × 4 = 8192). MI50 LDS per CU is 64 KB, so 2 WG/CU × 8 KB
// fits comfortably. Stage the slice into LDS at the start of the call,
// do all 504 row updates from LDS, write the slice back at the end.
// Per-row state ops become LDS-fast (~10 cycles) instead of HBM-slow
// (~200 ns).
//
// LDS layout chosen as state_lds[VV][KK+PAD]: the per-thread inner loop
// (fixed vv, varying kk) reads contiguous addresses.
//
// Bank conflict avoidance: with stride exactly HEAD_DIM=128, every wave
// LDS access was a *64-way* bank conflict — addr/4 mod 32 reduced to
// (kk mod 32), and the old grp→kk mapping (kk = grp*32 + local) made
// all 4 grps share the same bank at every iteration. Two changes fix
// it:
//
//   1. Stride is HEAD_DIM + 1 = 129. lvv*129 mod 32 = lvv mod 32, so
//      the 16 lvvs in a grp span 16 distinct banks (no within-grp
//      conflict).
//   2. Per-grp kk traversal is `kk = local*4 + grp`. The 4 grps walk
//      4 different kk %4 classes in lockstep, so at each iteration the
//      cross-grp banks differ by 1. Combined with (1), the 64-lane
//      access pattern reduces to a 2-way conflict — a 32× cut in LDS
//      latency vs the 64-way layout.

#include <hip/hip_runtime.h>

#define HEAD_DIM 128
#define QUARTER  (HEAD_DIM / 4)    // 32
#define COLS     16
#define STATE_STRIDE (HEAD_DIM + 1)  // 129 (+1 pad to break bank conflicts)

__device__ __forceinline__ float softplus_stable_r(float x) {
    return (x > 0.0f) ? x + __logf(1.0f + __expf(-x))
                      :     __logf(1.0f + __expf(x));
}

extern "C" __global__ __launch_bounds__(64, 2)
void gdn_recurrent_step_fused_batched_lds128_f32(
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
    unsigned int head_dim,     // must be 128; kept for ABI parity
    unsigned int n_k_heads,
    unsigned int n_rows,
    unsigned int qk_row_stride,
    unsigned int v_row_stride,
    unsigned int ab_row_stride,
    unsigned int out_row_stride)
{
    (void)head_dim;
    // LDS layout:
    //   state_slice [COLS][STATE_STRIDE]   COLS*STATE_STRIDE floats
    //   q_lds       [HEAD_DIM]
    //   k_lds       [HEAD_DIM]
    //   a_lds       [n_rows]
    //   b_lds       [n_rows]
    extern __shared__ float lds[];
    float* state_lds = lds;
    float* q_lds     = lds + COLS * STATE_STRIDE;
    float* k_lds     = lds + COLS * STATE_STRIDE + HEAD_DIM;
    float* a_lds     = lds + COLS * STATE_STRIDE + 2 * HEAD_DIM;
    float* b_lds     = a_lds + n_rows;

    const int h = blockIdx.x;
    if (h >= (int)n_heads) return;
    const int kh        = h % (int)n_k_heads;
    const int tid       = threadIdx.x;
    const int grp       = tid >> 4;
    const int lvv       = tid & 15;
    const unsigned int tile_base = blockIdx.y * COLS;
    const unsigned int vv        = tile_base + lvv;

    const size_t hd = HEAD_DIM;
    const size_t head_base = (size_t)h * HEAD_DIM * HEAD_DIM;

    // Stage the state slice [tile_base .. tile_base+COLS) of head h into
    // LDS, transposing kk×vv → vv×kk. 2048 floats over 64 threads = 32
    // per thread. HBM source is coalesced across the wave (consecutive
    // vvs at the same kk).
    for (int i = tid; i < COLS * HEAD_DIM; i += 64) {
        const int kk     = i >> 4;      // 0..127  (16 vvs per kk)
        const int load_lvv = i & 15;
        state_lds[load_lvv * STATE_STRIDE + kk] =
            state[head_base + (size_t)kk * hd + tile_base + load_lvv];
    }
    // Preload per-row a, b scalars for this head into LDS so the inner
    // row loop reads them at LDS speed (no per-row HBM scalar latency).
    for (unsigned int r = tid; r < n_rows; r += 64) {
        a_lds[r] = a_in_batch[(size_t)r * ab_row_stride + h];
        b_lds[r] = b_in_batch[(size_t)r * ab_row_stride + h];
    }
    __syncthreads();

    const float ssm_a_h   = ssm_a[h];
    const float dt_bias_h = dt_bias[h];

    // The vv >= head_dim guard is moot here (vv = blockIdx.y*16 + lvv
    // with blockIdx.y < HEAD_DIM/COLS=8, so vv < 128 always), but kept
    // as a defensive no-op so the kernel matches the general one's
    // structure if anyone reads them side-by-side.

    for (unsigned int r = 0; r < n_rows; r++) {
        __syncthreads();
        for (int i = tid; i < HEAD_DIM; i += 64) {
            q_lds[i] = q_in_batch[(size_t)r * qk_row_stride + (size_t)kh * HEAD_DIM + i];
            k_lds[i] = k_in_batch[(size_t)r * qk_row_stride + (size_t)kh * HEAD_DIM + i];
        }
        __syncthreads();

        const float dec = __expf(ssm_a_h *
            softplus_stable_r(a_lds[r] + dt_bias_h));
        const float bet = 1.0f / (1.0f + __expf(-b_lds[r]));
        const float vval = v_in_batch[(size_t)r * v_row_stride
                                    + (size_t)h * HEAD_DIM + vv];

        // Per-thread base pointer into this thread's vv-row in the LDS
        // state slice. Stride includes the +1 pad to break bank conflicts.
        float* lds_vv = state_lds + (size_t)lvv * STATE_STRIDE;

        // Per-grp kk traversal: kk = local*4 + grp. Grp g owns the
        // kk %4 == g subset. The cross-grp reduction below sums all
        // four classes — total partial dot product is unchanged
        // (floats: addition is commutative within rounding, and the
        // 4-way reduce is structurally identical), but the LDS bank
        // pattern reduces from 64-way to 2-way conflict (see header).
        float s_arr[QUARTER];
        float pkv = 0.0f;
        #pragma unroll
        for (int local = 0; local < QUARTER; local++) {
            const int kk = local * 4 + grp;
            const float s = lds_vv[kk] * dec;
            s_arr[local] = s;
            pkv += s * k_lds[kk];
        }
        pkv += __shfl_xor(pkv, 16);
        const float kv = pkv + __shfl_xor(pkv, 32);
        const float delta = (vval - kv) * bet;

        // Phase 2 — rank-1 update from the register-held decayed state,
        // accumulate stateᵀ·q, write the updated state to LDS once.
        float pout = 0.0f;
        #pragma unroll
        for (int local = 0; local < QUARTER; local++) {
            const int kk = local * 4 + grp;
            const float s = s_arr[local] + k_lds[kk] * delta;
            lds_vv[kk] = s;
            pout += s * q_lds[kk];
        }
        pout += __shfl_xor(pout, 16);
        const float acc = pout + __shfl_xor(pout, 32);
        if (grp == 0) {
            out_batch[(size_t)r * out_row_stride + (size_t)h * HEAD_DIM + vv] = acc;
        }
    }

    // Write the LDS slice back to HBM. Mirrors the staging load.
    __syncthreads();
    for (int i = tid; i < COLS * HEAD_DIM; i += 64) {
        const int kk        = i >> 4;
        const int store_lvv = i & 15;
        state[head_base + (size_t)kk * hd + tile_base + store_lvv] =
            state_lds[store_lvv * STATE_STRIDE + kk];
    }
}
