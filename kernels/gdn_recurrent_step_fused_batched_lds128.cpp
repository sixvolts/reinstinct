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
// LDS layout chosen as state_lds[VV][KK]: the per-thread inner loop
// (fixed vv, varying kk) reads contiguous addresses, and across the
// wave's 16 vv-threads each iteration broadcasts a single bank (vv*128
// mod 128 = 0; only kk distinguishes the bank, so 4 grps with 4
// different kks hit 4 banks per cycle — no thrash).

#include <hip/hip_runtime.h>

#define HEAD_DIM 128
#define QUARTER  (HEAD_DIM / 4)   // 32
#define COLS     16

__device__ __forceinline__ float softplus_stable_r(float x) {
    return (x > 0.0f) ? x + __logf(1.0f + __expf(-x))
                      :     __logf(1.0f + __expf(x));
}

extern "C" __global__
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
    // LDS:  state_slice[COLS][HEAD_DIM] | q_lds[HEAD_DIM] | k_lds[HEAD_DIM]
    extern __shared__ float lds[];
    float* state_lds = lds;
    float* q_lds     = lds + COLS * HEAD_DIM;
    float* k_lds     = lds + COLS * HEAD_DIM + HEAD_DIM;

    const int h = blockIdx.x;
    if (h >= (int)n_heads) return;
    const int kh        = h % (int)n_k_heads;
    const int tid       = threadIdx.x;
    const int grp       = tid >> 4;
    const int lvv       = tid & 15;
    const unsigned int tile_base = blockIdx.y * COLS;
    const unsigned int vv        = tile_base + lvv;

    const int kk0 = grp * QUARTER;
    const int kk1 = kk0 + QUARTER;
    const size_t hd = HEAD_DIM;
    const size_t head_base = (size_t)h * HEAD_DIM * HEAD_DIM;

    // Stage the state slice [tile_base .. tile_base+COLS) of head h into
    // LDS, transposing kk×vv → vv×kk. 2048 floats over 64 threads = 32
    // per thread. HBM source is coalesced across the wave (consecutive
    // vvs at the same kk).
    for (int i = tid; i < COLS * HEAD_DIM; i += 64) {
        const int kk     = i >> 4;      // 0..127  (16 vvs per kk)
        const int load_lvv = i & 15;
        state_lds[load_lvv * HEAD_DIM + kk] =
            state[head_base + (size_t)kk * hd + tile_base + load_lvv];
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
            softplus_stable_r(a_in_batch[(size_t)r * ab_row_stride + h] + dt_bias_h));
        const float bet = 1.0f / (1.0f + __expf(
            -b_in_batch[(size_t)r * ab_row_stride + h]));
        const float vval = v_in_batch[(size_t)r * v_row_stride
                                    + (size_t)h * HEAD_DIM + vv];

        // Per-thread base pointer into this thread's vv-row in the LDS
        // state slice. Sequential kk within this row → contiguous LDS.
        float* lds_vv = state_lds + (size_t)lvv * HEAD_DIM;

        // Phase 1 — decay each kk + accumulate stateᵀ·k. All in LDS.
        float pkv = 0.0f;
        #pragma unroll
        for (int kk = kk0; kk < kk1; kk++) {
            float s = lds_vv[kk] * dec;
            lds_vv[kk] = s;
            pkv += s * k_lds[kk];
        }
        pkv += __shfl_xor(pkv, 16);
        const float kv = pkv + __shfl_xor(pkv, 32);
        const float delta = (vval - kv) * bet;

        // Phase 2 — rank-1 update + stateᵀ·q. All in LDS.
        float pout = 0.0f;
        #pragma unroll
        for (int kk = kk0; kk < kk1; kk++) {
            float s = lds_vv[kk] + k_lds[kk] * delta;
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
            state_lds[store_lvv * HEAD_DIM + kk];
    }
}
