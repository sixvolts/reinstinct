// SuperQuant 2-tier decode attention, WAVE-PARALLEL variant.
//
// Optimization vs attn_partial_superquant_rs.cpp: position dispatch is
// parallelized ACROSS wavefronts. Each of the workgroup's 4 wave64
// units handles a different cached position simultaneously — the
// per-position cooperative dequant collapses from 7-sync FWHT to
// zero-sync within-wave LDS-write + wave64-reduce.
//
// Score phase per position:
//   - One wave handles one position end-to-end
//   - Each lane (k in 0..63) dequants TWO elements (k, k+64) of the
//     128-element rotation group: code → centroid × norm
//   - Multiplies its two dequant values by Q_rot at the matching d
//   - Wave64 reduce → score for this (wave, position) pair
//
// V phase per position:
//   - Same wave-parallel dispatch
//   - Each wave accumulates into per-wave acc_v_cold buffer (no cross-
//     wave write conflict)
//   - After all positions: cross-wave reduce of per_wave_acc_v_cold,
//     apply iRHT once per group, sum with acc_warm
//
// Per-position sync barriers drop from ~7 (FWHT stages) to 0 inside a
// single wave's iteration. With 4 waves doing 4 positions in parallel,
// throughput scales ~4× on cold-heavy workloads.
//
// LDS (dynamic):
//   q_rot [head_dim] | q_orig [head_dim]  // Q for warm scoring
//   scores [chunk] | tmp [bs]
//   per_wave_acc_v_cold [4 × head_dim]
//   acc_warm [head_dim] | acc_cold [head_dim]
//   fwhtw [ROT_GROUP]                     // final iRHT scratch

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include "gfx906_dpp.h"

constexpr int ROT_GROUP        = 128;
constexpr int BLOCK_SIZE       = 32;
constexpr int BYTES_PER_BLOCK  = 16;
constexpr int BLOCKS_PER_GROUP = ROT_GROUP / BLOCK_SIZE;
constexpr int BYTES_PER_GROUP  = BLOCKS_PER_GROUP * BYTES_PER_BLOCK;
constexpr int WAVE_SIZE        = 64;
constexpr int N_WAVES          = 4;   // block = 256 = 4 × 64

__device__ __constant__ float CENTROIDS_WP[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};

// Dequant one element of a turbo3 block. `byte_off` is the position
// within the 32-element block (0..31). `blk` points to the start of
// the 16-byte block. Returns centroid × norm.
__device__ __forceinline__
float dequant_one(const unsigned char* __restrict__ blk, int byte_off, float norm)
{
    const unsigned char qs_byte    = blk[2 + (byte_off >> 2)];
    const unsigned char signs_byte = blk[10 + (byte_off >> 3)];
    const unsigned char lo = (qs_byte    >> ((byte_off & 3) * 2)) & 0x3;
    const unsigned char hi = (signs_byte >> (byte_off & 7))       & 0x1;
    const unsigned char code = (hi << 2) | lo;
    return CENTROIDS_WP[code] * norm;
}

// One wave (64 lanes) scores ONE cold position against Q_rot. Returns
// the score in lane 0 (and any other lane via the wave reduce).
__device__ __forceinline__
float wave_score_cold(const unsigned char* __restrict__ k_row_base,
                       const float*         __restrict__ q_rot,
                       int                  lane,
                       int                  groups_per_head)
{
    float local = 0.0f;
    for (int g = 0; g < groups_per_head; g++) {
        const unsigned char* grp = k_row_base + g * BYTES_PER_GROUP;
        // Two values per lane: positions `lane` and `lane + 64`
        // within the 128-element rotation group.
        #pragma unroll
        for (int sub = 0; sub < 2; sub++) {
            const int k = lane + sub * WAVE_SIZE;
            const int blk_idx = k >> 5;
            const int k_in_blk = k & 31;
            const unsigned char* blk = grp + blk_idx * BYTES_PER_BLOCK;
            // Norm shared across the block — read once. The compiler
            // will hoist this when k_in_blk varies but blk_idx
            // doesn't; here it varies, so each thread reads its own.
            const uint16_t nb = (uint16_t)blk[0] | ((uint16_t)blk[1] << 8);
            const float norm = __half2float(*reinterpret_cast<const __half*>(&nb));
            const float v    = dequant_one(blk, k_in_blk, norm);
            local += q_rot[g * ROT_GROUP + k] * v;
        }
    }
    return wave64_reduce_add_f32(local);
}

// One wave scores a warm position via int8 × fp32 dot against original Q.
__device__ __forceinline__
float wave_score_warm(const signed char* __restrict__ k_row,
                       const float*       __restrict__ q,
                       float              dk,
                       int                lane,
                       int                head_dim)
{
    float local = 0.0f;
    // Each lane handles head_dim/64 elements (stride 64).
    for (int d = lane; d < head_dim; d += WAVE_SIZE) {
        local += q[d] * (float)k_row[d];
    }
    return wave64_reduce_add_f32(local) * dk;
}

// One wave accumulates V[pos] into per-wave acc_v_cold (with weight w).
// dq is computed inline; no LDS-scratch needed because each thread
// writes to a unique d within the per-wave accumulator.
__device__ __forceinline__
void wave_accumulate_cold_v(const unsigned char* __restrict__ v_row_base,
                             float*               __restrict__ acc_v,
                             float                w,
                             int                  lane,
                             int                  groups_per_head)
{
    for (int g = 0; g < groups_per_head; g++) {
        const unsigned char* grp = v_row_base + g * BYTES_PER_GROUP;
        #pragma unroll
        for (int sub = 0; sub < 2; sub++) {
            const int k = lane + sub * WAVE_SIZE;
            const int blk_idx = k >> 5;
            const int k_in_blk = k & 31;
            const unsigned char* blk = grp + blk_idx * BYTES_PER_BLOCK;
            const uint16_t nb = (uint16_t)blk[0] | ((uint16_t)blk[1] << 8);
            const float norm = __half2float(*reinterpret_cast<const __half*>(&nb));
            const float v    = dequant_one(blk, k_in_blk, norm);
            acc_v[g * ROT_GROUP + k] += w * v;
        }
    }
}

// Cooperative iRHT-128 across the workgroup (256 threads). Same shape
// as attn_partial_superquant_rs.cpp's irht_group_inplace.
__device__ __forceinline__
void irht_group_inplace(float*        __restrict__ data,
                         const int8_t* __restrict__ signs1,
                         const int8_t* __restrict__ signs2,
                         int           lane_in_block,
                         float*        __restrict__ work)
{
    if (lane_in_block < ROT_GROUP) {
        work[lane_in_block] = data[lane_in_block] * (float)signs2[lane_in_block];
    }
    __syncthreads();
    if (lane_in_block < ROT_GROUP) {
        for (int h = 1; h < ROT_GROUP; h *= 2) {
            if ((lane_in_block & h) == 0) {
                const float a = work[lane_in_block];
                const float b = work[lane_in_block + h];
                work[lane_in_block]     = a + b;
                work[lane_in_block + h] = a - b;
            }
            __syncthreads();
        }
        const float scale = 1.0f / sqrtf((float)ROT_GROUP);
        data[lane_in_block] = work[lane_in_block] * scale * (float)signs1[lane_in_block];
    } else {
        for (int h = 1; h < ROT_GROUP; h *= 2) __syncthreads();
    }
    __syncthreads();
}

extern "C" __global__
void attn_partial_superquant_wp_f32(
    const float*        __restrict__ q,
    const float*        __restrict__ q_rot,
    const signed char*  __restrict__ warm_k,
    const float*        __restrict__ warm_ks,
    const signed char*  __restrict__ warm_v,
    const float*        __restrict__ warm_vs,
    const unsigned char* __restrict__ cold_k,
    const unsigned char* __restrict__ cold_v,
    const int8_t*       __restrict__ signs1_v,
    const int8_t*       __restrict__ signs2_v,
    float*              __restrict__ o_partial,
    float*              __restrict__ m_partial,
    float*              __restrict__ l_partial,
    unsigned int n_heads,
    unsigned int n_kv_heads,
    unsigned int head_dim,
    unsigned int cold_count,
    unsigned int warm_count,
    float        scaling)
{
    extern __shared__ float lds[];
    const int h   = blockIdx.x;
    const int sp  = blockIdx.y;
    if (h >= (int)n_heads) return;
    const int tid  = threadIdx.x;
    const int bs   = blockDim.x;
    const int wave = tid >> 6;          // 0..3
    const int lane = tid & 63;          // 0..63 within wave
    const int groups = (int)(n_heads / n_kv_heads);
    const int kv_h   = h / groups;
    const unsigned int total_len = cold_count + warm_count;
    if (total_len == 0) return;
    const int n_splits = (int)gridDim.y;

    const int chunk = ((int)total_len + n_splits - 1) / n_splits;
    const int slice_start = sp * chunk;
    int slice_end = (int)total_len;
    if (slice_start + chunk < slice_end) slice_end = slice_start + chunk;
    const int slice_len = slice_end - slice_start;

    float* qf32     = lds;
    float* qrot     = qf32 + head_dim;
    float* scores   = qrot + head_dim;
    float* tmp      = scores + chunk;
    float* pwave_v  = tmp + bs;                          // [N_WAVES, head_dim]
    float* acc_w    = pwave_v + N_WAVES * head_dim;
    float* acc_c    = acc_w + head_dim;
    float* fwhtw    = acc_c + head_dim;

    if (slice_len <= 0) {
        for (int d = tid; d < (int)head_dim; d += bs)
            o_partial[((size_t)h * n_splits + sp) * head_dim + d] = 0.0f;
        if (tid == 0) {
            m_partial[(size_t)h * n_splits + sp] = -INFINITY;
            l_partial[(size_t)h * n_splits + sp] = 0.0f;
        }
        return;
    }

    // Stage Q + Q_rot rows.
    const float* qh    = q     + (size_t)h * head_dim;
    const float* qroth = q_rot + (size_t)h * head_dim;
    for (int d = tid; d < (int)head_dim; d += bs) {
        qf32[d] = qh[d];
        qrot[d] = qroth[d];
    }
    // Init per-wave V accumulator + warm + cold combined accumulators.
    for (int w = 0; w < N_WAVES; w++) {
        for (int d = tid; d < (int)head_dim; d += bs) {
            pwave_v[w * head_dim + d] = 0.0f;
        }
    }
    for (int d = tid; d < (int)head_dim; d += bs) {
        acc_w[d] = 0.0f;
        acc_c[d] = 0.0f;
    }
    __syncthreads();

    const unsigned int groups_per_head = head_dim / ROT_GROUP;
    const size_t cold_row_bytes = (size_t)n_kv_heads * groups_per_head * BYTES_PER_GROUP;
    const size_t warm_row_elem  = (size_t)n_kv_heads * head_dim;

    // === Score phase: 4 waves each handle a strided position ===
    for (int i_base = 0; i_base < slice_len; i_base += N_WAVES) {
        const int i = i_base + wave;
        float score = 0.0f;
        bool active = (i < slice_len);
        if (active) {
            const int t = slice_start + i;
            if (t < (int)cold_count) {
                const unsigned char* k_row = cold_k
                    + (size_t)t * cold_row_bytes
                    + (size_t)kv_h * groups_per_head * BYTES_PER_GROUP;
                score = wave_score_cold(k_row, qrot, lane, (int)groups_per_head) * scaling;
            } else {
                const int wt = t - (int)cold_count;
                const signed char* k_row = warm_k + (size_t)wt * warm_row_elem
                                                  + (size_t)kv_h * head_dim;
                const float dk = warm_ks[(size_t)wt * n_kv_heads + kv_h];
                score = wave_score_warm(k_row, qf32, dk, lane, (int)head_dim) * scaling;
            }
        }
        if (active && lane == 0) scores[i] = score;
        // No __syncthreads here — waves write to disjoint scores[i] slots.
    }
    __syncthreads();

    // === Stable softmax (all 256 threads) ===
    float m = -INFINITY;
    for (int i = tid; i < slice_len; i += bs) m = fmaxf(m, scores[i]);
    tmp[tid] = m;
    __syncthreads();
    for (int r = bs >> 1; r > 0; r >>= 1) {
        if (tid < r) tmp[tid] = fmaxf(tmp[tid], tmp[tid + r]);
        __syncthreads();
    }
    const float mx = tmp[0];
    __syncthreads();
    float sum = 0.0f;
    for (int i = tid; i < slice_len; i += bs) {
        float e = __expf(scores[i] - mx);
        scores[i] = e;
        sum += e;
    }
    tmp[tid] = sum;
    __syncthreads();
    for (int r = bs >> 1; r > 0; r >>= 1) {
        if (tid < r) tmp[tid] += tmp[tid + r];
        __syncthreads();
    }
    const float l = tmp[0];
    __syncthreads();

    // === V phase: 4 waves each handle a strided position ===
    // Cold positions accumulate to per_wave_v_cold[wave].
    // Warm positions: each wave's lane sums w × v_row[d=lane,lane+64,…]
    // into the same per_wave_acc — but that's wrong because warm
    // shouldn't be iRHT'd. So warm gets its own pass after.
    for (int i_base = 0; i_base < slice_len; i_base += N_WAVES) {
        const int i = i_base + wave;
        if (i >= slice_len) continue;
        const int t = slice_start + i;
        if (t < (int)cold_count) {
            const float w_score = scores[i];
            const unsigned char* v_row_base = cold_v
                + (size_t)t * cold_row_bytes
                + (size_t)kv_h * groups_per_head * BYTES_PER_GROUP;
            wave_accumulate_cold_v(v_row_base,
                                    pwave_v + wave * head_dim,
                                    w_score, lane, (int)groups_per_head);
        }
    }
    __syncthreads();

    // === Warm V — separate pass, all 256 threads cooperate ===
    for (int i = 0; i < slice_len; i++) {
        const int t = slice_start + i;
        if (t < (int)cold_count) continue;
        const float w_score = scores[i];
        const int wt = t - (int)cold_count;
        const signed char* v_row = warm_v + (size_t)wt * warm_row_elem
                                          + (size_t)kv_h * head_dim;
        const float dv = warm_vs[(size_t)wt * n_kv_heads + kv_h];
        for (int d = tid; d < (int)head_dim; d += bs) {
            acc_w[d] += w_score * ((float)v_row[d] * dv);
        }
    }
    __syncthreads();

    // === Reduce per_wave_v_cold across the 4 waves into acc_c ===
    for (int d = tid; d < (int)head_dim; d += bs) {
        float s = 0.0f;
        #pragma unroll
        for (int w = 0; w < N_WAVES; w++) {
            s += pwave_v[w * head_dim + d];
        }
        acc_c[d] = s;
    }
    __syncthreads();

    // === iRHT acc_c per group, then sum with acc_w ===
    for (unsigned int g = 0; g < groups_per_head; g++) {
        irht_group_inplace(acc_c + g * ROT_GROUP, signs1_v, signs2_v, tid, fwhtw);
    }

    for (int d = tid; d < (int)head_dim; d += bs) {
        o_partial[((size_t)h * n_splits + sp) * head_dim + d] = acc_w[d] + acc_c[d];
    }
    if (tid == 0) {
        m_partial[(size_t)h * n_splits + sp] = mx;
        l_partial[(size_t)h * n_splits + sp] = l;
    }
}
