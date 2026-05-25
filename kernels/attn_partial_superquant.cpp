// SuperQuant 2-tier decode attention.
//
//   pos in [0, cold_count)                  → Cold tier (turbo3, 3.5 bpv)
//   pos in [cold_count, +warm_count)        → Warm tier (int8 + scale)
//
// Each workgroup handles one (Q head, context slice). Per position, the
// K dot is computed inline against the dequantized K of that tier, the
// scaled score lands in LDS, stable softmax over the slice, V
// accumulation reads the matching tier's V (also inline-dequantized).
//
// CORRECTNESS-first impl. Cold-tier scoring + V accumulation do a full
// per-position cooperative iRHT (128-element FWHT + sign mults) — SLOW
// for long cold runs. Optimization (eager dequant pass + staged scoring)
// is follow-up work; this kernel exists to prove the 2-tier read path
// produces correct attention output.
//
// grid = (n_heads, n_splits). block = 256.
// LDS (dynamic):
//   qf32   [head_dim * 4]    Q row for this head (fp32)
//   scores [chunk * 4]       per-slice scores → softmax probs
//   tmp    [bs * 4]          reduce scratch
//   dqbuf  [head_dim * 4]    scratch for one position's dequanted K or V
//   fwhtw  [ROT_GROUP * 4]   FWHT work buffer

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

constexpr int ROT_GROUP        = 128;
constexpr int BLOCK_SIZE       = 32;
constexpr int BYTES_PER_BLOCK  = 16;
constexpr int BLOCKS_PER_GROUP = ROT_GROUP / BLOCK_SIZE;
constexpr int BYTES_PER_GROUP  = BLOCKS_PER_GROUP * BYTES_PER_BLOCK;

__device__ __constant__ float CENTROIDS_SQ[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};

__device__ __forceinline__
void dequant_group_lds(const unsigned char* __restrict__ src_grp,
                       const int8_t*        __restrict__ signs1,
                       const int8_t*        __restrict__ signs2,
                       float*               __restrict__ out_lds,
                       int                  lane,
                       int                  bs,
                       float*               __restrict__ work)
{
    if (lane < ROT_GROUP) {
        const int blk_idx = lane >> 5;
        const int k       = lane & 31;
        const unsigned char* blk = src_grp + blk_idx * BYTES_PER_BLOCK;
        const uint16_t nb  = (uint16_t)blk[0] | ((uint16_t)blk[1] << 8);
        const float    norm = __half2float(*reinterpret_cast<const __half*>(&nb));
        const unsigned char qs_byte    = blk[2 + (k >> 2)];
        const unsigned char signs_byte = blk[10 + (k >> 3)];
        const unsigned char lo = (qs_byte    >> ((k & 3) * 2)) & 0x3;
        const unsigned char hi = (signs_byte >> (k & 7))       & 0x1;
        const unsigned char code = (hi << 2) | lo;
        const float rotated = CENTROIDS_SQ[code] * norm;
        work[lane] = rotated * (float)signs2[lane];
    }
    __syncthreads();

    if (lane < ROT_GROUP) {
        for (int h = 1; h < ROT_GROUP; h *= 2) {
            if ((lane & h) == 0) {
                const float a = work[lane];
                const float b = work[lane + h];
                work[lane]     = a + b;
                work[lane + h] = a - b;
            }
            __syncthreads();
        }
        const float scale = 1.0f / sqrtf((float)ROT_GROUP);
        out_lds[lane] = work[lane] * scale * (float)signs1[lane];
    } else {
        for (int h = 1; h < ROT_GROUP; h *= 2) __syncthreads();
    }
    __syncthreads();
}

extern "C" __global__
void attn_partial_superquant_f32(
    const float*        __restrict__ q,             // [n_heads, head_dim]
    // Warm tier (int8 + scale)
    const signed char*  __restrict__ warm_k,        // [warm_cap, n_kv, head_dim]
    const float*        __restrict__ warm_ks,       // [warm_cap, n_kv]
    const signed char*  __restrict__ warm_v,
    const float*        __restrict__ warm_vs,
    // Cold tier (turbo3 packed)
    const unsigned char* __restrict__ cold_k,
    const unsigned char* __restrict__ cold_v,
    // RHT sign masks
    const int8_t*       __restrict__ signs1_k,
    const int8_t*       __restrict__ signs2_k,
    const int8_t*       __restrict__ signs1_v,
    const int8_t*       __restrict__ signs2_v,
    // Outputs (partial)
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
    const int tid = threadIdx.x;
    const int bs  = blockDim.x;
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

    // LDS layout:
    //   qf32      [head_dim]     Q row
    //   scores    [chunk]        per-position scores → softmax probs
    //   tmp       [bs]           reduce scratch
    //   dqbuf_K   [head_dim]     K dequant output (used in score phase)
    //   acc_V     [head_dim]     V accumulator (used in P·V phase)
    //   dq_group  [ROT_GROUP]    one-group dequant output (V phase)
    //   fwhtw     [ROT_GROUP]    FWHT cooperative work buffer
    float* qf32     = lds;
    float* scores   = qf32 + head_dim;
    float* tmp      = scores + chunk;
    float* dqbuf    = tmp + bs;                  // K dequant scratch
    float* acc_v    = dqbuf + head_dim;          // V accumulator
    float* dq_group = acc_v + head_dim;          // one-group V dequant
    float* fwhtw    = dq_group + ROT_GROUP;      // FWHT work

    if (slice_len <= 0) {
        for (int d = tid; d < (int)head_dim; d += bs)
            o_partial[((size_t)h * n_splits + sp) * head_dim + d] = 0.0f;
        if (tid == 0) {
            m_partial[(size_t)h * n_splits + sp] = -INFINITY;
            l_partial[(size_t)h * n_splits + sp] = 0.0f;
        }
        return;
    }

    const float* qh = q + (size_t)h * head_dim;
    for (int i = tid; i < (int)head_dim; i += bs) qf32[i] = qh[i];
    __syncthreads();

    const unsigned int groups_per_head = head_dim / ROT_GROUP;
    const size_t cold_row_bytes = (size_t)n_kv_heads * groups_per_head * BYTES_PER_GROUP;
    const size_t warm_row_elem  = (size_t)n_kv_heads * head_dim;

    // === Score phase ===
    for (int i = 0; i < slice_len; i++) {
        const int t = slice_start + i;
        float score;

        if (t < (int)cold_count) {
            const unsigned char* k_row = cold_k
                + (size_t)t * cold_row_bytes
                + (size_t)kv_h * groups_per_head * BYTES_PER_GROUP;
            for (unsigned int g = 0; g < groups_per_head; g++) {
                dequant_group_lds(k_row + g * BYTES_PER_GROUP,
                                  signs1_k, signs2_k,
                                  dqbuf + g * ROT_GROUP,
                                  tid, bs, fwhtw);
            }
            float local = 0.0f;
            for (int d = tid; d < (int)head_dim; d += bs) local += qf32[d] * dqbuf[d];
            tmp[tid] = local;
            __syncthreads();
            for (int r = bs >> 1; r > 0; r >>= 1) {
                if (tid < r) tmp[tid] += tmp[tid + r];
                __syncthreads();
            }
            score = tmp[0] * scaling;
        } else {
            const int wt = t - (int)cold_count;
            const signed char* k_row = warm_k + (size_t)wt * warm_row_elem
                                              + (size_t)kv_h * head_dim;
            const float dk = warm_ks[(size_t)wt * n_kv_heads + kv_h];
            float local = 0.0f;
            for (int d = tid; d < (int)head_dim; d += bs) local += qf32[d] * (float)k_row[d];
            tmp[tid] = local;
            __syncthreads();
            for (int r = bs >> 1; r > 0; r >>= 1) {
                if (tid < r) tmp[tid] += tmp[tid + r];
                __syncthreads();
            }
            score = tmp[0] * dk * scaling;
        }
        if (tid == 0) scores[i] = score;
        __syncthreads();
    }

    // === Stable softmax ===
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

    // === P·V — position-major loop. All threads cooperate on the
    // SAME group at a time (cold tier requires this for the FWHT
    // cooperative reduce). Accumulator lives in LDS so dq_group can
    // overwrite freely; final per-d output read from acc_v.
    for (int d = tid; d < (int)head_dim; d += bs) acc_v[d] = 0.0f;
    __syncthreads();

    for (int i = 0; i < slice_len; i++) {
        const int t = slice_start + i;
        const float w = scores[i];
        if (t < (int)cold_count) {
            // Cold V: walk the rotation groups one at a time so all
            // threads cooperate on the same FWHT. After each group's
            // dequant, threads owning a d in that group's range
            // accumulate w × dq_group[d - g*ROT_GROUP] into acc_v[d].
            const unsigned char* v_row_base = cold_v
                + (size_t)t * cold_row_bytes
                + (size_t)kv_h * groups_per_head * BYTES_PER_GROUP;
            for (unsigned int g = 0; g < groups_per_head; g++) {
                dequant_group_lds(v_row_base + g * BYTES_PER_GROUP,
                                  signs1_v, signs2_v,
                                  dq_group, tid, bs, fwhtw);
                const int g_lo = g * ROT_GROUP;
                const int g_hi = g_lo + ROT_GROUP;
                for (int d = tid; d < (int)head_dim; d += bs) {
                    if (d >= g_lo && d < g_hi) {
                        acc_v[d] += w * dq_group[d - g_lo];
                    }
                }
                __syncthreads();
            }
        } else {
            const int wt = t - (int)cold_count;
            const signed char* v_row = warm_v + (size_t)wt * warm_row_elem
                                              + (size_t)kv_h * head_dim;
            const float dv = warm_vs[(size_t)wt * n_kv_heads + kv_h];
            for (int d = tid; d < (int)head_dim; d += bs) {
                acc_v[d] += w * ((float)v_row[d] * dv);
            }
            __syncthreads();
        }
    }

    for (int d = tid; d < (int)head_dim; d += bs) {
        o_partial[((size_t)h * n_splits + sp) * head_dim + d] = acc_v[d];
    }
    if (tid == 0) {
        m_partial[(size_t)h * n_splits + sp] = mx;
        l_partial[(size_t)h * n_splits + sp] = l;
    }
}
