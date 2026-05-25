// SuperQuant 3-tier decode attention — FlashDecoding split-K shape but
// reads K/V from one of three tiers based on the cached position's tier:
//
//   pos in [0, cold_count)              → Cold tier, turbo3 (3.5 bpv)
//   pos in [cold_count, +warm_count)    → Warm tier, int8 + per-(slot,head) scale
//   pos in [+warm_count, +hot_count)    → Hot tier, fp16
//
// Each workgroup handles one (Q head, context slice). Per position, the
// K dot is computed inline against the dequantized K of that tier, the
// scaled score lands in LDS, stable softmax happens over the slice, and
// V accumulation reads the matching tier's V (also inline-dequantized).
//
// CORRECTNESS-first impl. Cold-tier dequant does the full per-position
// inverse RHT (128-element FWHT + sign mults) cooperatively in LDS for
// each cold position scored. That's ~7 LDS syncs per cold position —
// SLOW for long-context cold runs. Optimization (eager dequant pass +
// staged attention) is a follow-up; this kernel exists to prove the
// 3-tier read path is correct end-to-end.
//
// grid = (n_heads, n_splits). block = 256.
// LDS (dynamic):
//   qf32  [head_dim * 4]      Q row for this head (fp32)
//   scores [chunk * 4]        per-slice scores → softmax probs
//   tmp    [bs * 4]           reduce scratch
//   dqbuf  [head_dim * 4]     scratch for one position's dequanted K or V

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

// Cooperative dequant + iRHT of one (kv_head) position's 128-element
// rotation group from a turbo3 cache row. Output goes into `out[]`
// (head_dim-wide LDS slot at offset `grp_off`). Caller __syncs after.
__device__ __forceinline__
void dequant_group_lds(const unsigned char* __restrict__ src_grp,
                       const int8_t*        __restrict__ signs1,
                       const int8_t*        __restrict__ signs2,
                       float*               __restrict__ out_lds,
                       int                  lane,
                       int                  bs,
                       float*               __restrict__ work)
{
    // Each block has 256 threads but the group is 128 — only lanes
    // [0, 128) participate in the per-element work; the rest idle.
    if (lane < ROT_GROUP) {
        // Read code for lane's position.
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
        // FWHT-128 in place.
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
        // Idle lanes still must hit the FWHT __syncthreads (7 of them).
        for (int h = 1; h < ROT_GROUP; h *= 2) __syncthreads();
    }
    __syncthreads();
}

extern "C" __global__
void attn_partial_superquant_f32(
    // Q (current token)
    const float*        __restrict__ q,             // [n_heads, head_dim]
    // Hot tier (fp16)
    const __half*       __restrict__ hot_k,         // [hot_cap, n_kv, head_dim]
    const __half*       __restrict__ hot_v,
    // Warm tier (int8 + scale)
    const signed char*  __restrict__ warm_k,        // [warm_cap, n_kv, head_dim]
    const float*        __restrict__ warm_ks,       // [warm_cap, n_kv]
    const signed char*  __restrict__ warm_v,
    const float*        __restrict__ warm_vs,
    // Cold tier (turbo3 packed)
    const unsigned char* __restrict__ cold_k,       // [cold_cap, n_kv, head_dim/32 * 16]
    const unsigned char* __restrict__ cold_v,
    // RHT sign masks (resident)
    const int8_t*       __restrict__ signs1_k,
    const int8_t*       __restrict__ signs2_k,
    const int8_t*       __restrict__ signs1_v,
    const int8_t*       __restrict__ signs2_v,
    // Outputs (partial)
    float*              __restrict__ o_partial,     // [n_heads, n_splits, head_dim]
    float*              __restrict__ m_partial,     // [n_heads, n_splits]
    float*              __restrict__ l_partial,     // [n_heads, n_splits]
    // Shape + tier counts
    unsigned int n_heads,
    unsigned int n_kv_heads,
    unsigned int head_dim,
    unsigned int cold_count,
    unsigned int warm_count,
    unsigned int hot_count,
    float        scaling,
    unsigned int n_splits)
{
    extern __shared__ float lds[];
    const int h   = blockIdx.x;
    const int sp  = blockIdx.y;
    if (h >= (int)n_heads) return;
    const int tid = threadIdx.x;
    const int bs  = blockDim.x;
    const int groups = (int)(n_heads / n_kv_heads);
    const int kv_h   = h / groups;
    const unsigned int total_len = cold_count + warm_count + hot_count;
    if (total_len == 0) return;

    // Slice [slice_start, slice_end) — block owns split `sp`.
    const int chunk = ((int)total_len + (int)n_splits - 1) / (int)n_splits;
    const int slice_start = sp * chunk;
    int slice_end = (int)total_len;
    if (slice_start + chunk < slice_end) slice_end = slice_start + chunk;
    const int slice_len = slice_end - slice_start;

    float* qf32  = lds;
    float* scores = qf32 + head_dim;        // [chunk]
    float* tmp    = scores + chunk;         // [bs]
    float* dqbuf  = tmp + bs;               // [head_dim] — one position's dequant scratch
    float* fwhtw  = dqbuf + head_dim;       // [ROT_GROUP] — FWHT work buffer

    // Empty split: neutral partials.
    if (slice_len <= 0) {
        for (int d = tid; d < (int)head_dim; d += bs)
            o_partial[((size_t)h * n_splits + sp) * head_dim + d] = 0.0f;
        if (tid == 0) {
            m_partial[(size_t)h * n_splits + sp] = -INFINITY;
            l_partial[(size_t)h * n_splits + sp] = 0.0f;
        }
        return;
    }

    // Stage Q row into LDS (fp32).
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
            // Cold tier — dequant K cooperatively into dqbuf.
            const unsigned char* k_row = cold_k
                + (size_t)t * cold_row_bytes
                + (size_t)kv_h * groups_per_head * BYTES_PER_GROUP;
            for (unsigned int g = 0; g < groups_per_head; g++) {
                dequant_group_lds(k_row + g * BYTES_PER_GROUP,
                                  signs1_k, signs2_k,
                                  dqbuf + g * ROT_GROUP,
                                  tid, bs, fwhtw);
            }
            // Dot Q · K (fp32 × fp32).
            float local = 0.0f;
            for (int d = tid; d < (int)head_dim; d += bs) local += qf32[d] * dqbuf[d];
            tmp[tid] = local;
            __syncthreads();
            for (int r = bs >> 1; r > 0; r >>= 1) {
                if (tid < r) tmp[tid] += tmp[tid + r];
                __syncthreads();
            }
            score = tmp[0] * scaling;
        } else if (t < (int)(cold_count + warm_count)) {
            // Warm tier — int8 × fp32 dot (precision-equivalent to the
            // existing q8 attention; we skip the Q-quantize trick here
            // for clarity since this kernel's bottleneck is Cold dequant
            // anyway).
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
        } else {
            // Hot tier — fp16 × fp32 dot.
            const int ht = t - (int)cold_count - (int)warm_count;
            const __half* k_row = hot_k + (size_t)ht * (n_kv_heads * head_dim)
                                          + (size_t)kv_h * head_dim;
            float local = 0.0f;
            for (int d = tid; d < (int)head_dim; d += bs)
                local += qf32[d] * __half2float(k_row[d]);
            tmp[tid] = local;
            __syncthreads();
            for (int r = bs >> 1; r > 0; r >>= 1) {
                if (tid < r) tmp[tid] += tmp[tid + r];
                __syncthreads();
            }
            score = tmp[0] * scaling;
        }
        if (tid == 0) scores[i] = score;
        __syncthreads();
    }

    // === Stable softmax over the slice ===
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

    // === P·V — position-major loop so cold V dequant can be
    // cooperative (LDS sync between FWHT stages requires all
    // threads in lockstep). Each thread accumulates its own d.
    //
    // For head_dim > bs, the d range is strided: thread tid covers
    // d = tid, tid+bs, tid+2*bs, ... We hold the accumulator for the
    // CURRENT d-stride iteration in a register; outer loop is d.
    for (int d = tid; d < (int)head_dim; d += bs) {
        float my_acc = 0.0f;
        for (int i = 0; i < slice_len; i++) {
            const int t = slice_start + i;
            const float w = scores[i];
            float v_val;
            if (t < (int)cold_count) {
                // Cooperative dequant of the rotation group containing d.
                // dqbuf only holds ONE group's worth (head_dim regions);
                // we dequant the group that contains element d. All
                // threads cooperate even though only the lane reading
                // d_in_group benefits — the cost is dominated by the
                // FWHT itself, not redundant reads.
                const int g = d / ROT_GROUP;
                const unsigned char* v_grp = cold_v
                    + (size_t)t * cold_row_bytes
                    + (size_t)kv_h * groups_per_head * BYTES_PER_GROUP
                    + g * BYTES_PER_GROUP;
                dequant_group_lds(v_grp, signs1_v, signs2_v,
                                  dqbuf + g * ROT_GROUP, tid, bs, fwhtw);
                v_val = dqbuf[d];
            } else if (t < (int)(cold_count + warm_count)) {
                const int wt = t - (int)cold_count;
                const signed char* v_row = warm_v + (size_t)wt * warm_row_elem
                                                   + (size_t)kv_h * head_dim;
                const float dv = warm_vs[(size_t)wt * n_kv_heads + kv_h];
                v_val = (float)v_row[d] * dv;
            } else {
                const int ht = t - (int)cold_count - (int)warm_count;
                const __half* v_row = hot_v + (size_t)ht * (n_kv_heads * head_dim)
                                              + (size_t)kv_h * head_dim;
                v_val = __half2float(v_row[d]);
            }
            my_acc += w * v_val;
        }
        o_partial[((size_t)h * n_splits + sp) * head_dim + d] = my_acc;
    }
    if (tid == 0) {
        m_partial[(size_t)h * n_splits + sp] = mx;
        l_partial[(size_t)h * n_splits + sp] = l;
    }
}
