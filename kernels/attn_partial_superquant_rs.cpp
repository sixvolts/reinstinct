// SuperQuant 2-tier decode attention, rotated-space variant.
//
// Key optimization vs attn_partial_superquant.cpp: we pre-rotate Q
// ONCE per attention call (rotate_q_rht_f32 kernel), then score cold
// K in rotated space — no per-position iRHT. V accumulates in
// rotated space too; ONE iRHT per (head, rotation group) at the end
// recovers the un-rotated output.
//
// Per-cached-position cost drops from "cooperative FWHT + dot" to
// just "centroid lookup × norm + dot". For an attention call with N
// cold positions, FWHT count goes from N*groups_per_head to
// groups_per_head (one per group at the end) — a factor-of-N
// reduction in synchronization overhead.
//
// Algorithm:
//   1. Score: for cold i, score[i] = <Q_rot, K_stored[i] · norm[i]> · scaling
//             for warm i, score[i] = <Q,    K_warm[i] · scale[i]> · scaling
//   2. softmax(scores)
//   3. V accumulation, separate per tier:
//        acc_rot  [d] = Σ_{i in cold} scores[i] × V_stored[i, d] · norm[i]
//        acc_warm [d] = Σ_{i in warm} scores[i] × V_warm[i, d] · scale[i]
//   4. iRHT(acc_rot) per group with V's signs → contribution in original space
//   5. output[d] = acc_warm[d] + iRHT(acc_rot)[d]
//
// Grid: (n_heads, n_splits). Block: 256 threads.
// LDS:
//   qf32  [head_dim] | qrot [head_dim] | scores [chunk] | tmp [bs]
//   acc_warm [head_dim] | acc_rot [head_dim] | dq_group [ROT_GROUP]
//   fwhtw [ROT_GROUP]

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

constexpr int ROT_GROUP        = 128;
constexpr int BLOCK_SIZE       = 32;
constexpr int BYTES_PER_BLOCK  = 16;
constexpr int BLOCKS_PER_GROUP = ROT_GROUP / BLOCK_SIZE;
constexpr int BYTES_PER_GROUP  = BLOCKS_PER_GROUP * BYTES_PER_BLOCK;

__device__ __constant__ float CENTROIDS_RS[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};

// Cheap dequant: unpack 128 codes, look up centroid × norm. No FWHT,
// no sign multiplies. Output is in RHT space.
__device__ __forceinline__
void dequant_group_centroid(const unsigned char* __restrict__ src_grp,
                             float*               __restrict__ out_lds,
                             int                  lane)
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
        out_lds[lane] = CENTROIDS_RS[code] * norm;
    }
    __syncthreads();
}

// One iRHT-128 pass cooperatively: signs2 × FWHT × (1/√N) × signs1
// applied in place to `data` (length ROT_GROUP). Used at the end of
// attention to un-rotate the per-head accumulator.
__device__ __forceinline__
void irht_group_inplace(float*        __restrict__ data,
                         const int8_t* __restrict__ signs1,
                         const int8_t* __restrict__ signs2,
                         int           lane,
                         float*        __restrict__ work)
{
    if (lane < ROT_GROUP) {
        work[lane] = data[lane] * (float)signs2[lane];
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
        data[lane] = work[lane] * scale * (float)signs1[lane];
    } else {
        for (int h = 1; h < ROT_GROUP; h *= 2) __syncthreads();
    }
    __syncthreads();
}

extern "C" __global__
void attn_partial_superquant_rs_f32(
    const float*        __restrict__ q,             // [n_heads, head_dim]  (original)
    const float*        __restrict__ q_rot,         // [n_heads, head_dim]  (R_K · Q)
    // Warm tier
    const signed char*  __restrict__ warm_k,
    const float*        __restrict__ warm_ks,
    const signed char*  __restrict__ warm_v,
    const float*        __restrict__ warm_vs,
    // Cold tier (turbo3 in rotated space)
    const unsigned char* __restrict__ cold_k,
    const unsigned char* __restrict__ cold_v,
    // V sign masks (for the final iRHT on acc_rot)
    const int8_t*       __restrict__ signs1_v,
    const int8_t*       __restrict__ signs2_v,
    // Outputs (partial — caller merges via attn_merge)
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

    float* qf32    = lds;
    float* qrot    = qf32 + head_dim;
    float* scores  = qrot + head_dim;
    float* tmp     = scores + chunk;
    float* acc_w   = tmp + bs;          // warm-tier (original-space) accumulator
    float* acc_r   = acc_w + head_dim;  // cold-tier (rotated-space) accumulator
    float* dq_grp  = acc_r + head_dim;  // one-group dequant scratch
    float* fwhtw   = dq_grp + ROT_GROUP;

    if (slice_len <= 0) {
        for (int d = tid; d < (int)head_dim; d += bs)
            o_partial[((size_t)h * n_splits + sp) * head_dim + d] = 0.0f;
        if (tid == 0) {
            m_partial[(size_t)h * n_splits + sp] = -INFINITY;
            l_partial[(size_t)h * n_splits + sp] = 0.0f;
        }
        return;
    }

    // Stage Q + Q_rot rows into LDS.
    const float* qh    = q     + (size_t)h * head_dim;
    const float* qroth = q_rot + (size_t)h * head_dim;
    for (int d = tid; d < (int)head_dim; d += bs) {
        qf32[d] = qh[d];
        qrot[d] = qroth[d];
    }
    // Init accumulators.
    for (int d = tid; d < (int)head_dim; d += bs) {
        acc_w[d] = 0.0f;
        acc_r[d] = 0.0f;
    }
    __syncthreads();

    const unsigned int groups_per_head = head_dim / ROT_GROUP;
    const size_t cold_row_bytes = (size_t)n_kv_heads * groups_per_head * BYTES_PER_GROUP;
    const size_t warm_row_elem  = (size_t)n_kv_heads * head_dim;

    // === Score phase ===
    for (int i = 0; i < slice_len; i++) {
        const int t = slice_start + i;
        float score;

        if (t < (int)cold_count) {
            // Cold: dequant all K groups (centroid lookup only),
            // dot with Q_rot. No FWHT.
            const unsigned char* k_row = cold_k
                + (size_t)t * cold_row_bytes
                + (size_t)kv_h * groups_per_head * BYTES_PER_GROUP;
            float local = 0.0f;
            for (unsigned int g = 0; g < groups_per_head; g++) {
                dequant_group_centroid(k_row + g * BYTES_PER_GROUP, dq_grp, tid);
                const int g_lo = g * ROT_GROUP;
                for (int d = tid; d < (int)ROT_GROUP; d += bs) {
                    local += qrot[g_lo + d] * dq_grp[d];
                }
                __syncthreads();
            }
            tmp[tid] = local;
            __syncthreads();
            for (int r = bs >> 1; r > 0; r >>= 1) {
                if (tid < r) tmp[tid] += tmp[tid + r];
                __syncthreads();
            }
            score = tmp[0] * scaling;
        } else {
            // Warm: int8 × fp32 dot with original Q.
            const int wt = t - (int)cold_count;
            const signed char* k_row = warm_k + (size_t)wt * warm_row_elem
                                              + (size_t)kv_h * head_dim;
            const float dk = warm_ks[(size_t)wt * n_kv_heads + kv_h];
            float local = 0.0f;
            for (int d = tid; d < (int)head_dim; d += bs)
                local += qf32[d] * (float)k_row[d];
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

    // === V accumulation, per tier ===
    for (int i = 0; i < slice_len; i++) {
        const int t = slice_start + i;
        const float w = scores[i];
        if (t < (int)cold_count) {
            // Cold V: dequant groups one at a time, accumulate to acc_r.
            const unsigned char* v_row_base = cold_v
                + (size_t)t * cold_row_bytes
                + (size_t)kv_h * groups_per_head * BYTES_PER_GROUP;
            for (unsigned int g = 0; g < groups_per_head; g++) {
                dequant_group_centroid(v_row_base + g * BYTES_PER_GROUP, dq_grp, tid);
                const int g_lo = g * ROT_GROUP;
                const int g_hi = g_lo + ROT_GROUP;
                for (int d = tid; d < (int)head_dim; d += bs) {
                    if (d >= g_lo && d < g_hi) {
                        acc_r[d] += w * dq_grp[d - g_lo];
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
                acc_w[d] += w * ((float)v_row[d] * dv);
            }
            __syncthreads();
        }
    }

    // === Un-rotate acc_r (one iRHT per group), then merge with acc_w ===
    for (unsigned int g = 0; g < groups_per_head; g++) {
        irht_group_inplace(acc_r + g * ROT_GROUP, signs1_v, signs2_v, tid, fwhtw);
    }

    for (int d = tid; d < (int)head_dim; d += bs) {
        o_partial[((size_t)h * n_splits + sp) * head_dim + d] = acc_w[d] + acc_r[d];
    }
    if (tid == 0) {
        m_partial[(size_t)h * n_splits + sp] = mx;
        l_partial[(size_t)h * n_splits + sp] = l;
    }
}

