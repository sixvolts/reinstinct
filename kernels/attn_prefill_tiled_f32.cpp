// Tiled causal attention for prefill over the f32 KV cache (Qwen
// full-attention layers). Replaces attn_prefill_flash_f32 for the
// causal path; same argument list.
//
// attn_prefill_flash gave one wavefront to one query and, per key, did
// 4 FMAs per lane and a 64-lane shuffle reduction — plus an online
// softmax step per key — so it ran at 0.7 TFLOPS: 274 ms per layer at
// 3971 tokens on the 27B, 4.4 s of a 22 s prefill.
//
// Here one workgroup owns BQ = 32 queries and streams the keys in
// BK = 16 tiles, each staged once into LDS as fp16 and reused by every
// query; the 256 threads block the work so that LDS traffic stays well
// under the FMA rate:
//   * scores: thread (qb, kb, dq) = (query pair, key within a KPP-key
//     pass, 1/NDQ slice of head_dim; NDQ = HD/32 so a slice is always 32
//     dims = 16 half2 of Q per query, register-resident — Q is read once
//     per workgroup). Per key it reads only its K slice, as half2, and
//     accumulates with v_dot2_f32_f16 (two MACs per instruction, f32
//     accumulate); the NDQ slices reduce over lanes with DPP. Slices are
//     dim-interleaved (dims 2*NDQ*j + 2dq, +1) so the 16 distinct K
//     addresses of a wave hit 16 banks.
//   * softmax: online per tile — running m / l per query live in the
//     16 lanes of the query pair's row; p goes to LDS as fp16.
//   * P.V: thread (qb, db) owns dims db + 16j of its two queries; V is
//     staged key-pair-interleaved (Vt[dim][key], half2 = two keys) so
//     the accumulation is v_dot2 as well: two keys per instruction.
//
// 3.5 TFLOPS on the 27B (0.7 before): 273 -> 56 ms per layer at 3968
// rows, fp16-tile rounding of 2e-4 relative. `window` > 0 limits each
// query to the last `window` positions (Gemma's sliding layers).
// Compiled with `#define HD <head_dim>` prepended (128, 256 or 512).
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "attn_gqa_common.h"

#ifndef HD
#define HD 256
#endif
#define BQ   32
#define BK   16
#define BS   256
#define NQB  (BQ / 2)          // query pairs
#define NDQ  (HD / 32)         // head_dim slices of 32 dims
#define KPP  (16 / NDQ)        // keys per pass
#define NPASS (BK / KPP)
#define KROW (HD / 2 + 8)      // Ks row stride in half2 (padded: 16 banks)
#define VROW (BK + 2)          // Vt row stride in half (9 dwords: 16 banks)
#define PROW (BK + 2)          // Ps row stride in half
#define QJ   16                // half2 per thread per query (its 32-dim slice)
#define DPT  (HD / 16)         // dims per thread in P.V

static_assert(HD == 128 || HD == 256 || HD == 512, "head_dim must be 128, 256 or 512");
static_assert(NQB * KPP * NDQ == BS, "thread mapping");
static_assert(NDQ * QJ * 2 == HD, "slice covers head_dim");

typedef _Float16 f16x2_t __attribute__((ext_vector_type(2)));
__device__ __forceinline__ float fdot2(__half2 a, __half2 b, float c) {
    return __builtin_amdgcn_fdot2(*reinterpret_cast<f16x2_t*>(&a),
                                  *reinterpret_cast<f16x2_t*>(&b), c, false);
}
// Row-wide (16-lane) reductions: the NDQ slice lanes of a key hold
// identical values after seg_sum, so a row max is the tile max over the
// KPP keys, and a row sum is NDQ x the key sum (exact: power of two).
__device__ __forceinline__ float row16_max(float v) { float a[1] = { v }; seg_max_multi<1, 16>(a); return a[0]; }
__device__ __forceinline__ float row16_sum(float v) { float a[1] = { v }; seg_sum_multi<1, 16>(a); return a[0]; }

#ifndef OCC
#define OCC 1                  // register budget: one workgroup per CU measures 25% faster than two (which spills)
#endif
extern "C" __global__ __launch_bounds__(BS, OCC)
void attn_prefill_tiled_f32(const float* __restrict__ q,       // [n_rows, n_heads, HD]
                            const float* __restrict__ k,       // [kv_len, n_kv_heads, HD]
                            const float* __restrict__ v,
                            float*       __restrict__ out,     // [n_rows, n_heads, HD]
                            unsigned int n_heads,
                            unsigned int n_kv_heads,
                            unsigned int head_dim,             // == HD (ABI parity)
                            unsigned int window,               // 0 = full causal; else sliding
                            float        scaling,
                            unsigned int n_rows,
                            unsigned int base_pos)
{
    (void)head_dim;
    __shared__ __half2 Ks[BK][KROW];
    __shared__ __half  Vt[HD][VROW];
    __shared__ __half  Ps[BQ][PROW];

    const int tid  = threadIdx.x;
    const int qb   = tid >> 4;             // query pair 0..15
    const int kb   = (tid & 15) / NDQ;     // key within pass
    const int dq   = tid % NDQ;            // head_dim slice
    const int db   = tid & 15;             // P.V: dim slot
    const unsigned int h     = blockIdx.x;
    const unsigned int qbase = blockIdx.y * BQ;
    const int groups = (int)(n_heads / n_kv_heads);
    const unsigned int kv_h = h / groups;
    const size_t kv_row = (size_t)n_kv_heads * HD;
    const int kv_len = (int)(base_pos + n_rows);

    // Query slices (dims 2*NDQ*j + 2dq, +1), pre-scaled, as half2.
    __half2 qh[2][QJ];
    int q_pos[2], q_lo[2];
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        const unsigned int q_row = qbase + 2 * qb + i;
        const bool act = q_row < n_rows;
        q_pos[i] = act ? (int)(base_pos + q_row) : -1;
        q_lo[i]  = (window > 0 && q_pos[i] + 1 > (int)window) ? q_pos[i] + 1 - (int)window : 0;
        const float* qp = q + ((size_t)(act ? q_row : 0) * n_heads + h) * HD + 2 * dq;
        #pragma unroll
        for (int j = 0; j < QJ; j++) {
            const float2 t = *reinterpret_cast<const float2*>(qp + 2 * NDQ * j);
            qh[i][j] = __floats2half2_rn(act ? t.x * scaling : 0.0f, act ? t.y * scaling : 0.0f);
        }
    }
    float m[2] = { -INFINITY, -INFINITY }, l[2] = { 0.0f, 0.0f };
    float o[2][DPT];
    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < DPT; j++) o[i][j] = 0.0f;

    // Causal: keys up to the workgroup's last query position; with a
    // window, from the workgroup's first query's lower bound.
    int kt_end = (int)base_pos + (int)qbase + BQ;
    if (kt_end > kv_len) kt_end = kv_len;
    const int q0 = (int)base_pos + (int)qbase;
    const int lo0 = (window > 0 && q0 + 1 > (int)window) ? q0 + 1 - (int)window : 0;
    const int kt_start = (lo0 / BK) * BK;

    for (int kt0 = kt_start; kt0 < kt_end; kt0 += BK) {
        // ---- stage the K / V tile as fp16 (K row-major padded, V key-interleaved) ----
        for (int e = tid; e < BK * HD / 4; e += BS) {
            const int key = e / (HD / 4), d = (e % (HD / 4)) * 4;
            const int kpos = kt0 + key;
            float4 kk = make_float4(0.f, 0.f, 0.f, 0.f), vv = kk;
            if (kpos < kv_len) {
                const size_t off = (size_t)kpos * kv_row + (size_t)kv_h * HD + d;
                kk = *reinterpret_cast<const float4*>(k + off);
                vv = *reinterpret_cast<const float4*>(v + off);
            }
            Ks[key][d / 2]     = __floats2half2_rn(kk.x, kk.y);
            Ks[key][d / 2 + 1] = __floats2half2_rn(kk.z, kk.w);
            Vt[d][key]     = __float2half_rn(vv.x);
            Vt[d + 1][key] = __float2half_rn(vv.y);
            Vt[d + 2][key] = __float2half_rn(vv.z);
            Vt[d + 3][key] = __float2half_rn(vv.w);
        }
        __syncthreads();

        // ---- scores for this tile: s[i][pass] = q_i . K[pass*KPP + kb] ----
        float s[2][NPASS];
        #pragma unroll
        for (int p = 0; p < NPASS; p++) {
            const int key = p * KPP + kb;
            float a[2] = { 0.0f, 0.0f };
            const __half2* kr = &Ks[key][dq];
            #pragma unroll
            for (int j = 0; j < QJ; j++) {
                const __half2 kh = kr[NDQ * j];
                a[0] = fdot2(qh[0][j], kh, a[0]);
                a[1] = fdot2(qh[1][j], kh, a[1]);
            }
            seg_sum_multi<2, NDQ>(a);          // over the NDQ slices (every lane gets the sum)
            const int kpos = kt0 + key;
            s[0][p] = (kpos <= q_pos[0] && kpos >= q_lo[0]) ? a[0] : -INFINITY;
            s[1][p] = (kpos <= q_pos[1] && kpos >= q_lo[1]) ? a[1] : -INFINITY;
            // Keep the scheduler from hoisting the next pass's 16 K reads
            // over this one (it did, and spilled 440 B per lane).
            __builtin_amdgcn_sched_barrier(0);
        }
        // ---- online softmax over the tile (per query: passes, then the row's key lanes) ----
        float corr[2];
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            float tm = s[i][0];
            #pragma unroll
            for (int p = 1; p < NPASS; p++) tm = fmaxf(tm, s[i][p]);
            tm = row16_max(tm);
            const float mn = fmaxf(m[i], tm);
            // A tile with no live key for this query leaves m unchanged.
            corr[i] = (m[i] == -INFINITY || mn == -INFINITY) ? ((mn == -INFINITY) ? 1.0f : 0.0f) : __expf(m[i] - mn);
            float ps = 0.0f;
            #pragma unroll
            for (int p = 0; p < NPASS; p++) {
                const float e = (s[i][p] == -INFINITY) ? 0.0f : __expf(s[i][p] - mn);
                ps += e;
                if (dq == 0) Ps[2 * qb + i][p * KPP + kb] = __float2half_rn(e);
            }
            ps = row16_sum(ps) * (1.0f / NDQ);
            l[i] = l[i] * corr[i] + ps;
            m[i] = mn;
        }
        __syncthreads();

        // ---- P.V: o[i][j] (dim db + 16 j) += sum over key pairs, v_dot2 ----
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            const __half2* pr = reinterpret_cast<const __half2*>(&Ps[2 * qb + i][0]);
            __half2 pp[BK / 2];
            #pragma unroll
            for (int kk = 0; kk < BK / 2; kk++) pp[kk] = pr[kk];
            #pragma unroll
            for (int j = 0; j < DPT; j++) {
                const __half2* vr = reinterpret_cast<const __half2*>(&Vt[db + 16 * j][0]);
                float acc = o[i][j] * corr[i];
                #pragma unroll
                for (int kk = 0; kk < BK / 2; kk++) acc = fdot2(pp[kk], vr[kk], acc);
                o[i][j] = acc;
                __builtin_amdgcn_sched_barrier(0);
            }
        }
        __syncthreads();                       // Ks / Vt / Ps reused by the next tile
    }

    #pragma unroll
    for (int i = 0; i < 2; i++) {
        const unsigned int q_row = qbase + 2 * qb + i;
        if (q_row < n_rows && l[i] > 0.0f) {
            const float inv = 1.0f / l[i];
            float* op = out + ((size_t)q_row * n_heads + h) * HD;
            #pragma unroll
            for (int j = 0; j < DPT; j++) op[db + 16 * j] = o[i][j] * inv;
        }
    }
}
