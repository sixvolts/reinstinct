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
//   * scores: thread (qb, kb2, dq) = (query pair, key within a 2-key
//     pass, eighth of head_dim). Its two query slices are register-
//     resident (Q is read once per workgroup); per key it reads only
//     the K eighth, as half2, and accumulates with v_dot2_f32_f16
//     (two MACs per instruction, f32 accumulate); the 8 eighths reduce
//     over lanes with three DPP steps. Eighths are dim-interleaved
//     (dims 16j+2dq, +1) so the 16 distinct K addresses of a wave hit
//     16 banks. (Quarters left 64 half2 of Q per thread and spilled.)
//   * softmax: online per tile — running m / l per query live in the
//     16 lanes of the query pair's row; p goes to LDS as fp16.
//   * P.V: thread (qb, db) owns dims db + 16j of its two queries; V is
//     staged key-pair-interleaved (Vt[dim][key], half2 = two keys) so
//     the accumulation is v_dot2 as well: two keys per instruction.
//
// 3.5 TFLOPS on the 27B (0.7 before): 273 -> 56 ms per layer at 3968
// rows, fp16-tile rounding of 2e-4 relative. Compiled with `#define HD
// <head_dim>` prepended (a multiple of 64, <= 256).
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
#define KPP  2                 // keys per pass
#define NDQ  8                 // head_dim eighths
#define NPASS (BK / KPP)
#define KROW (HD / 2 + 8)      // Ks row stride in half2 (padded: 16 banks)
#define VROW (BK + 2)          // Vt row stride in half (9 dwords: 16 banks)
#define PROW (BK + 2)          // Ps row stride in half
#define QJ   (HD / 16)         // half2 per thread per query (its eighth)
#define DPT  (HD / 16)         // dims per thread in P.V

static_assert(HD % 64 == 0 && HD <= 256, "head_dim must be a multiple of 64, <= 256");
static_assert(NQB * KPP * NDQ == BS, "thread mapping");

typedef _Float16 f16x2_t __attribute__((ext_vector_type(2)));
__device__ __forceinline__ float fdot2(__half2 a, __half2 b, float c) {
    return __builtin_amdgcn_fdot2(*reinterpret_cast<f16x2_t*>(&a),
                                  *reinterpret_cast<f16x2_t*>(&b), c, false);
}
__device__ __forceinline__ float dpp_ror8(float v) { return dpp_f32<DPP_ROW_ROR8, 0xf, 0xf>(0.0f, v); }

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
                            unsigned int window,               // 0 = full causal (only 0 supported)
                            float        scaling,
                            unsigned int n_rows,
                            unsigned int base_pos)
{
    (void)head_dim; (void)window;
    __shared__ __half2 Ks[BK][KROW];
    __shared__ __half  Vt[HD][VROW];
    __shared__ __half  Ps[BQ][PROW];

    const int tid  = threadIdx.x;
    const int qb   = tid >> 4;             // query pair 0..15
    const int kb2  = (tid >> 3) & 1;       // key within pass
    const int dq   = tid & 7;              // head_dim eighth
    const int db   = tid & 15;             // P.V: dim slot
    const unsigned int h     = blockIdx.x;
    const unsigned int qbase = blockIdx.y * BQ;
    const int groups = (int)(n_heads / n_kv_heads);
    const unsigned int kv_h = h / groups;
    const size_t kv_row = (size_t)n_kv_heads * HD;
    const int kv_len = (int)(base_pos + n_rows);

    // Query slices (dims 16j + 2dq, +1), pre-scaled, as half2.
    __half2 qh[2][QJ];
    int q_pos[2];
    #pragma unroll
    for (int i = 0; i < 2; i++) {
        const unsigned int q_row = qbase + 2 * qb + i;
        const bool act = q_row < n_rows;
        q_pos[i] = act ? (int)(base_pos + q_row) : -1;
        const float* qp = q + ((size_t)(act ? q_row : 0) * n_heads + h) * HD + 2 * dq;
        #pragma unroll
        for (int j = 0; j < QJ; j++) {
            const float2 t = *reinterpret_cast<const float2*>(qp + 16 * j);
            qh[i][j] = __floats2half2_rn(act ? t.x * scaling : 0.0f, act ? t.y * scaling : 0.0f);
        }
    }
    float m[2] = { -INFINITY, -INFINITY }, l[2] = { 0.0f, 0.0f };
    float o[2][DPT];
    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < DPT; j++) o[i][j] = 0.0f;

    // Causal: keys up to the workgroup's last query position.
    int kt_end = (int)base_pos + (int)qbase + BQ;
    if (kt_end > kv_len) kt_end = kv_len;

    for (int kt0 = 0; kt0 < kt_end; kt0 += BK) {
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

        // ---- scores for this tile: s[i][pass] = q_i . K[pass*2 + kb2] ----
        float s[2][NPASS];
        #pragma unroll
        for (int p = 0; p < NPASS; p++) {
            const int key = p * KPP + kb2;
            float a[2] = { 0.0f, 0.0f };
            const __half2* kr = &Ks[key][dq];
            #pragma unroll
            for (int j = 0; j < QJ; j++) {
                const __half2 kh = kr[8 * j];
                a[0] = fdot2(qh[0][j], kh, a[0]);
                a[1] = fdot2(qh[1][j], kh, a[1]);
            }
            seg_sum_multi<2, NDQ>(a);          // over the 8 eighths (every lane gets the sum)
            const int kpos = kt0 + key;
            s[0][p] = (kpos <= q_pos[0]) ? a[0] : -INFINITY;
            s[1][p] = (kpos <= q_pos[1]) ? a[1] : -INFINITY;
            // Keep the scheduler from hoisting the next pass's 16 K reads
            // over this one (it did, and spilled 440 B per lane).
            __builtin_amdgcn_sched_barrier(0);
        }
        // ---- online softmax over the tile (per query: passes, then the two kb2 lanes) ----
        float corr[2];
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            float tm = s[i][0];
            #pragma unroll
            for (int p = 1; p < NPASS; p++) tm = fmaxf(tm, s[i][p]);
            tm = fmaxf(tm, dpp_ror8(tm));
            const float mn = fmaxf(m[i], tm);
            corr[i] = (m[i] == -INFINITY) ? 0.0f : __expf(m[i] - mn);
            float ps = 0.0f;
            #pragma unroll
            for (int p = 0; p < NPASS; p++) {
                const float e = (s[i][p] == -INFINITY) ? 0.0f : __expf(s[i][p] - mn);
                ps += e;
                if (dq == 0) Ps[2 * qb + i][p * KPP + kb2] = __float2half_rn(e);
            }
            ps += dpp_ror8(ps);
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
