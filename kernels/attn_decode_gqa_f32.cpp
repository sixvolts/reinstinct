// GQA flash-decoding attention over an f32 KV cache (Qwen full-attention
// decode). Replaces attn_partial_f32 for the per-token step; the merge
// kernel (attn_merge.cpp) is shared and unchanged.
//
// Why: attn_partial_f32 gave one workgroup to one *query* head, walked
// K rows with one scalar load per dimension (uncoalesced across the
// wave) and P·V with one dependent load per token, and read the shared
// KV head once per query head. At 8K context it ran at 74 GB/s: 0.9 ms
// per block, 14 ms per token on the 27B (24 q heads over 4 kv heads of 256).
//
// Here one workgroup owns (kv head, group of GH query heads, split):
//   * a K/V row is read once per group as float4s, LPT = HD/4 lanes per
//     token row, TPW = 64/LPT token rows per wave step, UNR steps of
//     loads in flight per lane;
//   * the GH query-head slices live in registers, scores are reduced
//     across the LPT lanes with DPP, and softmax runs online per TILE
//     tokens (running m / l / acc rescaled per tile), so any context
//     length fits in a fixed LDS footprint;
//   * the partial (m, l, o) per query head goes to the same buffers the
//     merge kernel already consumes.
//
// Compiled with `#define HD <head_dim>` and `#define GH <heads per group>`
// prepended by the launcher (HD in {64, 128, 256}; GH <= 8).
#include <hip/hip_runtime.h>
#include "gfx906_dpp.h"
#include "attn_gqa_common.h"

#ifndef HD
#define HD 128
#endif
#ifndef GH
#define GH 6
#endif
#define BS   256
#ifndef OCC
#define OCC  2                      // min workgroups per CU the compiler budgets registers for
#endif
#define NW   (BS / 64)
#define LPT  (HD / 4)               // lanes per token row
#define TPW  (64 / LPT)             // token rows per wave step
#define TPB  (NW * TPW)             // token rows per block step
#define TILE 256                    // tokens per online-softmax tile (== BS)
#ifndef UNR
#define UNR  4                      // block steps unrolled: loads in flight per lane (4: 106 VGPRs; 8 spills)
#endif

static_assert(HD % 4 == 0 && LPT >= 1 && LPT <= 64 && (64 % LPT) == 0, "head_dim must be 4..256, 64 % (HD/4) == 0");
static_assert(TILE == BS, "the exp pass maps one thread to one tile token");
static_assert(TILE % (TPB * UNR) == 0, "tile must be a whole number of unrolled block steps");
static_assert(GH >= 1 && GH <= 8, "GH <= 8");

// Segment reductions: attn_gqa_common.h (all-DPP, row_bcast for the
// cross-row steps; result valid in the segment's last lane).
// Combine across the TPW token slots of a wave (lane bits above LPT).
__device__ __forceinline__ float slot_sum(float x) {
    for (int o = LPT; o < 64; o <<= 1) x += __shfl_xor(x, o);
    return x;
}
__device__ __forceinline__ float slot_max(float x) {
    for (int o = LPT; o < 64; o <<= 1) x = fmaxf(x, __shfl_xor(x, o));
    return x;
}

extern "C" __global__ __launch_bounds__(BS, OCC)
void attn_decode_gqa_f32(const float* __restrict__ q,          // [n_heads, HD]
                         const float* __restrict__ k_cache,    // [max_seq, n_kv, HD]
                         const float* __restrict__ v_cache,
                         float*       __restrict__ o_partial,  // [n_heads, n_splits, HD]
                         float*       __restrict__ m_partial,  // [n_heads, n_splits]
                         float*       __restrict__ l_partial,  // [n_heads, n_splits]
                         unsigned int n_heads,
                         unsigned int n_kv_heads,
                         const unsigned int* __restrict__ pos_ptr,
                         float        scaling,
                         unsigned int n_splits)
{
    __shared__ float s_p[GH][TILE];        // tile probabilities
    __shared__ float s_red[NW][GH];        // per-wave max / sum exchange
    __shared__ float s_x[GH][HD];          // cross-wave combine exchange (end)

    const int G     = (int)(n_heads / n_kv_heads);
    const int kvh   = blockIdx.x % n_kv_heads;
    const int grp   = blockIdx.x / n_kv_heads;
    const int h0    = kvh * G + grp * GH;
    const int gh_n  = min(GH, G - grp * GH);          // live heads in this group
    const int sp    = blockIdx.y;
    const int tid   = threadIdx.x;
    const int wave  = tid >> 6;
    const int lane  = tid & 63;
    const int tl    = lane / LPT;                     // token slot in the wave step
    const int dl    = lane % LPT;                     // dims dl*4 .. dl*4+3
    const bool red_lane = (dl == LPT - 1);            // holds the segment sums
    const int total_len = (int)(*pos_ptr) + 1;
    const int chunk = (total_len + (int)n_splits - 1) / (int)n_splits;
    const int start = sp * chunk;
    const int end   = min(start + chunk, total_len);
    const size_t kv_row = (size_t)n_kv_heads * HD;

    if (end <= start) {
        for (int idx = tid; idx < GH * HD; idx += BS) {
            const int g = idx / HD, d = idx % HD;
            if (g < gh_n) o_partial[((size_t)(h0 + g) * n_splits + sp) * HD + d] = 0.0f;
        }
        if (tid < gh_n) {
            m_partial[(size_t)(h0 + tid) * n_splits + sp] = -INFINITY;
            l_partial[(size_t)(h0 + tid) * n_splits + sp] = 0.0f;
        }
        return;
    }

    // Query slices, pre-scaled, in registers.
    float4 qr[GH];
    #pragma unroll
    for (int g = 0; g < GH; g++) {
        if (g < gh_n) {
            float4 t = *reinterpret_cast<const float4*>(q + (size_t)(h0 + g) * HD + dl * 4);
            qr[g] = make_float4(t.x * scaling, t.y * scaling, t.z * scaling, t.w * scaling);
        } else {
            qr[g] = make_float4(0.f, 0.f, 0.f, 0.f);
        }
    }
    float m[GH], l[GH]; float4 acc[GH];
    #pragma unroll
    for (int g = 0; g < GH; g++) { m[g] = -INFINITY; l[g] = 0.0f; acc[g] = make_float4(0.f, 0.f, 0.f, 0.f); }

    const float* kbase = k_cache + (size_t)kvh * HD + dl * 4;
    const float* vbase = v_cache + (size_t)kvh * HD + dl * 4;

    for (int t0 = start; t0 < end; t0 += TILE) {
        const int tn = min(TILE, end - t0);

        // ---- scores: s_p[g][i] = q_g . K[t0 + i], tile max per head ----
        float tmax[GH];
        #pragma unroll
        for (int g = 0; g < GH; g++) tmax[g] = -INFINITY;
        const int n_st = (tn + TPB * UNR - 1) / (TPB * UNR) * UNR;   // steps this tile needs
        for (int st = 0; st < n_st; st += UNR) {
            float4 kk[UNR];
            #pragma unroll
            for (int u = 0; u < UNR; u++) {
                const int i = (st + u) * TPB + wave * TPW + tl;
                const int t = t0 + min(i, tn - 1);            // clamped: always a valid row
                kk[u] = *reinterpret_cast<const float4*>(kbase + (size_t)t * kv_row);
            }
            #pragma unroll
            for (int u = 0; u < UNR; u++) {
                const int i = (st + u) * TPB + wave * TPW + tl;
                const bool live = red_lane && (i < tn);
                float x[GH];
                #pragma unroll
                for (int g = 0; g < GH; g++)
                    x[g] = qr[g].x * kk[u].x + qr[g].y * kk[u].y + qr[g].z * kk[u].z + qr[g].w * kk[u].w;
                seg_sum_multi<GH, LPT>(x);
                #pragma unroll
                for (int g = 0; g < GH; g++) {
                    if (live) {
                        s_p[g][i] = x[g];
                        tmax[g] = fmaxf(tmax[g], x[g]);
                    }
                }
            }
        }
        #pragma unroll
        for (int g = 0; g < GH; g++) {
            const float v = slot_max(tmax[g]);          // valid where red_lane
            if (lane == LPT - 1) s_red[wave][g] = v;
        }
        __syncthreads();
        // ---- online rescale to the new running max ----
        #pragma unroll
        for (int g = 0; g < GH; g++) {
            float tm = s_red[0][g];
            #pragma unroll
            for (int w = 1; w < NW; w++) tm = fmaxf(tm, s_red[w][g]);
            // Wave-uniform: keep m / l in SGPRs (the compiler cannot see
            // that an LDS read is uniform), which is worth ~20 VGPRs here.
            const float mn = __int_as_float(__builtin_amdgcn_readfirstlane(__float_as_int(fmaxf(m[g], tm))));
            const float alpha = __expf(m[g] - mn);          // 0 when m was -inf
            m[g] = mn;
            l[g] *= alpha;
            acc[g].x *= alpha; acc[g].y *= alpha; acc[g].z *= alpha; acc[g].w *= alpha;
        }
        __syncthreads();                                   // s_red reused below
        // ---- exp in place, tile sum per head ----
        float psum[GH];
        #pragma unroll
        for (int g = 0; g < GH; g++) {
            float e = 0.0f;
            if (tid < tn) { e = __expf(s_p[g][tid] - m[g]); s_p[g][tid] = e; }
            psum[g] = e;
        }
        #pragma unroll
        for (int g = 0; g < GH; g++) {
            const float v = wave64_reduce_add_f32(psum[g]);
            if (lane == 0) s_red[wave][g] = v;
        }
        __syncthreads();
        #pragma unroll
        for (int g = 0; g < GH; g++) {
            float ts = 0.0f;
            #pragma unroll
            for (int w = 0; w < NW; w++) ts += s_red[w][g];
            l[g] = __int_as_float(__builtin_amdgcn_readfirstlane(__float_as_int(l[g] + ts)));
        }
        // ---- P·V over the tile ----
        for (int st = 0; st < n_st; st += UNR) {
            float4 vv[UNR];
            #pragma unroll
            for (int u = 0; u < UNR; u++) {
                const int i = (st + u) * TPB + wave * TPW + tl;
                const int t = t0 + min(i, tn - 1);
                vv[u] = *reinterpret_cast<const float4*>(vbase + (size_t)t * kv_row);
            }
            #pragma unroll
            for (int u = 0; u < UNR; u++) {
                const int i = (st + u) * TPB + wave * TPW + tl;
                const bool live = i < tn;
                #pragma unroll
                for (int g = 0; g < GH; g++) {
                    const float p = live ? s_p[g][i] : 0.0f;
                    acc[g].x += p * vv[u].x; acc[g].y += p * vv[u].y;
                    acc[g].z += p * vv[u].z; acc[g].w += p * vv[u].w;
                }
            }
        }
        __syncthreads();                                   // s_p / s_red reuse next tile
    }

    // ---- combine token slots, then waves (one exchange buffer, wave 0
    // accumulates), write the partials ----
    #pragma unroll
    for (int g = 0; g < GH; g++) {
        acc[g].x = slot_sum(acc[g].x); acc[g].y = slot_sum(acc[g].y);
        acc[g].z = slot_sum(acc[g].z); acc[g].w = slot_sum(acc[g].w);
    }
    for (int w = 1; w < NW; w++) {
        if (wave == w && tl == 0) {
            #pragma unroll
            for (int g = 0; g < GH; g++) *reinterpret_cast<float4*>(&s_x[g][dl * 4]) = acc[g];
        }
        __syncthreads();
        if (wave == 0 && tl == 0) {
            #pragma unroll
            for (int g = 0; g < GH; g++) {
                const float4 t = *reinterpret_cast<const float4*>(&s_x[g][dl * 4]);
                acc[g].x += t.x; acc[g].y += t.y; acc[g].z += t.z; acc[g].w += t.w;
            }
        }
        __syncthreads();
    }
    if (wave == 0 && tl == 0) {
        #pragma unroll
        for (int g = 0; g < GH; g++)
            if (g < gh_n)
                *reinterpret_cast<float4*>(o_partial + ((size_t)(h0 + g) * n_splits + sp) * HD + dl * 4) = acc[g];
    }
    if (tid == 0) {
        #pragma unroll
        for (int g = 0; g < GH; g++) {
            if (g < gh_n) {
                m_partial[(size_t)(h0 + g) * n_splits + sp] = m[g];
                l_partial[(size_t)(h0 + g) * n_splits + sp] = l[g];
            }
        }
    }
}
