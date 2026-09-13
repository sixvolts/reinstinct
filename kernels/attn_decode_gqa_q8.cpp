// GQA flash-decoding attention over the int8 KV cache (Gemma decode):
// the attn_decode_gqa_f32.cpp design on attn_partial_q8's cache format
// — int8 K/V rows with one f32 scale per (token, kv head), Q quantised
// per head to int8 inside the kernel, dp4a scores, optional sliding
// window. Replaces attn_partial_q8 for the per-token step; the merge
// kernel (attn_merge.cpp) is shared and unchanged.
//
// One workgroup per (kv head, group of GH query heads, split). Lanes
// hold 8 int8 of a K/V row (LPT = HD/8 lanes per row, TPW = 64/LPT rows
// per wave step, UNR steps of loads in flight); the GH quantised query
// slices live in registers as 2 x u32 each; scores are 2 sdot4 per head
// per row, reduced across the row's lanes with DPP; softmax runs online
// per TILE tokens; partial (m, l, o) into the buffers the merge reads.
//
// Compiled with `#define HD <head_dim>` (a multiple of 64 up to 512) and
// `#define GH <query heads per group>` prepended by the launcher.
#include <hip/hip_runtime.h>
#include "gfx906_dpp.h"
#include "attn_gqa_common.h"

#ifndef HD
#define HD 256
#endif
#ifndef GH
#define GH 2
#endif
#define BS   256
#ifndef OCC
#define OCC  2
#endif
#define NW   (BS / 64)
#ifndef EPL
#define EPL  8                      // int8 elements per lane (8: uint2 loads, 16: uint4)
#endif
#define WPL  (EPL / 4)              // u32 words per lane
#define LPT  (HD / EPL)             // lanes per token row
#define TPW  (64 / LPT)             // token rows per wave step
#define TPB  (NW * TPW)
#define TILE 256
#ifndef UNR
#define UNR  4
#endif

static_assert(EPL == 8 || EPL == 16, "EPL is 8 or 16");
static_assert(HD % 64 == 0 && LPT >= 4 && LPT <= 64, "head_dim must be 64..512, a multiple of 64");
static_assert(TILE == BS, "the exp pass maps one thread to one tile token");
static_assert(TILE % (TPB * UNR) == 0, "tile must be a whole number of unrolled block steps");
static_assert(GH >= 1 && GH <= 8, "GH <= 8");

__device__ __forceinline__ float slot_sum(float x) {
    for (int o = LPT; o < 64; o <<= 1) x += __shfl_xor(x, o);
    return x;
}
__device__ __forceinline__ float slot_max(float x) {
    for (int o = LPT; o < 64; o <<= 1) x = fmaxf(x, __shfl_xor(x, o));
    return x;
}
__device__ __forceinline__ float uniform_f32(float v, int lane) {
    return __int_as_float(__builtin_amdgcn_readlane(__float_as_int(v), lane));
}
__device__ __forceinline__ unsigned int pack4(float a, float b, float c, float d, float inv) {
    int x0 = max(-127, min(127, (int)rintf(a * inv)));
    int x1 = max(-127, min(127, (int)rintf(b * inv)));
    int x2 = max(-127, min(127, (int)rintf(c * inv)));
    int x3 = max(-127, min(127, (int)rintf(d * inv)));
    return (unsigned)(x0 & 0xff) | ((unsigned)(x1 & 0xff) << 8) | ((unsigned)(x2 & 0xff) << 16) | ((unsigned)(x3 & 0xff) << 24);
}
__device__ __forceinline__ void unpack4(unsigned int w, float* f) {
    f[0] = (float)(int)(signed char)(w & 0xff);
    f[1] = (float)(int)(signed char)((w >> 8) & 0xff);
    f[2] = (float)(int)(signed char)((w >> 16) & 0xff);
    f[3] = (float)(int)(signed char)(w >> 24);
}
struct Row { unsigned int w[WPL]; };
__device__ __forceinline__ Row load_row(const signed char* p) {
    Row r;
    if (WPL == 2) { const uint2 t = *reinterpret_cast<const uint2*>(p); r.w[0] = t.x; r.w[1] = t.y; }
    else { const uint4 t = *reinterpret_cast<const uint4*>(p); r.w[0] = t.x; r.w[1] = t.y; r.w[2] = t.z; r.w[3] = t.w; }
    return r;
}

extern "C" __global__ __launch_bounds__(BS, OCC)
void attn_decode_gqa_q8_f32(const float*       __restrict__ q,          // [n_heads, HD]
                            const signed char* __restrict__ k_cache,    // [max_seq, n_kv, HD]
                            const float*       __restrict__ k_scale,    // [max_seq, n_kv]
                            const signed char* __restrict__ v_cache,
                            const float*       __restrict__ v_scale,
                            float*             __restrict__ o_partial,  // [n_heads, n_splits, HD]
                            float*             __restrict__ m_partial,  // [n_heads, n_splits]
                            float*             __restrict__ l_partial,
                            unsigned int n_heads,
                            unsigned int n_kv_heads,
                            const unsigned int* __restrict__ pos_ptr,
                            unsigned int window,
                            float        scaling,
                            unsigned int n_splits)
{
    __shared__ float s_p[GH][TILE];
    __shared__ float s_red[NW][GH];
    __shared__ float s_x[GH][HD];

    const int G     = (int)(n_heads / n_kv_heads);
    const int kvh   = blockIdx.x % n_kv_heads;
    const int grp   = blockIdx.x / n_kv_heads;
    const int h0    = kvh * G + grp * GH;
    const int gh_n  = min(GH, G - grp * GH);
    const int sp    = blockIdx.y;
    const int tid   = threadIdx.x;
    const int wave  = tid >> 6;
    const int lane  = tid & 63;
    const int tl    = lane / LPT;
    const int dl    = lane % LPT;                     // elements dl*EPL .. dl*EPL+EPL-1
    const bool red_lane = (dl == LPT - 1);
    const int total_len = (int)(*pos_ptr) + 1;
    const int lo = (window > 0 && total_len > (int)window) ? total_len - (int)window : 0;
    const int win_len = total_len - lo;
    const int chunk = (win_len + (int)n_splits - 1) / (int)n_splits;
    const int start = lo + sp * chunk;
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

    // Quantise the GH query slices to int8 with a per-head scale (the
    // per-head amax reduced across the row's lanes; every token slot
    // computes the same, so the result is wave-uniform).
    unsigned int qi[GH][WPL];
    float dq[GH];
    {
        float amax[GH];
        float qf[GH][EPL];
        #pragma unroll
        for (int g = 0; g < GH; g++) {
            float a = 0.0f;
            #pragma unroll
            for (int c = 0; c < WPL; c++) {
                float4 t = make_float4(0.f, 0.f, 0.f, 0.f);
                if (g < gh_n) t = *reinterpret_cast<const float4*>(q + (size_t)(h0 + g) * HD + dl * EPL + c * 4);
                qf[g][c * 4 + 0] = t.x; qf[g][c * 4 + 1] = t.y; qf[g][c * 4 + 2] = t.z; qf[g][c * 4 + 3] = t.w;
            }
            #pragma unroll
            for (int j = 0; j < EPL; j++) a = fmaxf(a, fabsf(qf[g][j]));
            amax[g] = a;
        }
        seg_max_multi<GH, LPT>(amax);
        #pragma unroll
        for (int g = 0; g < GH; g++) {
            const float am = uniform_f32(amax[g], LPT - 1);
            dq[g] = am > 0.0f ? am / 127.0f * scaling : 0.0f;
            const float inv = am > 0.0f ? 127.0f / am : 0.0f;
            #pragma unroll
            for (int c = 0; c < WPL; c++)
                qi[g][c] = pack4(qf[g][c * 4], qf[g][c * 4 + 1], qf[g][c * 4 + 2], qf[g][c * 4 + 3], inv);
        }
    }

    float m[GH], l[GH], acc[GH][EPL];
    #pragma unroll
    for (int g = 0; g < GH; g++) {
        m[g] = -INFINITY; l[g] = 0.0f;
        #pragma unroll
        for (int j = 0; j < EPL; j++) acc[g][j] = 0.0f;
    }
    const signed char* kbase = k_cache + (size_t)kvh * HD + dl * EPL;
    const signed char* vbase = v_cache + (size_t)kvh * HD + dl * EPL;
    const float* ks = k_scale + kvh;
    const float* vs = v_scale + kvh;

    for (int t0 = start; t0 < end; t0 += TILE) {
        const int tn = min(TILE, end - t0);
        float tmax[GH];
        #pragma unroll
        for (int g = 0; g < GH; g++) tmax[g] = -INFINITY;
        const int n_st = (tn + TPB * UNR - 1) / (TPB * UNR) * UNR;
        for (int st = 0; st < n_st; st += UNR) {
            Row kk[UNR]; float dk[UNR];
            #pragma unroll
            for (int u = 0; u < UNR; u++) {
                const int i = (st + u) * TPB + wave * TPW + tl;
                const int t = t0 + min(i, tn - 1);
                kk[u] = load_row(kbase + (size_t)t * kv_row);
                dk[u] = ks[(size_t)t * n_kv_heads];
            }
            #pragma unroll
            for (int u = 0; u < UNR; u++) {
                const int i = (st + u) * TPB + wave * TPW + tl;
                const bool live = red_lane && (i < tn);
                float x[GH];
                #pragma unroll
                for (int g = 0; g < GH; g++) {
                    int idot = 0;
                    #pragma unroll
                    for (int c = 0; c < WPL; c++)
                        idot = __builtin_amdgcn_sdot4((int)qi[g][c], (int)kk[u].w[c], idot, false);
                    x[g] = (float)idot * dq[g] * dk[u];
                }
                seg_sum_multi<GH, LPT>(x);
                #pragma unroll
                for (int g = 0; g < GH; g++) {
                    if (live) { s_p[g][i] = x[g]; tmax[g] = fmaxf(tmax[g], x[g]); }
                }
            }
        }
        #pragma unroll
        for (int g = 0; g < GH; g++) {
            const float v = slot_max(tmax[g]);
            if (lane == LPT - 1) s_red[wave][g] = v;
        }
        __syncthreads();
        #pragma unroll
        for (int g = 0; g < GH; g++) {
            float tm = s_red[0][g];
            #pragma unroll
            for (int w = 1; w < NW; w++) tm = fmaxf(tm, s_red[w][g]);
            const float mn = uniform_f32(fmaxf(m[g], tm), 0);
            const float alpha = __expf(m[g] - mn);
            m[g] = mn;
            l[g] *= alpha;
            #pragma unroll
            for (int j = 0; j < EPL; j++) acc[g][j] *= alpha;
        }
        __syncthreads();
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
            l[g] = uniform_f32(l[g] + ts, 0);
        }
        // P.V with the per-token V scale folded into p.
        for (int st = 0; st < n_st; st += UNR) {
            Row vv[UNR]; float dv[UNR];
            #pragma unroll
            for (int u = 0; u < UNR; u++) {
                const int i = (st + u) * TPB + wave * TPW + tl;
                const int t = t0 + min(i, tn - 1);
                vv[u] = load_row(vbase + (size_t)t * kv_row);
                dv[u] = vs[(size_t)t * n_kv_heads];
            }
            #pragma unroll
            for (int u = 0; u < UNR; u++) {
                const int i = (st + u) * TPB + wave * TPW + tl;
                const bool live = i < tn;
                float vf[EPL];
                #pragma unroll
                for (int c = 0; c < WPL; c++) unpack4(vv[u].w[c], vf + c * 4);
                #pragma unroll
                for (int g = 0; g < GH; g++) {
                    const float p = live ? s_p[g][i] * dv[u] : 0.0f;
                    #pragma unroll
                    for (int j = 0; j < EPL; j++) acc[g][j] += p * vf[j];
                }
            }
        }
        __syncthreads();
    }

    // Combine token slots, then waves (one exchange buffer, wave 0
    // accumulates), write the partials.
    #pragma unroll
    for (int g = 0; g < GH; g++) {
        #pragma unroll
        for (int j = 0; j < EPL; j++) acc[g][j] = slot_sum(acc[g][j]);
    }
    for (int w = 1; w < NW; w++) {
        if (wave == w && tl == 0) {
            #pragma unroll
            for (int g = 0; g < GH; g++) {
                #pragma unroll
                for (int j = 0; j < EPL; j++) s_x[g][dl * EPL + j] = acc[g][j];
            }
        }
        __syncthreads();
        if (wave == 0 && tl == 0) {
            #pragma unroll
            for (int g = 0; g < GH; g++) {
                #pragma unroll
                for (int j = 0; j < EPL; j++) acc[g][j] += s_x[g][dl * EPL + j];
            }
        }
        __syncthreads();
    }
    if (wave == 0 && tl == 0) {
        #pragma unroll
        for (int g = 0; g < GH; g++) {
            if (g < gh_n) {
                float* o = o_partial + ((size_t)(h0 + g) * n_splits + sp) * HD + dl * EPL;
                #pragma unroll
                for (int j = 0; j < EPL; j++) o[j] = acc[g][j];
            }
        }
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
