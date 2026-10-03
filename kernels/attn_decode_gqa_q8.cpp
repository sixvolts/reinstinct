// GQA flash-decoding attention over the int8 KV cache (Gemma decode):
// the attn_decode_gqa_f32.cpp design on attn_partial_q8's cache format
// — int8 K/V rows with one f32 scale per (token, kv head), Q quantised
// per head to int8 inside the kernel, dp4a scores, optional sliding
// window. Replaces attn_partial_q8 for the per-token step; the merge
// kernel (attn_merge.cpp) is shared and unchanged.
//
// One workgroup per (kv head, group of GH query heads, split). The
// scores pass gives each K row KLPT lanes of KEPL = HD/KLPT int8 (KUNR
// steps of loads in flight); the GH quantised query slices live in
// registers in that layout; scores are sdot4s per head per row, reduced
// across the row's lanes with DPP. The P.V pass keeps LPT = HD/EPL lanes
// per V row (UNR steps in flight). Softmax runs online per TILE tokens;
// partial (m, l, o) into the buffers the merge reads.
//
// `ring_mask`: cache rows are stored at slot (position & ring_mask) — a
// sliding-window layer's cache is a power-of-two ring of the last
// positions; ~0u for a full-length cache (slot = position).
//
// Compiled with `#define HD <head_dim>` (a power of two, 64..512) and
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

// The score (Q.K) pass has its own lane mapping: KLPT lanes per K row,
// KEPL = HD / KLPT int8 each, read as uint4s. The P.V pass needs no
// per-row reduction and keeps LPT lanes per row; the scores pass does,
// and at HD 512 a 64-lane row cost a 6-step DPP reduction per head per
// row with 8-byte loads — the kernel ran at ~125 GB/s.
#ifndef KLPT
#define KLPT (HD >= 512 ? 16 : 8)
#endif
#define KEPL (HD / KLPT)            // int8 per lane in the scores pass
#define KWPL (KEPL / 4)
#define KTPW (64 / KLPT)            // K rows per wave step
#define KTPB (NW * KTPW)
#ifndef KUNR
#define KUNR 2
#endif
#ifndef MIN_CHUNK
#define MIN_CHUNK 1024
#endif
#ifndef MIN_SPLITS
#define MIN_SPLITS 16
#endif

static_assert(EPL == 8 || EPL == 16, "EPL is 8 or 16");
static_assert((KEPL % 16 == 0 || KEPL == 8) && KLPT >= 2 && KLPT <= 64, "scores pass: KEPL 8 or a multiple of 16");
#define KVEC (KEPL >= 16 ? 16 : 8)  // bytes per load: uint4, or uint2 at KEPL 8
typedef unsigned int kvec_t __attribute__((ext_vector_type(KVEC / 4)));
static_assert(TILE % (KTPB * KUNR) == 0, "tile must be a whole number of unrolled K steps");
static_assert(HD % 64 == 0 && LPT >= 4 && LPT <= 64, "head_dim must be 64..512, a multiple of 64");
static_assert(TILE == BS, "the exp pass maps one thread to one tile token");
static_assert(TILE % (TPB * UNR) == 0, "tile must be a whole number of unrolled block steps");
static_assert(GH >= 1 && GH <= 8, "GH <= 8");
static_assert((KLPT & (KLPT - 1)) == 0 && (LPT & (LPT - 1)) == 0, "lane groups must be powers of two (HD a power of two)");

__device__ __forceinline__ float slot_sum(float x) {
    for (int o = LPT; o < 64; o <<= 1) x += __shfl_xor(x, o);
    return x;
}
__device__ __forceinline__ float kslot_max(float x) {
    for (int o = KLPT; o < 64; o <<= 1) x = fmaxf(x, __shfl_xor(x, o));
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
                            unsigned int n_splits,
                            unsigned int ring_mask)
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
    // Splits in use: as many as give each at least MIN_CHUNK positions,
    // up to the n_splits launched. Decided from the device-side position,
    // so one captured graph launches enough splits for long contexts and
    // a short context leaves the extra workgroups empty (they write an
    // empty partial the merge ignores).
    const int eff = min((int)n_splits, max(MIN_SPLITS, (win_len + MIN_CHUNK - 1) / MIN_CHUNK));
    const int chunk = (win_len + eff - 1) / eff;
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
        }
    }

    // The same per-head int8 Q, laid out for the scores pass: this lane's
    // KEPL dims of each head.
    const int ktl = lane / KLPT;
    const int kdl = lane % KLPT;
    const bool kred_lane = (kdl == KLPT - 1);
    unsigned int qk[GH][KWPL];
    #pragma unroll
    for (int g = 0; g < GH; g++) {
        const float inv = dq[g] > 0.0f ? scaling / dq[g] : 0.0f;   // = 127 / amax
        #pragma unroll
        for (int c = 0; c < KWPL; c++) {
            float4 t = make_float4(0.f, 0.f, 0.f, 0.f);
            if (g < gh_n) t = *reinterpret_cast<const float4*>(q + (size_t)(h0 + g) * HD + kdl * KEPL + c * 4);
            qk[g][c] = pack4(t.x, t.y, t.z, t.w, inv);
        }
    }

    float m[GH], l[GH], acc[GH][EPL];
    #pragma unroll
    for (int g = 0; g < GH; g++) {
        m[g] = -INFINITY; l[g] = 0.0f;
        #pragma unroll
        for (int j = 0; j < EPL; j++) acc[g][j] = 0.0f;
    }
    const signed char* kbaseK = k_cache + (size_t)kvh * HD + kdl * KEPL;
    const signed char* vbase = v_cache + (size_t)kvh * HD + dl * EPL;
    const float* ks = k_scale + kvh;
    const float* vs = v_scale + kvh;

    for (int t0 = start; t0 < end; t0 += TILE) {
        const int tn = min(TILE, end - t0);
        float tmax[GH];
        #pragma unroll
        for (int g = 0; g < GH; g++) tmax[g] = -INFINITY;
        const int n_st = (tn + TPB * UNR - 1) / (TPB * UNR) * UNR;
        const int kn_st = (tn + KTPB * KUNR - 1) / (KTPB * KUNR) * KUNR;
        for (int st = 0; st < kn_st; st += KUNR) {
            kvec_t kk[KUNR][KEPL / KVEC]; float dk[KUNR];
            #pragma unroll
            for (int u = 0; u < KUNR; u++) {
                const int i = (st + u) * KTPB + wave * KTPW + ktl;
                const int t = (int)((unsigned)(t0 + min(i, tn - 1)) & ring_mask);
                const kvec_t* kp = reinterpret_cast<const kvec_t*>(kbaseK + (size_t)t * kv_row);
                #pragma unroll
                for (int c = 0; c < KEPL / KVEC; c++) kk[u][c] = kp[c];
                dk[u] = ks[(size_t)t * n_kv_heads];
            }
            #pragma unroll
            for (int u = 0; u < KUNR; u++) {
                const int i = (st + u) * KTPB + wave * KTPW + ktl;
                const bool live = kred_lane && (i < tn);
                const unsigned int* kw = reinterpret_cast<const unsigned int*>(kk[u]);
                float x[GH];
                #pragma unroll
                for (int g = 0; g < GH; g++) {
                    int idot = 0;
                    #pragma unroll
                    for (int c = 0; c < KWPL; c++)
                        idot = __builtin_amdgcn_sdot4((int)qk[g][c], (int)kw[c], idot, false);
                    x[g] = (float)idot * dq[g] * dk[u];
                }
                seg_sum_multi<GH, KLPT>(x);
                #pragma unroll
                for (int g = 0; g < GH; g++) {
                    if (live) { s_p[g][i] = x[g]; tmax[g] = fmaxf(tmax[g], x[g]); }
                }
            }
        }
        #pragma unroll
        for (int g = 0; g < GH; g++) {
            const float v = kslot_max(tmax[g]);
            if (lane == KLPT - 1) s_red[wave][g] = v;
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
                const int t = (int)((unsigned)(t0 + min(i, tn - 1)) & ring_mask);
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
