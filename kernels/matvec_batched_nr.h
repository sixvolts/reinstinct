// Shared body of the small-batch matvecs (spec-decode verify: 1..8
// activation rows against one repacked weight).
//
// Each weight sub-block is read once and dotted against every
// activation row. A workgroup is 4 waves x ROWS output rows; lane l of
// a wave takes sub-blocks l, l+64, ... of its ROWS weight rows. The
// activation (BlockQ8 [n_rows, n_sub]) is the same for all 4 waves and
// every workgroup, so with LDS each 64-sub-block chunk of all NR rows is
// staged once per workgroup instead of each wave fetching it through
// L1 — NR x 40 bytes per lane per ROWS weight sub-blocks is otherwise
// several times the weight traffic.
//
// NR <= 4: every load of a trip (ROWS weight sub-blocks, NR activation
// blocks) is issued before the first dot, behind a scheduling barrier —
// the decode matvec's schedule. NR > 4: the weight rows are decoded
// into registers and the activation rows streamed through them one at a
// time (holding NR activation blocks would cost NR x 10 VGPRs).
//
// EXACT: n_rows == NR, known at compile time. Otherwise n_rows <= NR;
// rows past n_rows read row n_rows-1 and are not written.
//
// The weight format is a decoder W:
//   W(wbase, out_dim, nsp, n_super)   plane pointers
//   W::Raw  load(row, sb) const       the loads for one sub-block
//   W::Dec  decode(const Raw&)        unpacked weights + scales
//   float   dot(const Dec&, const XAct&)
//   HALF_SUMS                         dot needs the int sums of each
//                                     16-element half of the activation
//
// Launch: 256 threads, grid.x = ceil(out_dim / (4 * ROWS)).

#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include "gfx906_dpp.h"

struct __attribute__((packed)) BlockQ8 {
    float  d;
    float  xsum;      // d * sum(qs)
    int8_t qs[32];
};
static_assert(sizeof(BlockQ8) == 40, "BlockQ8 must be 40 bytes");

// One activation sub-block as the dot functions see it.
struct XAct {
    int   w[8];
    float d, s;       // scale, d * sum(qs)
    int   h0, h1;     // sum(qs[0..16]), sum(qs[16..32]) — HALF_SUMS only
};

__device__ __forceinline__ float bnr_f16(uint16_t b) {
    return __half2float(*reinterpret_cast<const __half*>(&b));
}

__device__ __forceinline__ void bnr_half_sums(const int w[8], int& h0, int& h1) {
    int s0 = 0, s1 = 0;
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        s0 = __builtin_amdgcn_sdot4(w[j],     0x01010101, s0, false);
        s1 = __builtin_amdgcn_sdot4(w[j + 4], 0x01010101, s1, false);
    }
    h0 = s0; h1 = s1;
}

template<class W, int ROWS, int NR, bool LDS, bool EXACT>
__device__ __forceinline__
void mv_bnr(const uint8_t* __restrict__ wbase, const BlockQ8* __restrict__ xq,
            float* __restrict__ y, unsigned int in_dim, unsigned int out_dim,
            unsigned int n_rows)
{
    const int wave = threadIdx.x >> 6;
    const int lane = threadIdx.x & 63;
    const int row0 = blockIdx.x * (ROWS * 4) + wave * ROWS;
    const unsigned int n_sub = in_dim >> 5;
    const unsigned int nsp = ((n_sub & (n_sub - 1u)) == 0u) ? (n_sub + 1u) : n_sub;
    const W w(wbase, out_dim, nsp, n_sub >> 3);
    const int rmax = (int)out_dim - 1;
    const unsigned int nr = EXACT ? (unsigned int)NR : n_rows;

    float acc[ROWS][NR];
    #pragma unroll
    for (int r = 0; r < ROWS; r++)
        #pragma unroll
        for (int b = 0; b < NR; b++) acc[r][b] = 0.0f;

    __shared__ int4  s_q[LDS ? NR : 1][2][64];
    __shared__ float s_d[LDS ? NR : 1][64], s_s[LDS ? NR : 1][64];
    __shared__ int2  s_h[LDS && W::HALF_SUMS ? NR : 1][64];

    // Activation row b (clamped to the last present row), sub-block sb.
    auto fetch = [&](int b, unsigned int sb) {
        XAct x;
        if constexpr (LDS) {
            const int4 a = s_q[b][0][lane], c = s_q[b][1][lane];
            x.w[0] = a.x; x.w[1] = a.y; x.w[2] = a.z; x.w[3] = a.w;
            x.w[4] = c.x; x.w[5] = c.y; x.w[6] = c.z; x.w[7] = c.w;
            x.d = s_d[b][lane]; x.s = s_s[b][lane];
            if constexpr (W::HALF_SUMS) { const int2 h = s_h[b][lane]; x.h0 = h.x; x.h1 = h.y; }
        } else {
            const unsigned int bb = EXACT ? (unsigned int)b : min((unsigned int)b, nr - 1u);
            const BlockQ8* xb = xq + (size_t)bb * n_sub + sb;
            const int4* x4 = reinterpret_cast<const int4*>(xb->qs);
            const int4 a = x4[0], c = x4[1];
            x.w[0] = a.x; x.w[1] = a.y; x.w[2] = a.z; x.w[3] = a.w;
            x.w[4] = c.x; x.w[5] = c.y; x.w[6] = c.z; x.w[7] = c.w;
            x.d = xb->d; x.s = xb->xsum;
        }
        return x;
    };

    for (unsigned int c0 = 0; c0 < n_sub; c0 += 64) {
        const unsigned int sb = c0 + lane;
        if constexpr (LDS) {
            __syncthreads();
            for (int e = threadIdx.x; e < NR * 64; e += 256) {
                const int b = e >> 6, l = e & 63;
                const unsigned int bb = EXACT ? (unsigned int)b : min((unsigned int)b, nr - 1u);
                const BlockQ8* xb = xq + (size_t)bb * n_sub + min(c0 + l, n_sub - 1u);
                const int4* x4 = reinterpret_cast<const int4*>(xb->qs);
                const int4 a = x4[0], c = x4[1];
                s_q[b][0][l] = a; s_q[b][1][l] = c;
                s_d[b][l] = xb->d; s_s[b][l] = xb->xsum;
                if constexpr (W::HALF_SUMS) {
                    const int t[8] = { a.x, a.y, a.z, a.w, c.x, c.y, c.z, c.w };
                    int h0, h1; bnr_half_sums(t, h0, h1);
                    s_h[b][l] = make_int2(h0, h1);
                }
            }
            __syncthreads();
        }
        if (sb >= n_sub) continue;

        typename W::Raw raw[ROWS];
        #pragma unroll
        for (int r = 0; r < ROWS; r++) raw[r] = w.load(min(row0 + r, rmax), sb);

        if constexpr (NR <= 4) {
            XAct x[NR];
            #pragma unroll
            for (int b = 0; b < NR; b++) x[b] = fetch(b, sb);
            __builtin_amdgcn_sched_barrier(0);
            #pragma unroll
            for (int b = 0; b < NR; b++)
                if constexpr (W::HALF_SUMS && !LDS) bnr_half_sums(x[b].w, x[b].h0, x[b].h1);
            #pragma unroll
            for (int r = 0; r < ROWS; r++) {
                const typename W::Dec d = W::decode(raw[r]);
                #pragma unroll
                for (int b = 0; b < NR; b++) acc[r][b] += W::dot(d, x[b]);
            }
        } else {
            typename W::Dec d[ROWS];
            #pragma unroll
            for (int r = 0; r < ROWS; r++) d[r] = W::decode(raw[r]);
            #pragma unroll
            for (int b = 0; b < NR; b++) {
                if (EXACT || (unsigned int)b < nr) {
                    XAct x = fetch(b, sb);
                    if constexpr (W::HALF_SUMS && !LDS) bnr_half_sums(x.w, x.h0, x.h1);
                    #pragma unroll
                    for (int r = 0; r < ROWS; r++) acc[r][b] += W::dot(d[r], x);
                }
            }
        }
    }

    #pragma unroll
    for (int r = 0; r < ROWS; r++)
        #pragma unroll
        for (int b = 0; b < NR; b++) {
            if (EXACT || (unsigned int)b < nr) {
                const float a = wave64_reduce_add_f32(acc[r][b]);
                if (lane == 0 && row0 + r <= rmax) y[(size_t)b * out_dim + (row0 + r)] = a;
            }
        }
}

// One entry point; matvec_batched_nr_entries.h instantiates the set.
#define BNR_ENTRY(PREFIX, W, NAME, ROWS, NR, LDS, EXACT)                              \
extern "C" __global__ __launch_bounds__(256) void PREFIX##_##NAME##_f32(               \
    const uint8_t* __restrict__ wbase, const BlockQ8* __restrict__ xq,                 \
    float* __restrict__ y, unsigned int in_dim, unsigned int out_dim,                  \
    unsigned int n_rows)                                                               \
{ mv_bnr<W, ROWS, NR, LDS, EXACT>(wbase, xq, y, in_dim, out_dim, n_rows); }
