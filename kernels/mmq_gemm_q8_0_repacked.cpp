// int8 MMQ GEMM — repacked Q8_0 weights, dp4a, 2D-tiled with LDS
// staging. Y[P, out_dim] = Xq8[P, in_dim] · Wᵀ, consuming the quantised
// repacked weight directly (no dequant to fp16). Mirrors the Q4_K
// kernel's tile shape and dispatch; the per-sub-block math is simpler
// (no nibble unpack, no m/dmin term — Q8_0 stores signed int8 quants
// with a single fp16 scale per 32-element sub-block).
//
// Repacked layout (see src/quant/q8_0.rs::repack_for_matvec):
//   * quant planes: out_dim * nsp * 16 bytes each — quants 0-15, then 16-31.
//   * d  plane: out_dim * nsp * 2  bytes (fp16 scales).
// The two planes are concatenated in `wbase`.
//
// grid = (ceil(out_dim/BM), ceil(P/BN)).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#ifndef NARROW_THREADS
#define NARROW_THREADS 64
#endif
#ifndef NARROW_TXG
#define NARROW_TXG 4
#endif
#ifndef NARROW_BK
#define NARROW_BK 4
#endif
#ifndef NARROW_TM
#define NARROW_TM 1
#endif
#ifndef NARROW_TN
#define NARROW_TN 4
#endif
#define TYG (THREADS / TXG)  // row groups in the thread grid
#define BM (TYG * TM)        // weight rows per workgroup tile
#define BM_STRIDE TYG
#define BN (TXG * TN)        // tokens per workgroup tile

struct __attribute__((packed)) BlockQ8 {
    float  d;
    float  xsum;
    int8_t qs[32];
};
static_assert(sizeof(BlockQ8) == 40, "BlockQ8 must be 40 bytes");

template<int TM, int TN, int BK, int TXG, int THREADS>
__device__ __forceinline__
void mmq_q8_0_impl(const unsigned char* __restrict__ wbase,
                                const BlockQ8*       __restrict__ xq,
                                float*               __restrict__ y,
                                unsigned int in_dim,
                                unsigned int out_dim,
                                unsigned int p_rows)
{
    const int t  = threadIdx.x;          // 0..255
    const int tx = t % TXG;              // token group  0..TXG-1
    const int ty = t / TXG;              // row group    0..TYG-1
    const unsigned int row0 = blockIdx.x * BM;
    const unsigned int tok0 = blockIdx.y * BN;

    const unsigned int n_sub = in_dim >> 5;
    const unsigned int nsp = ((n_sub & (n_sub - 1u)) == 0u) ? (n_sub + 1u) : n_sub;
    // qs plane: int8 quants laid out [out_dim, nsp, 32]. Read as uint4.
    // Two-plane quant layout: quants 0-15 of every sub-block, then 16-31.
    const uint4* qlo = reinterpret_cast<const uint4*>(wbase);
    const uint4* qhi = reinterpret_cast<const uint4*>(wbase + (size_t)out_dim * nsp * 16);
    // d plane: fp16 scales [out_dim, nsp].
    const uint16_t* dp = reinterpret_cast<const uint16_t*>(
        wbase + (size_t)out_dim * nsp * 32);

    // Q8_0 sub-block = 32 bytes (32 signed int8s). uint4 is 16 bytes, so
    // we store two uint4s per sub-block in LDS — `sW_lo` for the first
    // 16 bytes, `sW_hi` for the last 16.
    __shared__ uint4   sW_lo[BM][BK];
    __shared__ uint4   sW_hi[BM][BK];
    __shared__ float   sWd[BM][BK];      // per-sub-block fp16 scale, dequant'd
    __shared__ BlockQ8 sX[BN][BK + 1];       // int8 acts

    float acc[TM][TN];
    #pragma unroll
    for (int r = 0; r < TM; r++)
        #pragma unroll
        for (int n = 0; n < TN; n++) acc[r][n] = 0.0f;

    for (unsigned int sb0 = 0; sb0 < n_sub; sb0 += BK) {
        // Cooperative load — consecutive threads → consecutive (row, sb).
        for (int e = t; e < BM * BK; e += THREADS) {
            const int lr = e / BK, lk = e % BK;
            const unsigned int wrow = row0 + lr;
            if (wrow < out_dim) {
                const unsigned int sb = sb0 + lk;
                // Two uint4 per sub-block at offsets (row*nsp + sb)*2 and *2+1.
                sW_lo[lr][lk] = qlo[(size_t)wrow * nsp + sb];
                sW_hi[lr][lk] = qhi[(size_t)wrow * nsp + sb];
                const uint16_t d_bits = dp[(size_t)wrow * nsp + sb];
                sWd[lr][lk] = __half2float(*reinterpret_cast<const __half*>(&d_bits));
            } else {
                sWd[lr][lk] = 0.0f;
            }
        }
        for (int e = t; e < BN * BK; e += THREADS) {
            const int lr = e / BK, lk = e % BK;
            const unsigned int xtok = tok0 + lr;
            if (xtok < p_rows) {
                sX[lr][lk] = xq[(size_t)xtok * n_sub + sb0 + lk];
            } else {
                sX[lr][lk].d = 0.0f;
                sX[lr][lk].xsum = 0.0f;
            }
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; kk++) {
            uint4 wq_lo[TM], wq_hi[TM];
            float dsc[TM];
            #pragma unroll
            for (int r = 0; r < TM; r++) {
                wq_lo[r] = sW_lo[ty + r * TYG][kk];
                wq_hi[r] = sW_hi[ty + r * TYG][kk];
                dsc[r]   = sWd[ty + r * TYG][kk];
            }
            #pragma unroll
            for (int n = 0; n < TN; n++) {
                const BlockQ8* xb   = &sX[tx + n * TXG][kk];
                const int*     xq32 = reinterpret_cast<const int*>(xb->qs);
                const float    dx   = xb->d;
                // Q8_0 has no m/offset term — sum is dsc * dx * (q · qx).
                // Eight sdot4s per (r, n): 4 from lo, 4 from hi.
                #pragma unroll
                for (int r = 0; r < TM; r++) {
                    const uint32_t lo[4] = { wq_lo[r].x, wq_lo[r].y, wq_lo[r].z, wq_lo[r].w };
                    const uint32_t hi[4] = { wq_hi[r].x, wq_hi[r].y, wq_hi[r].z, wq_hi[r].w };
                    int idot = 0;
                    #pragma unroll
                    for (int j = 0; j < 4; j++) {
                        idot = __builtin_amdgcn_sdot4((int)lo[j], xq32[j],     idot, false);
                        idot = __builtin_amdgcn_sdot4((int)hi[j], xq32[j + 4], idot, false);
                    }
                    acc[r][n] += dsc[r] * dx * (float)idot;
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int r = 0; r < TM; r++) {
        const unsigned int row = row0 + ty + r * TYG;
        if (row >= out_dim) continue;
        #pragma unroll
        for (int n = 0; n < TN; n++) {
            const unsigned int tok = tok0 + tx + n * TXG;
            if (tok < p_rows) y[(size_t)tok * out_dim + row] = acc[r][n];
        }
    }
}

extern "C" __global__ __launch_bounds__(256, 2)
void mmq_gemm_q8_0_repacked_f32(const unsigned char* __restrict__ wbase,
                                const BlockQ8*       __restrict__ xq,
                                float*               __restrict__ y,
                                unsigned int in_dim,
                                unsigned int out_dim,
                                unsigned int p_rows)
{
    mmq_q8_0_impl<4, 4, 4, 16, 256>(wbase, xq, y, in_dim, out_dim, p_rows);
}

// Narrow-token variant: TN=1 so BN=16 instead of 64.
//
// The wide tile is right for prefill, where P is hundreds of tokens. It is
// wrong for a 16-token verify: 16/64 of every workgroup does useful work
// and the other 48 token columns are masked-off waste. DFlash verifies a
// fixed 16-token block every round, so that waste was the single largest
// cost in a round. Narrowing the tile also cuts each thread's accumulators
// from TM*TN=16 to 4, which buys back occupancy.
extern "C" __global__ __launch_bounds__(NARROW_THREADS)
void mmq_gemm_q8_0_repacked_narrow_f32(const unsigned char* __restrict__ wbase,
                                const BlockQ8*       __restrict__ xq,
                                float*               __restrict__ y,
                                unsigned int in_dim,
                                unsigned int out_dim,
                                unsigned int p_rows)
{
    mmq_q8_0_impl<NARROW_TM, NARROW_TN, NARROW_BK, NARROW_TXG, NARROW_THREADS>(wbase, xq, y, in_dim, out_dim, p_rows);
}

// ---------------------------------------------------------------------
// `_rp_f32`: the llama fork's dense mmq_gemm_q8_0_repacked tile
// (gfx906-perf repack-gcn.cu; crossport ledger L8d) on reinstinct's
// two-plane weights and 40-byte BlockQ8 activations. Same tile and
// grid as `_f32` (64 rows x 64 tokens, 256 threads), but:
//   * 3 workgroups / CU (84-VGPR cap);
//   * the next BK step's weights are prefetched into registers during
//     the compute; activations load at the top of their own step;
//   * qs rows padded to 2*BK+1 uint4s, so b128 reads with distinct tx
//     rows are bank-conflict-free;
//   * each kk step runs as two 16-byte halves behind scheduling
//     barriers (hoisting both halves' LDS reads spills at the cap);
//   * the output tile is transposed through LDS for contiguous stores.
#define RP_BK 4
#define RP_TM 4
#define RP_TN 4
#define RP_BM (16 * RP_TM)
#define RP_XR (16 * RP_TN)
extern "C" __global__ __launch_bounds__(256, 3)
void mmq_gemm_q8_0_rp_f32(const unsigned char* __restrict__ wbase,
                          const BlockQ8*       __restrict__ xq,
                          float*               __restrict__ y,
                          unsigned int in_dim,
                          unsigned int out_dim,
                          unsigned int n_tok)
{
    const int t  = threadIdx.x;
    const int tx = t & 15;
    const int ty = t >> 4;
    const unsigned int row0 = blockIdx.x * RP_BM;
    const unsigned int tok0 = blockIdx.y * RP_XR;

    const unsigned int n_sub = in_dim >> 5;
    const unsigned int nsp   = ((n_sub & (n_sub - 1u)) == 0u) ? (n_sub + 1u) : n_sub;
    typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
    const u32x4*    qlo = reinterpret_cast<const u32x4*>(wbase);
    const u32x4*    qhi = reinterpret_cast<const u32x4*>(wbase + (size_t)out_dim * nsp * 16);
    const uint16_t* dp  = reinterpret_cast<const uint16_t*>(wbase + (size_t)out_dim * nsp * 32);

    constexpr int QS_LD  = 2 * RP_BK + 1;
    constexpr int Y_LD   = RP_BM + 2;
    constexpr int OFF_XQ = RP_BM * QS_LD * 16;
    constexpr int OFF_WD = OFF_XQ + RP_XR * QS_LD * 16;
    constexpr int OFF_XD = OFF_WD + RP_BM * RP_BK * 4;
    constexpr int SZ_K   = OFF_XD + RP_XR * RP_BK * 4;
    constexpr int SZ_Y   = RP_XR * Y_LD * 4;
    constexpr int SZ     = SZ_K > SZ_Y ? SZ_K : SZ_Y;
    __shared__ uint4 smem[SZ / 16];
    uint4 (*sW )[QS_LD]    = reinterpret_cast<uint4 (*)[QS_LD]>(smem);
    uint4 (*sXq)[QS_LD]    = reinterpret_cast<uint4 (*)[QS_LD]>((char*)smem + OFF_XQ);
    float (*sWd)[RP_BK]    = reinterpret_cast<float (*)[RP_BK]>((char*)smem + OFF_WD);
    float (*sXd)[RP_BK]    = reinterpret_cast<float (*)[RP_BK]>((char*)smem + OFF_XD);

    float acc[RP_TM][RP_TN] = {};
    const int lr = t >> 2;
    const int lk = t & 3;
    // Clamped staging rows: a clamped row only feeds outputs that are
    // never stored; a clamped sub-block is skipped by the K-tail check.
    const uint32_t w_off = min(row0 + lr, out_dim - 1) * nsp;
    const uint32_t x_off = min(tok0 + lr, n_tok - 1) * n_sub;

    u32x4 pw_lo, pw_hi, px_a, px_b;
    uint32_t pd, px_d;
    auto gload_w = [&](const unsigned int sb0) {
        const uint32_t wi = w_off + min(sb0 + lk, n_sub - 1);
        pw_lo = qlo[wi];
        pw_hi = qhi[wi];
        pd    = dp[wi];
    };
    // BlockQ8 dwords: [0] d (f32), [1] xsum (unused), [2..9] qs.
    auto gload_x = [&](const unsigned int sb0) {
        const uint32_t* xi = reinterpret_cast<const uint32_t*>(xq + (x_off + min(sb0 + lk, n_sub - 1)));
        px_d = xi[0];
        px_a = u32x4{xi[2], xi[3], xi[4], xi[5]};
        px_b = u32x4{xi[6], xi[7], xi[8], xi[9]};
    };
    auto lstore = [&]() {
        reinterpret_cast<u32x4*>(sW[lr])[2 * lk]     = pw_lo;
        reinterpret_cast<u32x4*>(sW[lr])[2 * lk + 1] = pw_hi;
        const uint16_t d_bits = (uint16_t)pd;
        sWd[lr][lk] = __half2float(*reinterpret_cast<const __half*>(&d_bits));
        reinterpret_cast<u32x4*>(sXq[lr])[2 * lk]     = px_a;
        reinterpret_cast<u32x4*>(sXq[lr])[2 * lk + 1] = px_b;
        sXd[lr][lk] = __uint_as_float(px_d);
    };

    gload_w(0);
    for (unsigned int sb0 = 0; sb0 < n_sub; sb0 += RP_BK) {
        gload_x(sb0);
        __syncthreads();
        lstore();
        __syncthreads();
        gload_w(sb0 + RP_BK);   // unconditional (clamped): a branch would force a wait at the join
        const int kk_end = min((int)RP_BK, (int)(n_sub - sb0));
        #pragma unroll 1
        for (int kk = 0; kk < kk_end; kk++) {
            int idot[RP_TM][RP_TN];
            #pragma unroll
            for (int hh = 0; hh < 2; hh++) {
                u32x4 wq[RP_TM];
                #pragma unroll
                for (int r = 0; r < RP_TM; r++)
                    wq[r] = reinterpret_cast<const u32x4*>(sW[ty + r * 16])[2 * kk + hh];
                __builtin_amdgcn_sched_barrier(0);
                #pragma unroll
                for (int n = 0; n < RP_TN; n++) {
                    const u32x4 xv = reinterpret_cast<const u32x4*>(sXq[tx + n * 16])[2 * kk + hh];
                    #pragma unroll
                    for (int r = 0; r < RP_TM; r++) {
                        int d = hh ? idot[r][n] : 0;
                        d = __builtin_amdgcn_sdot4((int)wq[r].x, (int)xv.x, d, false);
                        d = __builtin_amdgcn_sdot4((int)wq[r].y, (int)xv.y, d, false);
                        d = __builtin_amdgcn_sdot4((int)wq[r].z, (int)xv.z, d, false);
                        d = __builtin_amdgcn_sdot4((int)wq[r].w, (int)xv.w, d, false);
                        idot[r][n] = d;
                    }
                }
                __builtin_amdgcn_sched_barrier(0);
            }
            #pragma unroll
            for (int n = 0; n < RP_TN; n++) {
                const float dx = sXd[tx + n * 16][kk];
                #pragma unroll
                for (int r = 0; r < RP_TM; r++)
                    acc[r][n] += sWd[ty + r * 16][kk] * dx * (float)idot[r][n];
            }
        }
    }
    __syncthreads();

    // Transpose through LDS so each token row is stored contiguously.
    float (*tileY)[Y_LD] = reinterpret_cast<float (*)[Y_LD]>(smem);
    #pragma unroll
    for (int r = 0; r < RP_TM; r++)
        #pragma unroll
        for (int n = 0; n < RP_TN; n++)
            tileY[tx + n * 16][ty + r * 16] = acc[r][n];
    __syncthreads();
    for (int idx = t; idx < RP_BM * RP_XR; idx += 256) {
        const int tok = idx / RP_BM;
        const int row = idx % RP_BM;
        if (tok0 + tok < n_tok && row0 + row < out_dim)
            y[(size_t)(tok0 + tok) * out_dim + row0 + row] = tileY[tok][row];
    }
}
