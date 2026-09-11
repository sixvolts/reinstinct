// int8 MMQ GEMM — repacked Q4_K weights, dp4a, 2D-tiled with LDS
// staging. Y[P, out_dim] = Xq8[P, in_dim] · Wᵀ, consuming the quantised
// repacked weight directly (no dequant to fp16).
//
// A workgroup (256 threads, a 16×16 thread grid) computes a BM×BN
// output tile — BM = 16·TM weight rows, BN = 16·TN tokens. The
// contraction is walked in BK-sub-block chunks; each chunk's weight and
// activation tiles are cooperatively loaded into LDS, then every thread
// computes its TM×TN register micro-tile straight from LDS.
//
// The micro-tile is *strided* — thread (tx,ty) owns rows ty,ty+16,… and
// tokens tx,tx+16,… — so a wavefront's 16 distinct token reads land on
// 16 distinct LDS banks (a blocked mapping collides on the 40-byte
// BlockQ8 stride). The weight is held per chunk as raw uint4 and the
// nibbles unpacked inline (storing the unpacked planes spilled VGPRs).
//
// grid = (ceil(out_dim/BM), ceil(P/BN)).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

// Tile shape. An empirical sweep (BK ∈ {4,8}, TM/TN ∈ {4,8}, occupancy
// 1/2) found this 4×4 / occupancy-2 point flat-optimal: TM=8 at
// occupancy 1 won P=512 by ~2.6% but lost P=128 and ran at 235 VGPR
// (one edit from a catastrophic spill); BK=8 was slower. 4×4/occ-2 is
// the robust choice — alternatives are within measurement noise.
#ifndef NARROW_TXG
#define NARROW_TXG 16
#endif
#ifndef NARROW_BK
#define NARROW_BK 4
#endif
#ifndef NARROW_TM
#define NARROW_TM 2
#endif
#ifndef NARROW_TN
#define NARROW_TN 1
#endif
#define TYG (256 / TXG)      // row groups in the 256-thread grid
#define BM (TYG * TM)        // weight rows per workgroup tile
#define BM_STRIDE TYG
#define BN (TXG * TN)        // tokens per workgroup tile

struct __attribute__((packed)) BlockQ8 {
    float  d;
    float  xsum;
    int8_t qs[32];
};
static_assert(sizeof(BlockQ8) == 40, "BlockQ8 must be 40 bytes");

template<int TM, int TN, int BK, int TXG>
__device__ __forceinline__
void mmq_q4k_impl(const unsigned char* __restrict__ wbase,
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
    const unsigned int n_super = n_sub >> 3;       // 256-weight superblocks/row
    const uint4*    nib = reinterpret_cast<const uint4*>(wbase);
    const uint16_t* smp = reinterpret_cast<const uint16_t*>(   // v2: sc|m per sub-block
        wbase + (size_t)out_dim * nsp * 16);
    const uint32_t* ddp = reinterpret_cast<const uint32_t*>(   // v2: d|dmin per superblock
        wbase + (size_t)out_dim * nsp * 16 + (size_t)out_dim * nsp * 2);

    __shared__ uint4   sW[BM][BK];       // packed nibbles
    __shared__ float2  sWs[BM][BK];      // (dsc, deff) — formed from the v2 scales
    __shared__ BlockQ8 sX[BN][BK + 1];       // int8 acts

    float acc[TM][TN];
    #pragma unroll
    for (int r = 0; r < TM; r++)
        #pragma unroll
        for (int n = 0; n < TN; n++) acc[r][n] = 0.0f;

    for (unsigned int sb0 = 0; sb0 < n_sub; sb0 += BK) {
        // Cooperative load — consecutive threads → consecutive (row,sb).
        for (int e = t; e < BM * BK; e += 256) {
            const int lr = e / BK, lk = e % BK;
            const unsigned int wrow = row0 + lr;
            if (wrow < out_dim) {
                const unsigned int sb = sb0 + lk;
                sW[lr][lk] = nib[(size_t)wrow * nsp + sb];
                const uint16_t sm = smp[(size_t)wrow * nsp + sb];
                const uint32_t dd = ddp[(size_t)wrow * n_super + (sb >> 3)];
                const uint16_t d_bits    = (uint16_t)(dd & 0xFFFF);
                const uint16_t dmin_bits = (uint16_t)(dd >> 16);
                sWs[lr][lk] = make_float2(
                    __half2float(*reinterpret_cast<const __half*>(&d_bits))
                        * (float)(sm & 0xFFu),
                    __half2float(*reinterpret_cast<const __half*>(&dmin_bits))
                        * (float)(sm >> 8));
            } else {
                sWs[lr][lk] = make_float2(0.0f, 0.0f);
            }
        }
        for (int e = t; e < BN * BK; e += 256) {
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
            uint4 wq[TM];
            float dsc[TM], deff[TM];
            #pragma unroll
            for (int r = 0; r < TM; r++) {
                wq[r] = sW[ty + r * TYG][kk];
                const float2 s = sWs[ty + r * TYG][kk];
                dsc[r]  = s.x;
                deff[r] = s.y;
            }
            #pragma unroll
            for (int n = 0; n < TN; n++) {
                const BlockQ8* xb   = &sX[tx + n * TXG][kk];
                const int*     xq32 = reinterpret_cast<const int*>(xb->qs);
                const float    dx   = xb->d;
                const float    xsum = xb->xsum;
                #pragma unroll
                for (int r = 0; r < TM; r++) {
                    const uint32_t qa[4] = { wq[r].x, wq[r].y, wq[r].z, wq[r].w };
                    int idot = 0;
                    #pragma unroll
                    for (int j = 0; j < 4; j++) {
                        idot = __builtin_amdgcn_sdot4(
                            (int)( qa[j]       & 0x0F0F0F0Fu), xq32[j],     idot, false);
                        idot = __builtin_amdgcn_sdot4(
                            (int)((qa[j] >> 4) & 0x0F0F0F0Fu), xq32[j + 4], idot, false);
                    }
                    acc[r][n] += dsc[r] * dx * (float)idot - deff[r] * xsum;
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
void mmq_gemm_q4k_repacked_f32(const unsigned char* __restrict__ wbase,
                               const BlockQ8*       __restrict__ xq,
                               float*               __restrict__ y,
                               unsigned int in_dim,
                               unsigned int out_dim,
                               unsigned int p_rows)
{
    mmq_q4k_impl<4, 4, 4, 16>(wbase, xq, y, in_dim, out_dim, p_rows);
}

// Narrow-token variant: TN=1 so BN=16 instead of 64.
//
// The wide tile is right for prefill, where P is hundreds of tokens. It is
// wrong for a 16-token verify: 16/64 of every workgroup does useful work
// and the other 48 token columns are masked-off waste. DFlash verifies a
// fixed 16-token block every round, so that waste was the single largest
// cost in a round. Narrowing the tile also cuts each thread's accumulators
// from TM*TN=16 to 4, which buys back occupancy.
extern "C" __global__ __launch_bounds__(256, 4)
void mmq_gemm_q4k_repacked_narrow_f32(const unsigned char* __restrict__ wbase,
                               const BlockQ8*       __restrict__ xq,
                               float*               __restrict__ y,
                               unsigned int in_dim,
                               unsigned int out_dim,
                               unsigned int p_rows)
{
    mmq_q4k_impl<NARROW_TM, NARROW_TN, NARROW_BK, NARROW_TXG>(wbase, xq, y, in_dim, out_dim, p_rows);
}
