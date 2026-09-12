// int8 MMQ GEMM — repacked Q5_K weights, dp4a, 2D-tiled with LDS
// staging. The Q5_K analogue of mmq_gemm_q4k_repacked: same 64×64 tile,
// strided micro-tile and launch bounds; the weight carries an extra qh
// plane (the 5th bit of each quant) staged alongside the nibbles.
//
// grid = (ceil(out_dim/64), ceil(P/64)).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#define TYG (THREADS / TXG)  // row groups in the thread grid
#define BM (TYG * TM)        // weight rows per workgroup tile
#define BN (TXG * TN)        // tokens per workgroup tile
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

struct __attribute__((packed)) BlockQ8 {
    float  d;
    float  xsum;
    int8_t qs[32];
};
static_assert(sizeof(BlockQ8) == 40, "BlockQ8 must be 40 bytes");

// Spread 4 bits (b0..b3) to bit 4 of bytes 0..3.
__device__ __forceinline__ uint32_t spread4(uint32_t h) {
    return ((h & 1u) << 4) | ((h & 2u) << 11) | ((h & 4u) << 18) | ((h & 8u) << 25);
}

// The weight tile is expanded to int8 at LDS-load time — two uint4 per
// 32-weight sub-block (weights 0-15, then 16-31) in the order the
// activation's dp4a groups use — so the inner loop is the plain int8
// dot of the Q8_0 kernel. Unpacking there instead cost TM*TN*BK*8
// unpacks per tile against one per thread here, and the register
// pressure of the packed form spilled (5-49 VGPRs across the K-quants;
// Q6_K ran at 5.7 TOPS against Q8_0's 19.7 at the 27B FFN shape).

template<int TM, int TN, int BK, int TXG, int THREADS>
__device__ __forceinline__
void mmq_q5k_impl(const unsigned char* __restrict__ wbase,
                               const BlockQ8*       __restrict__ xq,
                               float*               __restrict__ y,
                               unsigned int in_dim,
                               unsigned int out_dim,
                               unsigned int p_rows)
{
    const int t  = threadIdx.x;
    const int tx = t % TXG;              // token group  0..TXG-1
    const int ty = t / TXG;              // row group    0..TYG-1
    const unsigned int row0 = blockIdx.x * BM;
    const unsigned int tok0 = blockIdx.y * BN;

    const unsigned int n_sub = in_dim >> 5;
    const unsigned int nsp = ((n_sub & (n_sub - 1u)) == 0u) ? (n_sub + 1u) : n_sub;
    const unsigned int n_super = n_sub >> 3;
    const uint4*    nib = reinterpret_cast<const uint4*>(wbase);
    const uint32_t* qhp = reinterpret_cast<const uint32_t*>(
        wbase + (size_t)out_dim * nsp * 16);
    const uint16_t* smp = reinterpret_cast<const uint16_t*>(   // v2: sc|m per sub-block
        wbase + (size_t)out_dim * nsp * 16 + (size_t)out_dim * nsp * 4);
    const uint32_t* ddp = reinterpret_cast<const uint32_t*>(   // v2: d|dmin per superblock
        wbase + (size_t)out_dim * nsp * 16 + (size_t)out_dim * nsp * 4
              + (size_t)out_dim * nsp * 2);

    __shared__ uint4    sW_lo[BM][BK];   // int8 weights 0-15 (5-bit values expanded)
    __shared__ uint4    sW_hi[BM][BK];   // int8 weights 16-31
    __shared__ float2   sWs [BM][BK];    // (dsc, deff) — formed from the v2 scales
    __shared__ BlockQ8  sX  [BN][BK + 1];    // int8 acts

    float acc[TM][TN];
    #pragma unroll
    for (int r = 0; r < TM; r++)
        #pragma unroll
        for (int n = 0; n < TN; n++) acc[r][n] = 0.0f;


    for (unsigned int sb0 = 0; sb0 < n_sub; sb0 += BK) {
        // Strided cooperative loads: the tile is BM*BK / BN*BK elements,
        // which is only 256 at the default BM=BN=64, BK=4. The narrow
        // tile is smaller, so a fixed one-element-per-thread mapping
        // would run threads off the end of the LDS arrays.
        for (int e = t; e < BM * BK; e += THREADS) {
            const int lr = e / BK, lk = e % BK;
            const unsigned int sb = sb0 + lk;
            const unsigned int wrow = row0 + lr;
            if (wrow < out_dim) {
                const uint4    q  = nib[(size_t)wrow * nsp + sb];
                const uint32_t qh = qhp[(size_t)wrow * nsp + sb];
                const uint32_t M = 0x0F0F0F0Fu;
                sW_lo[lr][lk] = make_uint4(
                    (q.x & M) | spread4( qh        & 0xFu), (q.y & M) | spread4((qh >>  8) & 0xFu),
                    (q.z & M) | spread4((qh >> 16) & 0xFu), (q.w & M) | spread4((qh >> 24) & 0xFu));
                sW_hi[lr][lk] = make_uint4(
                    ((q.x >> 4) & M) | spread4((qh >>  4) & 0xFu), ((q.y >> 4) & M) | spread4((qh >> 12) & 0xFu),
                    ((q.z >> 4) & M) | spread4((qh >> 20) & 0xFu), ((q.w >> 4) & M) | spread4((qh >> 28) & 0xFu));
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
        for (int e = t; e < BN * BK; e += THREADS) {
            const int lr = e / BK, lk = e % BK;
            const unsigned int sb = sb0 + lk;
            const unsigned int xtok = tok0 + lr;
            if (xtok < p_rows) {
                sX[lr][lk] = xq[(size_t)xtok * n_sub + sb];
            } else {
                sX[lr][lk].d = 0.0f;
                sX[lr][lk].xsum = 0.0f;
            }
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; kk++) {
            uint4 wlo[TM], whi[TM];
            float dsc[TM], deff[TM];
            #pragma unroll
            for (int r = 0; r < TM; r++) {
                wlo[r] = sW_lo[ty + r * TYG][kk];
                whi[r] = sW_hi[ty + r * TYG][kk];
                const float2 s = sWs[ty + r * TYG][kk];
                dsc[r]  = s.x;
                deff[r] = s.y;
            }
            // Not fully unrolled: with the int8-expanded tile the fully
            // unrolled token loop spilled 49 VGPRs (17 ms at the 27B FFN
            // shape); unroll 2 fits (7.7 ms).
            #pragma unroll 2
            for (int n = 0; n < TN; n++) {
                const BlockQ8* xb   = &sX[tx + n * TXG][kk];
                const int*     xq32 = reinterpret_cast<const int*>(xb->qs);
                const float    dx   = xb->d;
                const float    xsum = xb->xsum;
                #pragma unroll
                for (int r = 0; r < TM; r++) {
                    const int wa[8] = { (int)wlo[r].x, (int)wlo[r].y, (int)wlo[r].z, (int)wlo[r].w,
                                        (int)whi[r].x, (int)whi[r].y, (int)whi[r].z, (int)whi[r].w };
                    int idot = 0;
                    #pragma unroll
                    for (int j = 0; j < 8; j++)
                        idot = __builtin_amdgcn_sdot4(wa[j], xq32[j], idot, false);
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
void mmq_gemm_q5k_repacked_f32(const unsigned char* __restrict__ wbase,
                               const BlockQ8*       __restrict__ xq,
                               float*               __restrict__ y,
                               unsigned int in_dim,
                               unsigned int out_dim,
                               unsigned int p_rows)
{
    mmq_q5k_impl<4, 4, 4, 16, 256>(wbase, xq, y, in_dim, out_dim, p_rows);
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
void mmq_gemm_q5k_repacked_narrow_f32(const unsigned char* __restrict__ wbase,
                               const BlockQ8*       __restrict__ xq,
                               float*               __restrict__ y,
                               unsigned int in_dim,
                               unsigned int out_dim,
                               unsigned int p_rows)
{
    mmq_q5k_impl<NARROW_TM, NARROW_TN, NARROW_BK, NARROW_TXG, NARROW_THREADS>(wbase, xq, y, in_dim, out_dim, p_rows);
}
