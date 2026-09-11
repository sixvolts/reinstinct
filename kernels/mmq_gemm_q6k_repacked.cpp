// int8 MMQ GEMM — repacked Q6_K weights, dp4a, 2D-tiled with LDS
// staging. The Q6_K analogue of mmq_gemm_q4k_repacked: same 64×64 tile,
// strided micro-tile and launch bounds. Q6_K is symmetric (quant−32) so
// each sub-block carries an 8-byte high-bit plane and a per-token
// activation sum supplies the −32 correction.
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

// Spread 2-bit groups (b0b1, b2b3, …) to bits 4-5 of bytes 0..3.
__device__ __forceinline__ uint32_t spread2(uint32_t h) {
    return ((h & 0x03u) << 4) | ((h & 0x0Cu) << 10)
         | ((h & 0x30u) << 16) | ((h & 0xC0u) << 22);
}

template<int TM, int TN, int BK, int TXG, int THREADS>
__device__ __forceinline__
void mmq_q6k_impl(const unsigned char* __restrict__ wbase,
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
    const uint2*    h2p = reinterpret_cast<const uint2*>(
        wbase + (size_t)out_dim * nsp * 16);
    const uint16_t* smp = reinterpret_cast<const uint16_t*>(   // v2: sc_lo|sc_hi int8
        wbase + (size_t)out_dim * nsp * 16 + (size_t)out_dim * nsp * 8);
    const uint16_t* ddp = reinterpret_cast<const uint16_t*>(   // v2: d per superblock
        wbase + (size_t)out_dim * nsp * 16 + (size_t)out_dim * nsp * 8
              + (size_t)out_dim * nsp * 2);

    __shared__ uint4    sW  [BM][BK];    // packed nibbles
    __shared__ uint2    sWh2[BM][BK];    // high-bit plane
    __shared__ float2   sWs [BM][BK];    // (dsc_lo, dsc_hi) — from the v2 scales
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
                sW  [lr][lk] = nib[(size_t)wrow * nsp + sb];
                sWh2[lr][lk] = h2p[(size_t)wrow * nsp + sb];
                const uint16_t sm     = smp[(size_t)wrow * nsp + sb];
                const uint16_t d_bits = ddp[(size_t)wrow * n_super + (sb >> 3)];
                const float d = __half2float(*reinterpret_cast<const __half*>(&d_bits));
                sWs[lr][lk] = make_float2(d * (float)(int)(int8_t)(sm & 0xFFu),
                                          d * (float)(int)(int8_t)(sm >> 8));
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
            uint4 wq[TM]; uint2 wh2[TM];
            float dlo[TM], dhi[TM];
            #pragma unroll
            for (int r = 0; r < TM; r++) {
                wq[r]  = sW  [ty + r * TYG][kk];
                wh2[r] = sWh2[ty + r * TYG][kk];
                const float2 s = sWs[ty + r * TYG][kk];
                dlo[r] = s.x;
                dhi[r] = s.y;
            }
            #pragma unroll
            for (int n = 0; n < TN; n++) {
                const BlockQ8* xb   = &sX[tx + n * TXG][kk];
                const int*     xq32 = reinterpret_cast<const int*>(xb->qs);
                const float    dx   = xb->d;
                // Activation sums for the symmetric −32 correction.
                int xis0 = 0, xis1 = 0;
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    xis0 = __builtin_amdgcn_sdot4(xq32[j],     0x01010101, xis0, false);
                    xis1 = __builtin_amdgcn_sdot4(xq32[j + 4], 0x01010101, xis1, false);
                }
                #pragma unroll
                for (int r = 0; r < TM; r++) {
                    const uint32_t qa[4] = { wq[r].x, wq[r].y, wq[r].z, wq[r].w };
                    const uint32_t h2lo = wh2[r].x, h2hi = wh2[r].y;
                    int idot0 = 0, idot1 = 0;
                    #pragma unroll
                    for (int j = 0; j < 4; j++) {
                        const uint32_t ge = 2 * j;
                        const uint32_t go = 2 * j + 1;
                        const uint32_t he = ((ge < 4 ? h2lo : h2hi) >> (8 * (ge & 3))) & 0xFFu;
                        const uint32_t ho = ((go < 4 ? h2lo : h2hi) >> (8 * (go & 3))) & 0xFFu;
                        const uint32_t q6lo = ( qa[j]       & 0x0F0F0F0Fu) | spread2(he);
                        const uint32_t q6hi = ((qa[j] >> 4) & 0x0F0F0F0Fu) | spread2(ho);
                        idot0 = __builtin_amdgcn_sdot4((int)q6lo, xq32[j],     idot0, false);
                        idot1 = __builtin_amdgcn_sdot4((int)q6hi, xq32[j + 4], idot1, false);
                    }
                    acc[r][n] += dlo[r] * dx * (float)(idot0 - 32 * xis0)
                               + dhi[r] * dx * (float)(idot1 - 32 * xis1);
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
void mmq_gemm_q6k_repacked_f32(const unsigned char* __restrict__ wbase,
                               const BlockQ8*       __restrict__ xq,
                               float*               __restrict__ y,
                               unsigned int in_dim,
                               unsigned int out_dim,
                               unsigned int p_rows)
{
    mmq_q6k_impl<4, 4, 4, 16, 256>(wbase, xq, y, in_dim, out_dim, p_rows);
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
void mmq_gemm_q6k_repacked_narrow_f32(const unsigned char* __restrict__ wbase,
                               const BlockQ8*       __restrict__ xq,
                               float*               __restrict__ y,
                               unsigned int in_dim,
                               unsigned int out_dim,
                               unsigned int p_rows)
{
    mmq_q6k_impl<NARROW_TM, NARROW_TN, NARROW_BK, NARROW_TXG, NARROW_THREADS>(wbase, xq, y, in_dim, out_dim, p_rows);
}
