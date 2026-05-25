// int8 MMQ GEMM — repacked Q5_K weights, dp4a, 2D-tiled with LDS
// staging. The Q5_K analogue of mmq_gemm_q4k_repacked: same 64×64 tile,
// strided micro-tile and launch bounds; the weight carries an extra qh
// plane (the 5th bit of each quant) staged alongside the nibbles.
//
// grid = (ceil(out_dim/64), ceil(P/64)).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include "gfx906_dpp.h"

#define BM 64
#define BN 64
#define BK 4
#define TM 4
#define TN 4

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

extern "C" __global__ __launch_bounds__(256, 2)
void mmq_gemm_q5k_repacked_f32(const unsigned char* __restrict__ wbase,
                               const BlockQ8*       __restrict__ xq,
                               float*               __restrict__ y,
                               unsigned int in_dim,
                               unsigned int out_dim,
                               unsigned int p_rows)
{
    const int t  = threadIdx.x;
    const int tx = t & 15;
    const int ty = t >> 4;
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

    __shared__ uint4    sW  [BM][BK];    // packed nibbles
    __shared__ uint32_t sWqh[BM][BK];    // 5th-bit plane
    __shared__ float2   sWs [BM][BK];    // (dsc, deff) — formed from the v2 scales
    __shared__ BlockQ8  sX  [BN][BK + 1];    // int8 acts

    float acc[TM][TN];
    #pragma unroll
    for (int r = 0; r < TM; r++)
        #pragma unroll
        for (int n = 0; n < TN; n++) acc[r][n] = 0.0f;

    const int lr = t >> 2;
    const int lk = t & 3;

    for (unsigned int sb0 = 0; sb0 < n_sub; sb0 += BK) {
        const unsigned int sb = sb0 + lk;
        const unsigned int wrow = row0 + lr;
        if (wrow < out_dim) {
            // Weight streams: read-once-per-K-tile, never reused — bypass
            // L1/L2 so the activation tile + LDS scales stay cache-resident.
            sW  [lr][lk] = loadnt_uint4(nib + (size_t)wrow * nsp + sb);
            sWqh[lr][lk] = loadnt(qhp + (size_t)wrow * nsp + sb);
            const uint16_t sm = loadnt(smp + (size_t)wrow * nsp + sb);
            const uint32_t dd = loadnt(ddp + (size_t)wrow * n_super + (sb >> 3));
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
        const unsigned int xtok = tok0 + lr;
        if (xtok < p_rows) {
            sX[lr][lk] = xq[(size_t)xtok * n_sub + sb];
        } else {
            sX[lr][lk].d = 0.0f;
            sX[lr][lk].xsum = 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; kk++) {
            uint4 wq[TM]; uint32_t wqh[TM];
            float dsc[TM], deff[TM];
            #pragma unroll
            for (int r = 0; r < TM; r++) {
                wq[r]  = sW  [ty + r * 16][kk];
                wqh[r] = sWqh[ty + r * 16][kk];
                const float2 s = sWs[ty + r * 16][kk];
                dsc[r]  = s.x;
                deff[r] = s.y;
            }
            #pragma unroll
            for (int n = 0; n < TN; n++) {
                const BlockQ8* xb   = &sX[tx + n * 16][kk];
                const int*     xq32 = reinterpret_cast<const int*>(xb->qs);
                const float    dx   = xb->d;
                const float    xsum = xb->xsum;
                #pragma unroll
                for (int r = 0; r < TM; r++) {
                    const uint32_t qa[4] = { wq[r].x, wq[r].y, wq[r].z, wq[r].w };
                    const uint32_t qh = wqh[r];
                    int idot = 0;
                    #pragma unroll
                    for (int j = 0; j < 4; j++) {
                        const uint32_t lo = ( qa[j]       & 0x0F0F0F0Fu)
                            | spread4((qh >> (8 * j))     & 0xFu);
                        const uint32_t hi = ((qa[j] >> 4) & 0x0F0F0F0Fu)
                            | spread4((qh >> (8 * j + 4)) & 0xFu);
                        idot = __builtin_amdgcn_sdot4((int)lo, xq32[j],     idot, false);
                        idot = __builtin_amdgcn_sdot4((int)hi, xq32[j + 4], idot, false);
                    }
                    acc[r][n] += dsc[r] * dx * (float)idot - deff[r] * xsum;
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int r = 0; r < TM; r++) {
        const unsigned int row = row0 + ty + r * 16;
        if (row >= out_dim) continue;
        #pragma unroll
        for (int n = 0; n < TN; n++) {
            const unsigned int tok = tok0 + tx + n * 16;
            if (tok < p_rows) y[(size_t)tok * out_dim + row] = acc[r][n];
        }
    }
}
