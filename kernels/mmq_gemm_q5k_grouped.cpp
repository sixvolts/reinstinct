// Grouped-expert MMQ GEMM — repacked Q5_K. See mmq_gemm_q4k_grouped.cpp
// for the design: one launch, a binary-search workgroup->expert
// prologue over tile_off, expert-sorted xq / y. This is the Q5_K dense
// MMQ body (extra qh plane = the 5th quant bit) with that prologue,
// BN=32, and BN-agnostic cooperative loads.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#define BK 4
#define TM 4
#define TN 1
#define BM (16 * TM)   // 64
#define BN (16 * TN)   // 32

struct __attribute__((packed)) BlockQ8 {
    float  d;
    float  xsum;
    int8_t qs[32];
};
static_assert(sizeof(BlockQ8) == 40, "BlockQ8 must be 40 bytes");

__device__ __forceinline__ uint32_t spread4(uint32_t h) {
    return ((h & 1u) << 4) | ((h & 2u) << 11) | ((h & 4u) << 18) | ((h & 8u) << 25);
}

extern "C" __global__ __launch_bounds__(256, 2)
void mmq_gemm_q5k_grouped_f32(const unsigned char* __restrict__ slab,
                              unsigned int bytes_per_expert,
                              const int*  __restrict__ expert_off,
                              const int*  __restrict__ tile_off,
                              unsigned int n_expert,
                              const BlockQ8* __restrict__ xq,
                              float*        __restrict__ y,
                              unsigned int in_dim,
                              unsigned int out_dim)
{
    const unsigned int by = blockIdx.y;
    if (by >= (unsigned int)tile_off[n_expert]) return;
    int lo = 0, hi = (int)n_expert;
    while (lo + 1 < hi) {
        int mid = (lo + hi) >> 1;
        if ((unsigned int)tile_off[mid] <= by) lo = mid; else hi = mid;
    }
    const unsigned int e          = (unsigned int)lo;
    const unsigned int local_tile = by - (unsigned int)tile_off[e];
    const unsigned int tok_base   = (unsigned int)expert_off[e] + local_tile * BN;
    const unsigned int tok_end    = (unsigned int)expert_off[e + 1];
    const unsigned char* wbase    = slab + (size_t)e * bytes_per_expert;

    const int t  = threadIdx.x;
    const int tx = t & 15;
    const int ty = t >> 4;
    const unsigned int row0 = blockIdx.x * BM;
    const unsigned int tok0 = tok_base;

    const unsigned int n_sub = in_dim >> 5;
    const unsigned int nsp = ((n_sub & (n_sub - 1u)) == 0u) ? (n_sub + 1u) : n_sub;
    const unsigned int n_super = n_sub >> 3;
    const uint4*    nib = reinterpret_cast<const uint4*>(wbase);
    const uint32_t* qhp = reinterpret_cast<const uint32_t*>(
        wbase + (size_t)out_dim * nsp * 16);
    const uint16_t* smp = reinterpret_cast<const uint16_t*>(
        wbase + (size_t)out_dim * nsp * 16 + (size_t)out_dim * nsp * 4);
    const uint32_t* ddp = reinterpret_cast<const uint32_t*>(
        wbase + (size_t)out_dim * nsp * 16 + (size_t)out_dim * nsp * 4
              + (size_t)out_dim * nsp * 2);

    __shared__ uint4    sW  [BM][BK];
    __shared__ uint32_t sWqh[BM][BK];
    __shared__ float2   sWs [BM][BK];
    __shared__ BlockQ8  sX  [BN][BK + 1];

    float acc[TM][TN];
    #pragma unroll
    for (int r = 0; r < TM; r++)
        #pragma unroll
        for (int n = 0; n < TN; n++) acc[r][n] = 0.0f;

    for (unsigned int sb0 = 0; sb0 < n_sub; sb0 += BK) {
        for (int e2 = t; e2 < BM * BK; e2 += 256) {
            const int lr = e2 / BK, lk = e2 % BK;
            const unsigned int wrow = row0 + lr;
            if (wrow < out_dim && sb0 + (unsigned int)lk < n_sub) {
                const unsigned int sb = sb0 + lk;
                sW  [lr][lk] = nib[(size_t)wrow * nsp + sb];
                sWqh[lr][lk] = qhp[(size_t)wrow * nsp + sb];
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
        for (int e2 = t; e2 < BN * BK; e2 += 256) {
            const int lr = e2 / BK, lk = e2 % BK;
            const unsigned int xtok = tok0 + lr;
            if (xtok < tok_end && sb0 + (unsigned int)lk < n_sub) {
                sX[lr][lk] = xq[(size_t)xtok * n_sub + sb0 + lk];
            } else {
                sX[lr][lk].d = 0.0f;
                sX[lr][lk].xsum = 0.0f;
            }
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
                        const uint32_t lo2 = ( qa[j]       & 0x0F0F0F0Fu)
                            | spread4((qh >> (8 * j))     & 0xFu);
                        const uint32_t hi2 = ((qa[j] >> 4) & 0x0F0F0F0Fu)
                            | spread4((qh >> (8 * j + 4)) & 0xFu);
                        idot = __builtin_amdgcn_sdot4((int)lo2, xq32[j],     idot, false);
                        idot = __builtin_amdgcn_sdot4((int)hi2, xq32[j + 4], idot, false);
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
            if (tok < tok_end) y[(size_t)tok * out_dim + row] = acc[r][n];
        }
    }
}
