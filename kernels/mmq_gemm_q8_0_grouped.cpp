// Grouped-expert MMQ GEMM — repacked Q8_0. See mmq_gemm_q4k_grouped.cpp
// for the design. Q8_0 dense MMQ body (a 32-byte sub-block = two uint4
// in LDS, no m/dmin term) with the workgroup->expert prologue, BN=32,
// BN-agnostic loads.
//
// Repacked Q8_0 layout: qs plane = out_dim*nsp*32 bytes, then a
// d plane = out_dim*nsp*2 fp16 scales.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#define BK 4
#define TM 4
#define TN 2
#define BM (16 * TM)   // 64
#define BN (16 * TN)   // 32

struct __attribute__((packed)) BlockQ8 {
    float  d;
    float  xsum;
    int8_t qs[32];
};
static_assert(sizeof(BlockQ8) == 40, "BlockQ8 must be 40 bytes");

extern "C" __global__ __launch_bounds__(256, 2)
void mmq_gemm_q8_0_grouped_f32(const unsigned char* __restrict__ slab,
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
    const uint4*    qsp = reinterpret_cast<const uint4*>(wbase);
    const uint16_t* dp  = reinterpret_cast<const uint16_t*>(
        wbase + (size_t)out_dim * nsp * 32);

    __shared__ uint4   sW_lo[BM][BK];
    __shared__ uint4   sW_hi[BM][BK];
    __shared__ float   sWd  [BM][BK];
    __shared__ BlockQ8 sX   [BN][BK];

    float acc[TM][TN];
    #pragma unroll
    for (int r = 0; r < TM; r++)
        #pragma unroll
        for (int n = 0; n < TN; n++) acc[r][n] = 0.0f;

    for (unsigned int sb0 = 0; sb0 < n_sub; sb0 += BK) {
        for (int e2 = t; e2 < BM * BK; e2 += 256) {
            const int lr = e2 / BK, lk = e2 % BK;
            const unsigned int wrow = row0 + lr;
            if (wrow < out_dim) {
                const unsigned int sb = sb0 + lk;
                sW_lo[lr][lk] = qsp[(size_t)(wrow * nsp + sb) * 2];
                sW_hi[lr][lk] = qsp[(size_t)(wrow * nsp + sb) * 2 + 1];
                const uint16_t d_bits = dp[(size_t)wrow * nsp + sb];
                sWd[lr][lk] = __half2float(*reinterpret_cast<const __half*>(&d_bits));
            } else {
                sWd[lr][lk] = 0.0f;
            }
        }
        for (int e2 = t; e2 < BN * BK; e2 += 256) {
            const int lr = e2 / BK, lk = e2 % BK;
            const unsigned int xtok = tok0 + lr;
            if (xtok < tok_end) {
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
                wq_lo[r] = sW_lo[ty + r * 16][kk];
                wq_hi[r] = sW_hi[ty + r * 16][kk];
                dsc[r]   = sWd[ty + r * 16][kk];
            }
            #pragma unroll
            for (int n = 0; n < TN; n++) {
                const BlockQ8* xb   = &sX[tx + n * 16][kk];
                const int*     xq32 = reinterpret_cast<const int*>(xb->qs);
                const float    dx   = xb->d;
                #pragma unroll
                for (int r = 0; r < TM; r++) {
                    const uint32_t lo2[4] = { wq_lo[r].x, wq_lo[r].y, wq_lo[r].z, wq_lo[r].w };
                    const uint32_t hi2[4] = { wq_hi[r].x, wq_hi[r].y, wq_hi[r].z, wq_hi[r].w };
                    int idot = 0;
                    #pragma unroll
                    for (int j = 0; j < 4; j++) {
                        idot = __builtin_amdgcn_sdot4((int)lo2[j], xq32[j],     idot, false);
                        idot = __builtin_amdgcn_sdot4((int)hi2[j], xq32[j + 4], idot, false);
                    }
                    acc[r][n] += dsc[r] * dx * (float)idot;
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
