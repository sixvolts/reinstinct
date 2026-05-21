// int8 MMQ GEMM — repacked Q8_0 weights, dp4a, 2D-tiled with LDS
// staging. Y[P, out_dim] = Xq8[P, in_dim] · Wᵀ, consuming the quantised
// repacked weight directly (no dequant to fp16). Mirrors the Q4_K
// kernel's tile shape and dispatch; the per-sub-block math is simpler
// (no nibble unpack, no m/dmin term — Q8_0 stores signed int8 quants
// with a single fp16 scale per 32-element sub-block).
//
// Repacked layout (see src/quant/q8_0.rs::repack_for_matvec):
//   * qs plane: out_dim * nsp * 32 bytes (raw int8 quants).
//   * d  plane: out_dim * nsp * 2  bytes (fp16 scales).
// The two planes are concatenated in `wbase`.
//
// grid = (ceil(out_dim/BM), ceil(P/BN)).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#define BK 4         // sub-blocks per contraction chunk
#define TM 4         // rows  per thread micro-tile
#define TN 4         // tokens per thread micro-tile
#define BM (16 * TM) // weight rows per workgroup tile
#define BN (16 * TN) // tokens per workgroup tile
#define LDW (BM * BK / 256)   // weight tile elems loaded per thread
#define LDX (BN * BK / 256)   // activation tile elems loaded per thread

struct __attribute__((packed)) BlockQ8 {
    float  d;
    float  xsum;
    int8_t qs[32];
};
static_assert(sizeof(BlockQ8) == 40, "BlockQ8 must be 40 bytes");

extern "C" __global__ __launch_bounds__(256, 2)
void mmq_gemm_q8_0_repacked_f32(const unsigned char* __restrict__ wbase,
                                const BlockQ8*       __restrict__ xq,
                                float*               __restrict__ y,
                                unsigned int in_dim,
                                unsigned int out_dim,
                                unsigned int p_rows)
{
    const int t  = threadIdx.x;          // 0..255
    const int tx = t & 15;               // token group  0..15
    const int ty = t >> 4;               // row group    0..15
    const unsigned int row0 = blockIdx.x * BM;
    const unsigned int tok0 = blockIdx.y * BN;

    const unsigned int n_sub = in_dim >> 5;
    const unsigned int nsp = ((n_sub & (n_sub - 1u)) == 0u) ? (n_sub + 1u) : n_sub;
    // qs plane: int8 quants laid out [out_dim, nsp, 32]. Read as uint4.
    const uint4* qsp = reinterpret_cast<const uint4*>(wbase);
    // d plane: fp16 scales [out_dim, nsp].
    const uint16_t* dp = reinterpret_cast<const uint16_t*>(
        wbase + (size_t)out_dim * nsp * 32);

    // Q8_0 sub-block = 32 bytes (32 signed int8s). uint4 is 16 bytes, so
    // we store two uint4s per sub-block in LDS — `sW_lo` for the first
    // 16 bytes, `sW_hi` for the last 16.
    __shared__ uint4   sW_lo[BM][BK];
    __shared__ uint4   sW_hi[BM][BK];
    __shared__ float   sWd[BM][BK];      // per-sub-block fp16 scale, dequant'd
    __shared__ BlockQ8 sX[BN][BK];       // int8 acts

    float acc[TM][TN];
    #pragma unroll
    for (int r = 0; r < TM; r++)
        #pragma unroll
        for (int n = 0; n < TN; n++) acc[r][n] = 0.0f;

    for (unsigned int sb0 = 0; sb0 < n_sub; sb0 += BK) {
        // Cooperative load — consecutive threads → consecutive (row, sb).
        #pragma unroll
        for (int i = 0; i < LDW; i++) {
            const int e  = t + i * 256;
            const int lr = e / BK, lk = e % BK;
            const unsigned int wrow = row0 + lr;
            if (wrow < out_dim) {
                const unsigned int sb = sb0 + lk;
                // Two uint4 per sub-block at offsets (row*nsp + sb)*2 and *2+1.
                sW_lo[lr][lk] = qsp[(size_t)(wrow * nsp + sb) * 2];
                sW_hi[lr][lk] = qsp[(size_t)(wrow * nsp + sb) * 2 + 1];
                const uint16_t d_bits = dp[(size_t)wrow * nsp + sb];
                sWd[lr][lk] = __half2float(*reinterpret_cast<const __half*>(&d_bits));
            } else {
                sWd[lr][lk] = 0.0f;
            }
        }
        #pragma unroll
        for (int i = 0; i < LDX; i++) {
            const int e  = t + i * 256;
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
                wq_lo[r] = sW_lo[ty + r * 16][kk];
                wq_hi[r] = sW_hi[ty + r * 16][kk];
                dsc[r]   = sWd[ty + r * 16][kk];
            }
            #pragma unroll
            for (int n = 0; n < TN; n++) {
                const BlockQ8* xb   = &sX[tx + n * 16][kk];
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
        const unsigned int row = row0 + ty + r * 16;
        if (row >= out_dim) continue;
        #pragma unroll
        for (int n = 0; n < TN; n++) {
            const unsigned int tok = tok0 + tx + n * 16;
            if (tok < p_rows) y[(size_t)tok * out_dim + row] = acc[r][n];
        }
    }
}
