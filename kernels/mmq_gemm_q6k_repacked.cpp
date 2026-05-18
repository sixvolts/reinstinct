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

// Spread 2-bit groups (b0b1, b2b3, …) to bits 4-5 of bytes 0..3.
__device__ __forceinline__ uint32_t spread2(uint32_t h) {
    return ((h & 0x03u) << 4) | ((h & 0x0Cu) << 10)
         | ((h & 0x30u) << 16) | ((h & 0xC0u) << 22);
}

extern "C" __global__ __launch_bounds__(256, 2)
void mmq_gemm_q6k_repacked_f32(const unsigned char* __restrict__ wbase,
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
    __shared__ BlockQ8  sX  [BN][BK];    // int8 acts

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
            uint4 wq[TM]; uint2 wh2[TM];
            float dlo[TM], dhi[TM];
            #pragma unroll
            for (int r = 0; r < TM; r++) {
                wq[r]  = sW  [ty + r * 16][kk];
                wh2[r] = sWh2[ty + r * 16][kk];
                const float2 s = sWs[ty + r * 16][kk];
                dlo[r] = s.x;
                dhi[r] = s.y;
            }
            #pragma unroll
            for (int n = 0; n < TN; n++) {
                const BlockQ8* xb   = &sX[tx + n * 16][kk];
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
        const unsigned int row = row0 + ty + r * 16;
        if (row >= out_dim) continue;
        #pragma unroll
        for (int n = 0; n < TN; n++) {
            const unsigned int tok = tok0 + tx + n * 16;
            if (tok < p_rows) y[(size_t)tok * out_dim + row] = acc[r][n];
        }
    }
}
