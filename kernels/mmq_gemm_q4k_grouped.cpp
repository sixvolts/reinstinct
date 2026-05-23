// Grouped-expert MMQ GEMM — repacked Q4_K.
//
// One launch covers every expert. A workgroup is (blockIdx.x = out-row
// tile, blockIdx.y = global token tile). The global token-tile index is
// mapped to its (expert, local tile) by a binary search over `tile_off`
// — the prefix sum of ceil(tokens_per_expert / BN). The activation `xq`
// and output `y` are in expert-contiguous (sorted) order: expert e owns
// rows expert_off[e] .. expert_off[e+1]. The expert's weight slab slice
// is `slab + e * bytes_per_expert`, repacked-Q4_K like the dense MMQ.
//
// grid = (ceil(out_dim/BM), tile_upper_bound); workgroups whose tile
// index is past the real total (tile_off[n_expert]) early-exit.
//
// BN=32: MoE routing spreads tokens thin (256-expert models see ~16
// tokens/expert at P=512), so a 64-wide token tile would be ~75% empty
// — and the MMQ tile cost is dp4a-bound and occupancy-independent, so
// empty columns are wasted work. BN=32 halves the waste. The
// cooperative loads use a `for e=t; e<N; e+=256` form so they stay
// correct for BN<64 (BN*BK can be < 256).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#define BK 4
#define TM 4
#define TN 2
#define BM (16 * TM)   // 64 weight rows / workgroup
#define BN (16 * TN)   // 32 tokens / workgroup

struct __attribute__((packed)) BlockQ8 {
    float  d;
    float  xsum;
    int8_t qs[32];
};
static_assert(sizeof(BlockQ8) == 40, "BlockQ8 must be 40 bytes");

extern "C" __global__ __launch_bounds__(256, 2)
void mmq_gemm_q4k_grouped_f32(const unsigned char* __restrict__ slab,
                              unsigned int bytes_per_expert,
                              const int*  __restrict__ expert_off,
                              const int*  __restrict__ tile_off,
                              unsigned int n_expert,
                              const BlockQ8* __restrict__ xq,
                              float*        __restrict__ y,
                              unsigned int in_dim,
                              unsigned int out_dim)
{
    // --- map workgroup → (expert, token tile) ---
    const unsigned int by = blockIdx.y;
    if (by >= (unsigned int)tile_off[n_expert]) return;   // over-launched
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
    const uint16_t* smp = reinterpret_cast<const uint16_t*>(
        wbase + (size_t)out_dim * nsp * 16);
    const uint32_t* ddp = reinterpret_cast<const uint32_t*>(
        wbase + (size_t)out_dim * nsp * 16 + (size_t)out_dim * nsp * 2);

    __shared__ uint4   sW[BM][BK];
    __shared__ float2  sWs[BM][BK];
    __shared__ BlockQ8 sX[BN][BK + 1];

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
            uint4 wq[TM];
            float dsc[TM], deff[TM];
            #pragma unroll
            for (int r = 0; r < TM; r++) {
                wq[r] = sW[ty + r * 16][kk];
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
        const unsigned int row = row0 + ty + r * 16;
        if (row >= out_dim) continue;
        #pragma unroll
        for (int n = 0; n < TN; n++) {
            const unsigned int tok = tok0 + tx + n * 16;
            if (tok < tok_end) y[(size_t)tok * out_dim + row] = acc[r][n];
        }
    }
}
