// int8 MMQ GEMM — repacked Q4_0 weights, dp4a, 2D-tiled with LDS
// staging. Y[P, out_dim] = Xq8[P, in_dim] · Wᵀ, consuming the quantised
// repacked weight directly (no dequant to fp16).
//
// Structurally identical to mmq_gemm_q4k_repacked — same 256-thread
// 16×16 grid, same BM×BN output tile, same strided micro-tile so a
// wavefront's 16 token reads hit 16 distinct LDS banks. The difference
// is the scale plane: Q4_0 carries one fp16 per 32-weight block and a
// constant −8 offset, so where Q4_K stages a (dsc, deff) float2 per
// (row, sub-block) built from a 6-bit scale/min pair plus a superblock
// (d, dmin), this stages a single float and folds the constant −8 into
// the integer accumulator:
//
//   acc += dw · dx · (<nibbles · int8 acts> − 8 · <sum of int8 acts>)
//
// The offset is taken against the summed *quantized* activations, not
// BlockQ8::xsum — see matvec_q4_0_repacked.cpp for why mixing the two
// domains costs 40× accuracy.
//
// That halves the scale LDS and drops the superblock plane read
// entirely; the nibble traffic — the term that actually bounds this
// kernel — is unchanged.
//
// grid = (ceil(out_dim/BM), ceil(P/BN)).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

// Tile shape carried over from the Q4_K kernel, whose sweep (BK ∈ {4,8},
// TM/TN ∈ {4,8}, occupancy 1/2) found 4×4 at occupancy 2 flat-optimal.
// The inner loop here is the same shape with strictly fewer registers,
// so the same point should hold; worth re-sweeping if Q4_0 becomes the
// primary prefill format.
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

// The weight tile is expanded to int8 at LDS-load time — two uint4 per
// 32-weight sub-block (weights 0-15, then 16-31) in the order the
// activation's dp4a groups use — so the inner loop is the plain int8
// dot of the Q8_0 kernel. Unpacking there instead cost TM*TN*BK*8
// unpacks per tile against one per thread here, and the register
// pressure of the packed form spilled (5-49 VGPRs across the K-quants;
// Q6_K ran at 5.7 TOPS against Q8_0's 19.7 at the 27B FFN shape).
// Bytewise `v - 8` on four nibbles without inter-byte borrow (set bit 7,
// subtract, clear bit 7) — the signed int8 Q4_0 weight sdot4 wants.
__device__ __forceinline__ uint32_t sub8(uint32_t v) {
    return ((v | 0x80808080u) - 0x08080808u) ^ 0x80808080u;
}

template<int TM, int TN, int BK, int TXG, int THREADS>
__device__ __forceinline__
void mmq_q4_0_impl(const unsigned char* __restrict__ wbase,
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
    const uint4*    nib = reinterpret_cast<const uint4*>(wbase);
    const uint16_t* dp  = reinterpret_cast<const uint16_t*>(
        wbase + (size_t)out_dim * nsp * 16);

    __shared__ uint4   sW_lo[BM][BK];    // int8 (nibble - 8) for weights 0-15
    __shared__ uint4   sW_hi[BM][BK];    // int8 (nibble - 8) for weights 16-31
    __shared__ float   sWd[BM][BK];      // per-block fp16 scale, widened
    __shared__ BlockQ8 sX[BN][BK + 1];   // int8 acts

    float acc[TM][TN];
    #pragma unroll
    for (int r = 0; r < TM; r++)
        #pragma unroll
        for (int n = 0; n < TN; n++) acc[r][n] = 0.0f;

    for (unsigned int sb0 = 0; sb0 < n_sub; sb0 += BK) {
        // Cooperative load — consecutive threads → consecutive (row,sb).
        for (int e = t; e < BM * BK; e += THREADS) {
            const int lr = e / BK, lk = e % BK;
            const unsigned int wrow = row0 + lr;
            if (wrow < out_dim) {
                const unsigned int sb = sb0 + lk;
                // The -8 offset is folded in bytewise here (sub8), so the
                // loop needs no quantised-activation-sum term.
                const uint4 q = nib[(size_t)wrow * nsp + sb];
                const uint32_t M = 0x0F0F0F0Fu;
                sW_lo[lr][lk] = make_uint4(sub8(q.x & M), sub8(q.y & M), sub8(q.z & M), sub8(q.w & M));
                sW_hi[lr][lk] = make_uint4(sub8((q.x >> 4) & M), sub8((q.y >> 4) & M),
                                           sub8((q.z >> 4) & M), sub8((q.w >> 4) & M));
                const uint16_t db = dp[(size_t)wrow * nsp + sb];
                sWd[lr][lk] = __half2float(*reinterpret_cast<const __half*>(&db));
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
            uint4 wlo[TM], whi[TM];
            float dw[TM];
            #pragma unroll
            for (int r = 0; r < TM; r++) {
                wlo[r] = sW_lo[ty + r * TYG][kk];
                whi[r] = sW_hi[ty + r * TYG][kk];
                dw[r]  = sWd[ty + r * TYG][kk];
            }
            #pragma unroll
            for (int n = 0; n < TN; n++) {
                const BlockQ8* xb   = &sX[tx + n * TXG][kk];
                const int*     xq32 = reinterpret_cast<const int*>(xb->qs);
                const float    dx   = xb->d;
                #pragma unroll
                for (int r = 0; r < TM; r++) {
                    const int wa[8] = { (int)wlo[r].x, (int)wlo[r].y, (int)wlo[r].z, (int)wlo[r].w,
                                        (int)whi[r].x, (int)whi[r].y, (int)whi[r].z, (int)whi[r].w };
                    int idot = 0;
                    #pragma unroll
                    for (int j = 0; j < 8; j++)
                        idot = __builtin_amdgcn_sdot4(wa[j], xq32[j], idot, false);
                    acc[r][n] += dw[r] * dx * (float)idot;
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
void mmq_gemm_q4_0_repacked_f32(const unsigned char* __restrict__ wbase,
                                const BlockQ8*       __restrict__ xq,
                                float*               __restrict__ y,
                                unsigned int in_dim,
                                unsigned int out_dim,
                                unsigned int p_rows)
{
    mmq_q4_0_impl<4, 4, 4, 16, 256>(wbase, xq, y, in_dim, out_dim, p_rows);
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
void mmq_gemm_q4_0_repacked_narrow_f32(const unsigned char* __restrict__ wbase,
                                const BlockQ8*       __restrict__ xq,
                                float*               __restrict__ y,
                                unsigned int in_dim,
                                unsigned int out_dim,
                                unsigned int p_rows)
{
    mmq_q4_0_impl<NARROW_TM, NARROW_TN, NARROW_BK, NARROW_TXG, NARROW_THREADS>(wbase, xq, y, in_dim, out_dim, p_rows);
}
