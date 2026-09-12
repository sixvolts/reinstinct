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

// Bytewise `v - 32` on four 6-bit values without inter-byte borrow:
// setting bit 7 first makes every byte >= 128 > 32, and clearing it
// afterwards leaves exactly (v - 32) mod 256 — the two's-complement
// int8 sdot4 wants.
__device__ __forceinline__ uint32_t sub32(uint32_t v) {
    return ((v | 0x80808080u) - 0x20202020u) ^ 0x80808080u;
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

    __shared__ uint4    sW_lo[BM][BK];   // int8 (q6 - 32) for weights 0-15
    __shared__ uint4    sW_hi[BM][BK];   // int8 (q6 - 32) for weights 16-31
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
                const uint4 q  = nib[(size_t)wrow * nsp + sb];
                const uint2 h2 = h2p[(size_t)wrow * nsp + sb];
                const uint32_t M = 0x0F0F0F0Fu;
                // Byte g of the 8-byte plane holds dp4a group g's four
                // 2-bit high fields, and the nibble plane interleaves the
                // groups: word j's low nibbles are group 2j (activation
                // group j), its high nibbles group 2j+1 (activation group
                // j+4). The symmetric -32 is folded in here (sub32), so
                // the loop needs no activation-sum correction.
                sW_lo[lr][lk] = make_uint4(
                    sub32((q.x & M) | spread2( h2.x        & 0xFFu)),   // group 0
                    sub32((q.y & M) | spread2((h2.x >> 16) & 0xFFu)),   // group 2
                    sub32((q.z & M) | spread2( h2.y        & 0xFFu)),   // group 4
                    sub32((q.w & M) | spread2((h2.y >> 16) & 0xFFu)));  // group 6
                sW_hi[lr][lk] = make_uint4(
                    sub32(((q.x >> 4) & M) | spread2((h2.x >>  8) & 0xFFu)),   // group 1
                    sub32(((q.y >> 4) & M) | spread2((h2.x >> 24) & 0xFFu)),   // group 3
                    sub32(((q.z >> 4) & M) | spread2((h2.y >>  8) & 0xFFu)),   // group 5
                    sub32(((q.w >> 4) & M) | spread2((h2.y >> 24) & 0xFFu)));  // group 7
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
            uint4 wlo[TM], whi[TM];
            float dlo[TM], dhi[TM];
            #pragma unroll
            for (int r = 0; r < TM; r++) {
                wlo[r] = sW_lo[ty + r * TYG][kk];
                whi[r] = sW_hi[ty + r * TYG][kk];
                const float2 s = sWs[ty + r * TYG][kk];
                dlo[r] = s.x;
                dhi[r] = s.y;
            }
            #pragma unroll
            for (int n = 0; n < TN; n++) {
                const BlockQ8* xb   = &sX[tx + n * TXG][kk];
                const int*     xq32 = reinterpret_cast<const int*>(xb->qs);
                const float    dx   = xb->d;
                #pragma unroll
                for (int r = 0; r < TM; r++) {
                    const int lo[4] = { (int)wlo[r].x, (int)wlo[r].y, (int)wlo[r].z, (int)wlo[r].w };
                    const int hi[4] = { (int)whi[r].x, (int)whi[r].y, (int)whi[r].z, (int)whi[r].w };
                    int idot0 = 0, idot1 = 0;
                    #pragma unroll
                    for (int j = 0; j < 4; j++) {
                        idot0 = __builtin_amdgcn_sdot4(lo[j], xq32[j],     idot0, false);
                        idot1 = __builtin_amdgcn_sdot4(hi[j], xq32[j + 4], idot1, false);
                    }
                    acc[r][n] += dlo[r] * dx * (float)idot0 + dhi[r] * dx * (float)idot1;
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
