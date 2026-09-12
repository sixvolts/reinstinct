// IQ4_XS MMQ GEMM — derived from the Q4_0 kernel of the same name.
//
// The repacked layout is byte-identical to Q4_0's (quant::iq4_xs::
// repack_for_matvec): a 16-byte nibble plane per 32-weight sub-block and
// an fp16 scale plane, here with the super-block d and 6-bit sub-scale
// pre-folded into `dl = d * (ls - 32)`. Two differences from Q4_0:
//   * each nibble maps through the 16-entry IQ4_NL codebook before the
//     dot, instead of the linear `n - 8`;
//   * the codebook values are already signed, so there is no constant
//     offset and no `8 * xqsum` term.
// Everything else — tiling, launch contract, entry points — is Q4_0's.
//
// Layout:
//   slab + 0                    : out_dim × nsp × 16  nibble bytes
//   slab + out_dim × nsp × 16   : out_dim × nsp × 2   fp16 dl
//   nsp = (n_sub power of two) ? n_sub + 1 : n_sub   // anti-alias

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

// Tile shape carried over from the Q4_K kernel, whose sweep (BK ∈ {4,8},
// TM/TN ∈ {4,8}, occupancy 1/2) found 4×4 at occupancy 2 flat-optimal.
// The inner loop here is the same shape with strictly fewer registers,
// so the same point should hold; worth re-sweeping if IQ4_XS becomes the
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

// IQ4_NL / IQ4_XS codebook — the 16 int8 values a nibble indexes, packed
// four per 32-bit word (little-endian): entries 0-3, 4-7, 8-11, 12-15.
//   {-127,-104,-83,-65,-49,-35,-22,-10, 1,13,25,38,53,69,89,113}
// Overridable: IQ3_S repacks into this same layout with the codebook
// {-15,-13,...,15} and compiles this source with the four words
// predefined (quant::iq3_s::kernel_source).
#ifndef IQ4NL_KV_0_3
#define IQ4NL_KV_0_3   0xbfad9881u
#define IQ4NL_KV_4_7   0xf6eaddcfu
#define IQ4NL_KV_8_11  0x26190d01u
#define IQ4NL_KV_12_15 0x71594535u
#endif

// Map 4 nibbles (one per byte of `n4`, masked to 0x0F0F0F0F) to 4 int8
// codebook values packed for sdot4 — in registers, via v_perm_b32.
//
// A scalar table lookup here was catastrophic: 8 constant-memory byte
// loads per sdot4 inside the compute-bound MMQ loop made prefill 5.8x
// slower than the Q8_0 transcode it was meant to beat. v_perm selects
// bytes from an 8-byte pair per selector byte, so the low 3 bits of each
// nibble pick within a half-table, and bit 3 chooses which half.
__device__ __forceinline__ int iq4xs_lut4(uint32_t n4) {
    const uint32_t sel = n4 & 0x07070707u;
    // perm(a, b, sel): selector 0-3 -> bytes of b, 4-7 -> bytes of a.
    const uint32_t lo = __builtin_amdgcn_perm(IQ4NL_KV_4_7,   IQ4NL_KV_0_3,  sel);
    const uint32_t hi = __builtin_amdgcn_perm(IQ4NL_KV_12_15, IQ4NL_KV_8_11, sel);
    const uint32_t m  = ((n4 >> 3) & 0x01010101u) * 0xFFu;   // 0xFF where nibble >= 8
    return (int)((hi & m) | (lo & ~m));
}

struct __attribute__((packed)) BlockQ8 {
    float  d;
    float  xsum;
    int8_t qs[32];
};
static_assert(sizeof(BlockQ8) == 40, "BlockQ8 must be 40 bytes");

template<int TM, int TN, int BK, int TXG, int THREADS>
__device__ __forceinline__
void mmq_iq4xs_impl(const unsigned char* __restrict__ wbase,
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

    // Codebook applied once at tile-load time: each 16-byte nibble
    // sub-block expands to 32 int8 codebook values (two uint4: weights
    // 0-15 from the low nibbles, 16-31 from the high), so the inner loop
    // is the plain int8 dot of the Q8_0 kernel. Doing the lookup here
    // costs one lut per thread per tile instead of TM*TN*BK*8 in the
    // loop, and the extra LDS (BM*BK*16 bytes) is well inside budget.
    __shared__ uint4   sW[BM][BK][2];    // int8 codebook values
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
                const uint4 q = nib[(size_t)wrow * nsp + sb];
                sW[lr][lk][0] = make_uint4(
                    (uint32_t)iq4xs_lut4(q.x & 0x0F0F0F0Fu), (uint32_t)iq4xs_lut4(q.y & 0x0F0F0F0Fu),
                    (uint32_t)iq4xs_lut4(q.z & 0x0F0F0F0Fu), (uint32_t)iq4xs_lut4(q.w & 0x0F0F0F0Fu));
                sW[lr][lk][1] = make_uint4(
                    (uint32_t)iq4xs_lut4((q.x >> 4) & 0x0F0F0F0Fu), (uint32_t)iq4xs_lut4((q.y >> 4) & 0x0F0F0F0Fu),
                    (uint32_t)iq4xs_lut4((q.z >> 4) & 0x0F0F0F0Fu), (uint32_t)iq4xs_lut4((q.w >> 4) & 0x0F0F0F0Fu));
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
                wlo[r] = sW[ty + r * TYG][kk][0];
                whi[r] = sW[ty + r * TYG][kk][1];
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
void mmq_gemm_iq4xs_repacked_f32(const unsigned char* __restrict__ wbase,
                                const BlockQ8*       __restrict__ xq,
                                float*               __restrict__ y,
                                unsigned int in_dim,
                                unsigned int out_dim,
                                unsigned int p_rows)
{
    mmq_iq4xs_impl<4, 4, 4, 16, 256>(wbase, xq, y, in_dim, out_dim, p_rows);
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
void mmq_gemm_iq4xs_repacked_narrow_f32(const unsigned char* __restrict__ wbase,
                                const BlockQ8*       __restrict__ xq,
                                float*               __restrict__ y,
                                unsigned int in_dim,
                                unsigned int out_dim,
                                unsigned int p_rows)
{
    mmq_iq4xs_impl<NARROW_TM, NARROW_TN, NARROW_BK, NARROW_TXG, NARROW_THREADS>(wbase, xq, y, in_dim, out_dim, p_rows);
}
