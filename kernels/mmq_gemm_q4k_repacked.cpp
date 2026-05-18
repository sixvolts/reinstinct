// int8 MMQ GEMM — repacked Q4_K weights, dp4a. The P>1 generalisation of
// matvec_q4k_repacked: Y[P, out_dim] = Xq8[P, in_dim] · Wᵀ, with the
// weight consumed straight from the quantised repacked layout (no
// dequant to fp16).
//
// Decomposition: lane m owns one weight row (= one output column); it
// loops the full contraction itself (no cross-lane reduction) and
// accumulates a TN-wide tile of token columns. Each weight sub-block is
// read once per lane and reused across the TN tokens in the tile —
// that token-tile reuse is what a matvec (P=1) cannot exploit.
//
// grid = (ceil(out_dim/256), ceil(P/TN)); block = 256 (4 wavefronts).
// This is the correctness-first version; tile sizes / LDS staging are
// left to the tuning pass.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#define TN 32        // token columns per workgroup tile (compile-time:
                     // acc[TN] must live in registers)

struct __attribute__((packed)) BlockQ8 {
    float  d;
    float  xsum;
    int8_t qs[32];
};
static_assert(sizeof(BlockQ8) == 40, "BlockQ8 must be 40 bytes");

extern "C" __global__
void mmq_gemm_q4k_repacked_f32(const unsigned char* __restrict__ w,
                               const BlockQ8*       __restrict__ x,
                               float*               __restrict__ y,
                               unsigned int in_dim,
                               unsigned int out_dim,
                               unsigned int p_rows)
{
    const int wave = threadIdx.x >> 6;
    const int lane = threadIdx.x & 63;
    const unsigned int row = blockIdx.x * 256u + wave * 64u + lane;  // weight row
    const unsigned int n0  = blockIdx.y * TN;                        // first token
    if (row >= out_dim || n0 >= p_rows) return;

    const unsigned int n_sub = in_dim >> 5;
    const unsigned int nsp = ((n_sub & (n_sub - 1u)) == 0u) ? (n_sub + 1u) : n_sub;
    const uint4*    nib = reinterpret_cast<const uint4*>(w);
    const uint32_t* scl = reinterpret_cast<const uint32_t*>(
        w + (size_t)out_dim * nsp * 16);

    const unsigned int tn = (p_rows - n0 < TN) ? (p_rows - n0) : TN;

    float acc[TN];
    #pragma unroll
    for (int n = 0; n < TN; n++) acc[n] = 0.0f;

    for (unsigned int sb = 0; sb < n_sub; sb++) {
        const uint4    q = nib[(size_t)row * nsp + sb];
        const uint32_t s = scl[(size_t)row * nsp + sb];
        const uint16_t dsc_bits  = (uint16_t)(s & 0xFFFF);
        const uint16_t deff_bits = (uint16_t)(s >> 16);
        const float dsc  = __half2float(*reinterpret_cast<const __half*>(&dsc_bits));
        const float deff = __half2float(*reinterpret_cast<const __half*>(&deff_bits));

        // Unpack the weight sub-block once; reused across the TN tokens.
        const uint32_t qa[4] = { q.x, q.y, q.z, q.w };
        uint32_t lo[4], hi[4];
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            lo[j] =  qa[j]       & 0x0F0F0F0Fu;
            hi[j] = (qa[j] >> 4) & 0x0F0F0F0Fu;
        }

        for (unsigned int n = 0; n < tn; n++) {
            const BlockQ8* xb   = x + (size_t)(n0 + n) * n_sub + sb;
            const int*     xq32 = reinterpret_cast<const int*>(xb->qs);
            int idot = 0;
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                idot = __builtin_amdgcn_sdot4((int)lo[j], xq32[j],     idot, false);
                idot = __builtin_amdgcn_sdot4((int)hi[j], xq32[j + 4], idot, false);
            }
            acc[n] += dsc * xb->d * (float)idot - deff * xb->xsum;
        }
    }

    for (unsigned int n = 0; n < tn; n++)
        y[(size_t)(n0 + n) * out_dim + row] = acc[n];
}
