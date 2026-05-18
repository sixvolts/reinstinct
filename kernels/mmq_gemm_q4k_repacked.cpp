// int8 MMQ GEMM — repacked Q4_K weights, dp4a. Y[P, out_dim] =
// Xq8[P, in_dim] · Wᵀ, consuming the quantised repacked weight directly
// (no dequant to fp16).
//
// Same decomposition as matvec_q4k_repacked — lane l sweeps sub-block l,
// l+64, … so consecutive lanes read consecutive 16-byte weight
// sub-blocks (and consecutive activation sub-blocks): both global reads
// are fully coalesced. Extended with an inner loop over TN token
// columns — each weight sub-block is unpacked once and reused across all
// TN tokens, the reuse a matvec (P=1) cannot exploit.
//
// TN is compile-time so the unrolled token loop indexes acc[r][n] with a
// constant n, keeping the accumulators in registers (a runtime bound
// would spill the whole acc array to scratch memory).
//
// 256-thread workgroup: 4 wavefronts, ROWS=2 rows each → 8 output rows.
// grid = (ceil(out_dim/8), ceil(P/TN)).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#define ROWS 2
#define TN   32        // token columns per workgroup tile

struct __attribute__((packed)) BlockQ8 {
    float  d;
    float  xsum;
    int8_t qs[32];
};
static_assert(sizeof(BlockQ8) == 40, "BlockQ8 must be 40 bytes");

extern "C" __global__
void mmq_gemm_q4k_repacked_f32(const unsigned char* __restrict__ wbase,
                               const BlockQ8*       __restrict__ xq,
                               float*               __restrict__ y,
                               unsigned int in_dim,
                               unsigned int out_dim,
                               unsigned int p_rows)
{
    const int wave = threadIdx.x >> 6;
    const int lane = threadIdx.x & 63;
    const int row0 = blockIdx.x * (ROWS * 4) + wave * ROWS;
    const unsigned int n0 = blockIdx.y * TN;
    if (n0 >= p_rows) return;

    const unsigned int n_sub = in_dim >> 5;
    const unsigned int nsp = ((n_sub & (n_sub - 1u)) == 0u) ? (n_sub + 1u) : n_sub;
    const uint4*    nib = reinterpret_cast<const uint4*>(wbase);
    const uint32_t* scl = reinterpret_cast<const uint32_t*>(
        wbase + (size_t)out_dim * nsp * 16);

    float acc[ROWS][TN];
    #pragma unroll
    for (int r = 0; r < ROWS; r++)
        #pragma unroll
        for (int n = 0; n < TN; n++) acc[r][n] = 0.0f;

    for (unsigned int sb = lane; sb < n_sub; sb += 64) {
        // Weight sub-block for each of the ROWS rows — unpacked once,
        // reused across all TN tokens.
        uint32_t lo[ROWS][4], hi[ROWS][4];
        float dsc[ROWS], deff[ROWS];
        #pragma unroll
        for (int r = 0; r < ROWS; r++) {
            const int row = row0 + r;
            if (row >= (int)out_dim) { dsc[r] = deff[r] = 0.0f; continue; }
            const uint4    q = nib[(size_t)row * nsp + sb];
            const uint32_t s = scl[(size_t)row * nsp + sb];
            const uint16_t dsc_bits  = (uint16_t)(s & 0xFFFF);
            const uint16_t deff_bits = (uint16_t)(s >> 16);
            dsc[r]  = __half2float(*reinterpret_cast<const __half*>(&dsc_bits));
            deff[r] = __half2float(*reinterpret_cast<const __half*>(&deff_bits));
            const uint32_t qa[4] = { q.x, q.y, q.z, q.w };
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                lo[r][j] =  qa[j]       & 0x0F0F0F0Fu;
                hi[r][j] = (qa[j] >> 4) & 0x0F0F0F0Fu;
            }
        }

        #pragma unroll
        for (int n = 0; n < TN; n++) {
            if (n0 + (unsigned)n >= p_rows) continue;
            const BlockQ8* xb   = xq + (size_t)(n0 + n) * n_sub + sb;
            const int*     xq32 = reinterpret_cast<const int*>(xb->qs);
            const float    dx   = xb->d;
            const float    xsum = xb->xsum;
            #pragma unroll
            for (int r = 0; r < ROWS; r++) {
                int idot = 0;
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    idot = __builtin_amdgcn_sdot4((int)lo[r][j], xq32[j],     idot, false);
                    idot = __builtin_amdgcn_sdot4((int)hi[r][j], xq32[j + 4], idot, false);
                }
                acc[r][n] += dsc[r] * dx * (float)idot - deff[r] * xsum;
            }
        }
    }

    // Reduce each (row, token) partial sum across the 64 lanes.
    #pragma unroll
    for (int r = 0; r < ROWS; r++) {
        const int row = row0 + r;
        #pragma unroll
        for (int n = 0; n < TN; n++) {
            if (n0 + (unsigned)n >= p_rows) continue;
            float a = acc[r][n];
            a += __shfl_xor(a, 32);
            a += __shfl_xor(a, 16);
            a += __shfl_xor(a,  8);
            a += __shfl_xor(a,  4);
            a += __shfl_xor(a,  2);
            a += __shfl_xor(a,  1);
            if (lane == 0 && row < (int)out_dim)
                y[(size_t)(n0 + n) * out_dim + row] = a;
        }
    }
}
