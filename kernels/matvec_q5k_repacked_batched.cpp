// Q5_K matvec for K=2..8 activation rows. Same idea as
// matvec_q4k_repacked_batched but with Q5_K's extra qh (5th-bit)
// plane. See matvec_q4k_repacked_batched.cpp for the batching
// rationale and matvec_q5k_repacked.cpp for the layout details.
//
// 256-thread workgroup, 4 waves × ROWS=2 per WG. grid = ceil(out_dim/8).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#define ROWS         2
#define N_ROWS_MAX   4   // K upper bound; see matvec_q4k_repacked_batched.cpp

struct __attribute__((packed)) BlockQ8 {
    float  d;
    float  xsum;
    int8_t qs[32];
};
static_assert(sizeof(BlockQ8) == 40, "BlockQ8 must be 40 bytes");

__device__ __forceinline__ uint32_t spread4(uint32_t h) {
    return ((h & 1u) << 4) | ((h & 2u) << 11) | ((h & 4u) << 18) | ((h & 8u) << 25);
}

extern "C" __global__
void matvec_q5k_repacked_batched_f32(const uint8_t* __restrict__ wbase,
                                     const BlockQ8* __restrict__ xq,
                                     float*         __restrict__ y,
                                     unsigned int in_dim,
                                     unsigned int out_dim,
                                     unsigned int n_rows)
{
    const int wave = threadIdx.x >> 6;
    const int lane = threadIdx.x & 63;
    const int row0 = blockIdx.x * (ROWS * 4) + wave * ROWS;
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

    float acc[ROWS][N_ROWS_MAX];
    #pragma unroll
    for (int r = 0; r < ROWS; r++)
        #pragma unroll
        for (int b = 0; b < N_ROWS_MAX; b++) acc[r][b] = 0.0f;

    for (unsigned int sb = lane; sb < n_sub; sb += 64) {
        #pragma unroll
        for (int r = 0; r < ROWS; r++) {
            const int row = row0 + r;
            if (row >= (int)out_dim) continue;

            const uint4    q  = nib[(size_t)row * nsp + sb];
            const uint32_t qh = qhp[(size_t)row * nsp + sb];
            const uint16_t sm = smp[(size_t)row * nsp + sb];
            const uint32_t dd = ddp[(size_t)row * n_super + (sb >> 3)];
            const uint16_t d_bits    = (uint16_t)(dd & 0xFFFF);
            const uint16_t dmin_bits = (uint16_t)(dd >> 16);
            const float dsc  = __half2float(*reinterpret_cast<const __half*>(&d_bits))
                               * (float)(sm & 0xFFu);
            const float deff = __half2float(*reinterpret_cast<const __half*>(&dmin_bits))
                               * (float)(sm >> 8);

            // Form the 8 q5 chunks (4 lo + 4 hi) once per sub-block —
            // they're shared across the n_rows dot products.
            const uint32_t qa[4] = { q.x, q.y, q.z, q.w };
            uint32_t lo[4], hi[4];
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                lo[j] = ( qa[j]       & 0x0F0F0F0Fu)
                    | spread4((qh >> (4 * (2 * j)))     & 0xFu);
                hi[j] = ((qa[j] >> 4) & 0x0F0F0F0Fu)
                    | spread4((qh >> (4 * (2 * j + 1))) & 0xFu);
            }

            for (unsigned int b = 0; b < n_rows; b++) {
                const BlockQ8* xb   = xq + (size_t)b * n_sub + sb;
                const float    dx   = xb->d;
                const float    xsum = xb->xsum;
                const int*     xq32 = reinterpret_cast<const int*>(xb->qs);

                int idot = 0;
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    idot = __builtin_amdgcn_sdot4((int)lo[j], xq32[j],     idot, false);
                    idot = __builtin_amdgcn_sdot4((int)hi[j], xq32[j + 4], idot, false);
                }
                acc[r][b] += dsc * dx * (float)idot - deff * xsum;
            }
        }
    }

    #pragma unroll
    for (int r = 0; r < ROWS; r++) {
        for (unsigned int b = 0; b < n_rows; b++) {
            float a = acc[r][b];
            a += __shfl_xor(a, 32);
            a += __shfl_xor(a, 16);
            a += __shfl_xor(a,  8);
            a += __shfl_xor(a,  4);
            a += __shfl_xor(a,  2);
            a += __shfl_xor(a,  1);
            if (lane == 0 && (row0 + r) < (int)out_dim) {
                y[(size_t)b * out_dim + (row0 + r)] = a;
            }
        }
    }
}
