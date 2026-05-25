// Q5_K matvec for the repacked three-plane layout
// (quant::q5_k::repack_for_matvec).
//
// Like matvec_q4k_repacked, plus a qh plane: per sub-block one u32 whose
// bit 4g+b is the 5th bit of weight b of dp4a group g. The kernel
// spreads those 4 bits to byte-bit-4 and ORs them onto the low nibbles
// to form the 0..31 quant before the dp4a.
//
//   dot = sum_sub [ dsc·dx·<q5·int8> - deff·xsum ]
//
// 256-thread workgroup: 4 wavefronts, ROWS=2 rows each. grid = ceil(out_dim/8).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include "gfx906_dpp.h"

#define ROWS 2

struct __attribute__((packed)) BlockQ8 {
    float  d;
    float  xsum;
    int8_t qs[32];
};
static_assert(sizeof(BlockQ8) == 40, "BlockQ8 must be 40 bytes");

// Spread 4 bits (b0..b3) to bit 4 of bytes 0..3.
__device__ __forceinline__ uint32_t spread4(uint32_t h) {
    return ((h & 1u) << 4) | ((h & 2u) << 11) | ((h & 4u) << 18) | ((h & 8u) << 25);
}

extern "C" __global__
void matvec_q5k_repacked_f32(const uint8_t* __restrict__ wbase,
                             const BlockQ8* __restrict__ xq,
                             float*         __restrict__ y,
                             unsigned int in_dim,
                             unsigned int out_dim)
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
    const uint16_t* smp = reinterpret_cast<const uint16_t*>(   // v2: sc|m per sub-block
        wbase + (size_t)out_dim * nsp * 16 + (size_t)out_dim * nsp * 4);
    const uint32_t* ddp = reinterpret_cast<const uint32_t*>(   // v2: d|dmin per superblock
        wbase + (size_t)out_dim * nsp * 16 + (size_t)out_dim * nsp * 4
              + (size_t)out_dim * nsp * 2);

    float acc[ROWS];
    #pragma unroll
    for (int r = 0; r < ROWS; r++) acc[r] = 0.0f;

    for (unsigned int sb = lane; sb < n_sub; sb += 64) {
        const BlockQ8* xb   = xq + sb;
        const float    dx   = xb->d;
        const float    xsum = xb->xsum;
        const int*     xq32 = reinterpret_cast<const int*>(xb->qs);

        #pragma unroll
        for (int r = 0; r < ROWS; r++) {
            const int row = row0 + r;
            if (row >= (int)out_dim) continue;

            const uint4    q  = loadnt_uint4(nib + (size_t)row * nsp + sb);
            const uint32_t qh = loadnt(qhp + (size_t)row * nsp + sb);
            const uint16_t sm = loadnt(smp + (size_t)row * nsp + sb);
            const uint32_t dd = loadnt(ddp + (size_t)row * n_super + (sb >> 3));
            const uint16_t d_bits    = (uint16_t)(dd & 0xFFFF);
            const uint16_t dmin_bits = (uint16_t)(dd >> 16);
            const float dsc  = __half2float(*reinterpret_cast<const __half*>(&d_bits))
                               * (float)(sm & 0xFFu);
            const float deff = __half2float(*reinterpret_cast<const __half*>(&dmin_bits))
                               * (float)(sm >> 8);

            const uint32_t qa[4] = { q.x, q.y, q.z, q.w };
            int idot = 0;
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                const uint32_t lo = ( qa[j]       & 0x0F0F0F0Fu)
                    | spread4((qh >> (4 * (2 * j)))     & 0xFu);
                const uint32_t hi = ((qa[j] >> 4) & 0x0F0F0F0Fu)
                    | spread4((qh >> (4 * (2 * j + 1))) & 0xFu);
                idot = __builtin_amdgcn_sdot4((int)lo, xq32[j],     idot, false);
                idot = __builtin_amdgcn_sdot4((int)hi, xq32[j + 4], idot, false);
            }
            acc[r] += dsc * dx * (float)idot - deff * xsum;
        }
    }

    #pragma unroll
    for (int r = 0; r < ROWS; r++) {
        float a = acc[r];
        a = wave64_reduce_add_f32(a);
        if (lane == 0 && (row0 + r) < (int)out_dim) y[row0 + r] = a;
    }
}
