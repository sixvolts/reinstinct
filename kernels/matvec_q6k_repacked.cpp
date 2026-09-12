// Q6_K matvec for the repacked three-plane layout
// (quant::q6_k::repack_for_matvec).
//
// Q6_K is symmetric: real value = q-32. Rather than subtracting 32 per
// byte (which borrows across the packed dp4a lanes) the offset is folded
// out:  sum (q-32)·x = sum q·x - 32·sum x.  Each 32-weight sub-block
// carries two scales (one per 16 weights); the low-nibble dp4a groups
// cover weights 0..15, the high-nibble groups 16..31.
//
//   dot = sum_sub [ dsc_lo·dx·(idot0 - 32·xis0)
//                 + dsc_hi·dx·(idot1 - 32·xis1) ]
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

// Spread four 2-bit fields (weight b at bits 2b..2b+1) to bits 4..5 of
// bytes 0..3 — the position the 6-bit quant's high pair occupies.
// Four 2-bit fields of h (bits 0-1, 2-3, 4-5, 6-7) to bits 4-5, 12-13,
// 20-21, 28-29. Two multiplies, one per pair of fields, so no two
// partial products share a bit (a single multiply would carry between
// fields 3 and 0 at bit 12).
__device__ __forceinline__ uint32_t spread2(uint32_t h) {
    return (((h & 0x33u) * 0x00010010u) & 0x00300030u)
         | ((((h >> 2) & 0x33u) * 0x01001000u) & 0x30003000u);
}

extern "C" __global__
void matvec_q6k_repacked_f32(const uint8_t* __restrict__ wbase,
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
    const uint32_t* h2p = reinterpret_cast<const uint32_t*>(
        wbase + (size_t)out_dim * nsp * 16);
    const uint16_t* smp = reinterpret_cast<const uint16_t*>(   // v2: sc_lo|sc_hi int8
        wbase + (size_t)out_dim * nsp * 16 + (size_t)out_dim * nsp * 8);
    const uint16_t* ddp = reinterpret_cast<const uint16_t*>(   // v2: d per superblock
        wbase + (size_t)out_dim * nsp * 16 + (size_t)out_dim * nsp * 8
              + (size_t)out_dim * nsp * 2);

    float acc[ROWS];
    #pragma unroll
    for (int r = 0; r < ROWS; r++) acc[r] = 0.0f;

    const int rmax = (int)out_dim - 1;   // clamped rows: see matvec_q4k_repacked.cpp
    for (unsigned int sb = lane; sb < n_sub; sb += 64) {
        uint4 q[ROWS]; uint32_t h2lo[ROWS], h2hi[ROWS]; uint16_t sm[ROWS], db[ROWS];
        #pragma unroll
        for (int r = 0; r < ROWS; r++) {
            const int row = min(row0 + r, rmax);
            const size_t idx = (size_t)row * nsp + sb;
            q[r]    = nib[idx];
            h2lo[r] = h2p[idx * 2];
            h2hi[r] = h2p[idx * 2 + 1];
            sm[r]   = smp[idx];
            db[r]   = ddp[(size_t)row * n_super + (sb >> 3)];
        }
        const BlockQ8* xb   = xq + sb;
        const float    dx   = xb->d;
        const int*     xq32 = reinterpret_cast<const int*>(xb->qs);
        int xis0 = 0, xis1 = 0;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            xis0 = __builtin_amdgcn_sdot4(xq32[j],     0x01010101, xis0, false);
            xis1 = __builtin_amdgcn_sdot4(xq32[j + 4], 0x01010101, xis1, false);
        }
        #pragma unroll
        for (int r = 0; r < ROWS; r++) {
            const float d = __half2float(*reinterpret_cast<const __half*>(&db[r]));
            const float dsc_lo = d * (float)(int)(int8_t)(sm[r] & 0xFFu);
            const float dsc_hi = d * (float)(int)(int8_t)(sm[r] >> 8);
            const uint32_t qa[4] = { q[r].x, q[r].y, q[r].z, q[r].w };
            int idot0 = 0, idot1 = 0;
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                // low nibbles of word j = group 2j, high = group 2j+1;
                // byte g of the plane holds group g's fields.
                const uint32_t he = ((2 * j < 4 ? h2lo[r] : h2hi[r]) >> (8 * ((2 * j) & 3))) & 0xFFu;
                const uint32_t ho = ((2 * j + 1 < 4 ? h2lo[r] : h2hi[r]) >> (8 * ((2 * j + 1) & 3))) & 0xFFu;
                const uint32_t q6lo = ( qa[j]       & 0x0F0F0F0Fu) | spread2(he);
                const uint32_t q6hi = ((qa[j] >> 4) & 0x0F0F0F0Fu) | spread2(ho);
                idot0 = __builtin_amdgcn_sdot4((int)q6lo, xq32[j],     idot0, false);
                idot1 = __builtin_amdgcn_sdot4((int)q6hi, xq32[j + 4], idot1, false);
            }
            acc[r] += dsc_lo * dx * (float)(idot0 - 32 * xis0)
                    + dsc_hi * dx * (float)(idot1 - 32 * xis1);
        }
    }

    #pragma unroll
    for (int r = 0; r < ROWS; r++) {
        float a = acc[r];
        a = wave64_reduce_add_f32(a);
        if (lane == 0 && (row0 + r) < (int)out_dim) y[row0 + r] = a;
    }
}
