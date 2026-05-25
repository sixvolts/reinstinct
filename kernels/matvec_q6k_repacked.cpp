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
__device__ __forceinline__ uint32_t spread2(uint32_t h) {
    return ((h & 0x03u) << 4) | ((h & 0x0Cu) << 10)
         | ((h & 0x30u) << 16) | ((h & 0xC0u) << 22);
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

    for (unsigned int sb = lane; sb < n_sub; sb += 64) {
        const BlockQ8* xb   = xq + sb;
        const float    dx   = xb->d;
        const int*     xq32 = reinterpret_cast<const int*>(xb->qs);

        // Activation half-sums for the symmetric −32 fold.
        int xis0 = 0, xis1 = 0;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            xis0 = __builtin_amdgcn_sdot4(xq32[j],     0x01010101, xis0, false);
            xis1 = __builtin_amdgcn_sdot4(xq32[j + 4], 0x01010101, xis1, false);
        }

        #pragma unroll
        for (int r = 0; r < ROWS; r++) {
            const int row = row0 + r;
            if (row >= (int)out_dim) continue;

            const size_t idx = (size_t)row * nsp + sb;
            const uint4    q    = loadnt_uint4(nib + idx);
            const uint32_t h2lo = loadnt(h2p + idx * 2);       // groups 0..3
            const uint32_t h2hi = loadnt(h2p + idx * 2 + 1);   // groups 4..7
            const uint16_t sm     = loadnt(smp + idx);
            const uint16_t d_bits = loadnt(ddp + (size_t)row * n_super + (sb >> 3));
            const float d = __half2float(*reinterpret_cast<const __half*>(&d_bits));
            const float dsc_lo = d * (float)(int)(int8_t)(sm & 0xFFu);
            const float dsc_hi = d * (float)(int)(int8_t)(sm >> 8);

            const uint32_t qa[4] = { q.x, q.y, q.z, q.w };
            int idot0 = 0, idot1 = 0;
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                const uint32_t ge = 2 * j;            // low-nibble group
                const uint32_t go = 2 * j + 1;        // high-nibble group
                const uint32_t he = ((ge < 4 ? h2lo : h2hi) >> (8 * (ge & 3))) & 0xFFu;
                const uint32_t ho = ((go < 4 ? h2lo : h2hi) >> (8 * (go & 3))) & 0xFFu;
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
