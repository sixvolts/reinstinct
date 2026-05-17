// Q4_K matvec for the repacked two-plane weight layout (see
// quant::q4_k::repack_for_matvec).
//
// The on-disk Q4_K superblock layout forces a strided, partly-duplicated
// global read that sustains only ~45% of HBM streaming bandwidth. The
// weights are repacked once at load time into two planes:
//
//   * nibble plane — every 32-weight sub-block in its own contiguous 16
//     bytes, sub-block order, nibbles pre-permuted so uint32 `j` carries
//     weights 4j..4j+3 (low nibbles) and 16+4j..16+4j+3 (high nibbles).
//   * scale plane — per sub-block, fp16(d·sc) and fp16(dmin·m).
//
// Lane l reads sub-block l's 16 bytes (one uint4) — consecutive lanes,
// consecutive memory — so the weight read is a fully-coalesced
// contiguous sweep, the pattern that hits peak bandwidth. No LDS, no
// gsm scale unpack.
//
//   dot = sum_sub [ dsc·dx·<nibbles·int8> - deff·xsum ]
//
// 256-thread workgroup: 4 independent wavefronts, ROWS=2 rows each.
// grid = ceil(out_dim / 8).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#define ROWS 2

struct __attribute__((packed)) BlockQ8 {
    float  d;
    float  xsum;
    int8_t qs[32];
};
static_assert(sizeof(BlockQ8) == 40, "BlockQ8 must be 40 bytes");

extern "C" __global__
void matvec_q4k_repacked_f32(const uint8_t* __restrict__ wbase,
                             const BlockQ8* __restrict__ xq,
                             float*         __restrict__ y,
                             unsigned int in_dim,
                             unsigned int out_dim)
{
    const int wave = threadIdx.x >> 6;          // 0..3
    const int lane = threadIdx.x & 63;
    const int row0 = blockIdx.x * (ROWS * 4) + wave * ROWS;
    const unsigned int n_sub = in_dim >> 5;
    // Padded sub-blocks per row: a power-of-two count gives a power-of-two
    // row stride that aliases all rows onto one HBM channel — pad by one
    // to break it. (Must match quant::q4_k::repacked_n_sub_padded.)
    const unsigned int nsp = ((n_sub & (n_sub - 1u)) == 0u) ? (n_sub + 1u) : n_sub;

    const uint4*    nib = reinterpret_cast<const uint4*>(wbase);
    const uint32_t* scl = reinterpret_cast<const uint32_t*>(
        wbase + (size_t)out_dim * nsp * 16);

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

            const uint4    q = nib[(size_t)row * nsp + sb];
            const uint32_t s = scl[(size_t)row * nsp + sb];
            const uint16_t dsc_bits  = (uint16_t)(s & 0xFFFF);
            const uint16_t deff_bits = (uint16_t)(s >> 16);
            const float dsc  = __half2float(*reinterpret_cast<const __half*>(&dsc_bits));
            const float deff = __half2float(*reinterpret_cast<const __half*>(&deff_bits));

            const uint32_t qa[4] = { q.x, q.y, q.z, q.w };
            int idot = 0;
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                idot = __builtin_amdgcn_sdot4((int)( qa[j]       & 0x0F0F0F0Fu),
                                              xq32[j],     idot, false);
                idot = __builtin_amdgcn_sdot4((int)((qa[j] >> 4) & 0x0F0F0F0Fu),
                                              xq32[j + 4], idot, false);
            }
            acc[r] += dsc * dx * (float)idot - deff * xsum;
        }
    }

    #pragma unroll
    for (int r = 0; r < ROWS; r++) {
        float a = acc[r];
        a += __shfl_xor(a, 32);
        a += __shfl_xor(a, 16);
        a += __shfl_xor(a,  8);
        a += __shfl_xor(a,  4);
        a += __shfl_xor(a,  2);
        a += __shfl_xor(a,  1);
        if (lane == 0 && (row0 + r) < (int)out_dim) y[row0 + r] = a;
    }
}
