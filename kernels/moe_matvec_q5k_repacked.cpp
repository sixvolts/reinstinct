// MoE expert matvec — repacked Q5_K, dp4a. Body is matvec_q5k_repacked's
// (nibble + packed-qh + scale planes) with the expert-indexing prologue.
//
// 256-thread workgroup: 4 wavefronts, ROWS=2 rows each.
// grid = (ceil(out_dim/8), n_used, n_tok) — grid.z batches prefill
// tokens; decode launches with grid.z = 1.

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

// Spread 4 bits (b0..b3) to bit 4 of bytes 0..3.
__device__ __forceinline__ uint32_t spread4(uint32_t h) {
    return ((h & 1u) << 4) | ((h & 2u) << 11) | ((h & 4u) << 18) | ((h & 8u) << 25);
}

extern "C" __global__
void moe_matvec_q5k_repacked_f32(const unsigned char* __restrict__ slab,
                                 const int*       __restrict__ ids,
                                 const BlockQ8*   __restrict__ xq,
                                 float*           __restrict__ y,
                                 unsigned int in_dim,
                                 unsigned int out_dim,
                                 unsigned int bytes_per_expert,
                                 unsigned int xq_tok_stride,
                                 unsigned int xq_slot_stride,
                                 unsigned int n_used)
{
    const int tok  = blockIdx.z;
    const int slot = blockIdx.y;
    const int eid  = ids[(size_t)tok * n_used + slot];
    const uint8_t* wbase = slab + (size_t)eid * bytes_per_expert;
    const BlockQ8* xqs = xq + (size_t)tok * xq_tok_stride
                            + (size_t)slot * xq_slot_stride;
    float* yo = y + ((size_t)tok * n_used + slot) * out_dim;

    const int wave = threadIdx.x >> 6;
    const int lane = threadIdx.x & 63;
    const int row0 = blockIdx.x * (ROWS * 4) + wave * ROWS;
    const unsigned int n_sub = in_dim >> 5;
    const unsigned int nsp = ((n_sub & (n_sub - 1u)) == 0u) ? (n_sub + 1u) : n_sub;

    const uint4*    nib = reinterpret_cast<const uint4*>(wbase);
    const uint32_t* qhp = reinterpret_cast<const uint32_t*>(
        wbase + (size_t)out_dim * nsp * 16);
    const uint32_t* scl = reinterpret_cast<const uint32_t*>(
        wbase + (size_t)out_dim * nsp * 16 + (size_t)out_dim * nsp * 4);

    float acc[ROWS];
    #pragma unroll
    for (int r = 0; r < ROWS; r++) acc[r] = 0.0f;

    for (unsigned int sb = lane; sb < n_sub; sb += 64) {
        const BlockQ8* xb   = xqs + sb;
        const float    dx   = xb->d;
        const float    xsum = xb->xsum;
        const int*     xq32 = reinterpret_cast<const int*>(xb->qs);

        #pragma unroll
        for (int r = 0; r < ROWS; r++) {
            const int row = row0 + r;
            if (row >= (int)out_dim) continue;

            const uint4    q  = nib[(size_t)row * nsp + sb];
            const uint32_t qh = qhp[(size_t)row * nsp + sb];
            const uint32_t s  = scl[(size_t)row * nsp + sb];
            const uint16_t dsc_bits  = (uint16_t)(s & 0xFFFF);
            const uint16_t deff_bits = (uint16_t)(s >> 16);
            const float dsc  = __half2float(*reinterpret_cast<const __half*>(&dsc_bits));
            const float deff = __half2float(*reinterpret_cast<const __half*>(&deff_bits));

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
        a += __shfl_xor(a, 32);
        a += __shfl_xor(a, 16);
        a += __shfl_xor(a,  8);
        a += __shfl_xor(a,  4);
        a += __shfl_xor(a,  2);
        a += __shfl_xor(a,  1);
        if (lane == 0 && (row0 + r) < (int)out_dim) yo[row0 + r] = a;
    }
}
