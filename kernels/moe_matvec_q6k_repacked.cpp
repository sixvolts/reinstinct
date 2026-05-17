// MoE expert matvec — repacked Q6_K (quant::q6_k::repack_for_matvec
// applied per expert slice), dp4a. One launch handles all routed
// experts: grid.y selects the expert slot, the expert id is read from
// the device `ids` buffer, and the weight base is offset into the 3D
// repacked expert slab. Body is matvec_q6k_repacked's, with the
// expert-indexing prologue of moe_matvec_q6k_dp4a.
//
// 256-thread workgroup: 4 wavefronts, ROWS=2 rows each.
// grid = (ceil(out_dim/8), n_used).

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

__device__ __forceinline__ uint32_t spread2(uint32_t h) {
    return ((h & 0x03u) << 4) | ((h & 0x0Cu) << 10)
         | ((h & 0x30u) << 16) | ((h & 0xC0u) << 22);
}

extern "C" __global__
void moe_matvec_q6k_repacked_f32(const unsigned char* __restrict__ slab,
                                 const int*       __restrict__ ids,
                                 const BlockQ8*   __restrict__ xq,
                                 float*           __restrict__ y,
                                 unsigned int in_dim,
                                 unsigned int out_dim,
                                 unsigned int bytes_per_expert,
                                 unsigned int xq_stride)
{
    const int slot = blockIdx.y;
    const int eid  = ids[slot];
    const uint8_t* wbase = slab + (size_t)eid * bytes_per_expert;
    const BlockQ8* xqs = xq + (size_t)slot * xq_stride;
    float* yo = y + (size_t)slot * out_dim;

    const int wave = threadIdx.x >> 6;
    const int lane = threadIdx.x & 63;
    const int row0 = blockIdx.x * (ROWS * 4) + wave * ROWS;
    const unsigned int n_sub = in_dim >> 5;
    const unsigned int nsp = ((n_sub & (n_sub - 1u)) == 0u) ? (n_sub + 1u) : n_sub;

    const uint4*    nib = reinterpret_cast<const uint4*>(wbase);
    const uint32_t* h2p = reinterpret_cast<const uint32_t*>(
        wbase + (size_t)out_dim * nsp * 16);
    const uint32_t* scl = reinterpret_cast<const uint32_t*>(
        wbase + (size_t)out_dim * nsp * 16 + (size_t)out_dim * nsp * 8);

    float acc[ROWS];
    #pragma unroll
    for (int r = 0; r < ROWS; r++) acc[r] = 0.0f;

    for (unsigned int sb = lane; sb < n_sub; sb += 64) {
        const BlockQ8* xb   = xqs + sb;
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
            const int row = row0 + r;
            if (row >= (int)out_dim) continue;

            const size_t idx = (size_t)row * nsp + sb;
            const uint4    q    = nib[idx];
            const uint32_t h2lo = h2p[idx * 2];
            const uint32_t h2hi = h2p[idx * 2 + 1];
            const uint32_t s    = scl[idx];
            const uint16_t lo_bits = (uint16_t)(s & 0xFFFF);
            const uint16_t hi_bits = (uint16_t)(s >> 16);
            const float dsc_lo = __half2float(*reinterpret_cast<const __half*>(&lo_bits));
            const float dsc_hi = __half2float(*reinterpret_cast<const __half*>(&hi_bits));

            const uint32_t qa[4] = { q.x, q.y, q.z, q.w };
            int idot0 = 0, idot1 = 0;
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                const uint32_t ge = 2 * j;
                const uint32_t go = 2 * j + 1;
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
        a += __shfl_xor(a, 32);
        a += __shfl_xor(a, 16);
        a += __shfl_xor(a,  8);
        a += __shfl_xor(a,  4);
        a += __shfl_xor(a,  2);
        a += __shfl_xor(a,  1);
        if (lane == 0 && (row0 + r) < (int)out_dim) yo[row0 + r] = a;
    }
}
