// MoE expert DOWN matvec — repacked Q6_K, row-packed for small in_dim.
// Q6_K analogue of moe_matvec_q5k_down — see that file for the rationale
// (the down projection's small in_dim leaves 3/4 of each wavefront idle
// under the standard lane->sub-block mapping).
//
// Each workgroup covers DOWN_R row groups with all loads issued first —
// see moe_matvec_q5k_down.
//
// grid = (ceil(out_dim / (rows_per_block * DOWN_R)), n_used, n_tok); block 256.

// No default: a kernel compiled with fewer groups than its launcher's
// grid assumes would leave rows unwritten.
#ifndef DOWN_R
#error "moe down kernel: the runtime must #define DOWN_R (see down_src)"
#endif

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

struct __attribute__((packed)) BlockQ8 {
    float  d;
    float  xsum;
    int8_t qs[32];
};
static_assert(sizeof(BlockQ8) == 40, "BlockQ8 must be 40 bytes");

// Spread 4 2-bit fields of h to bits 4-5 of bytes 0..3 — two multiplies
// (one would carry between fields), as in matvec_q6k_repacked.
__device__ __forceinline__ uint32_t spread2(uint32_t h) {
    return (((h & 0x33u) * 0x00010010u) & 0x00300030u)
         | ((((h >> 2) & 0x33u) * 0x01001000u) & 0x30003000u);
}

extern "C" __global__
void moe_matvec_q6k_down_f32(const unsigned char* __restrict__ slab,
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
    __shared__ float red[DOWN_R * 256];
    const int tok  = blockIdx.z;
    const int slot = blockIdx.y;
    const int eid  = ids[(size_t)tok * n_used + slot];
    const uint8_t* wbase = slab + (size_t)eid * bytes_per_expert;
    const BlockQ8* xqs = xq + (size_t)tok * xq_tok_stride
                            + (size_t)slot * xq_slot_stride;
    float* yo = y + ((size_t)tok * n_used + slot) * out_dim;

    const unsigned int n_sub   = in_dim >> 5;
    const unsigned int nsp     = ((n_sub & (n_sub - 1u)) == 0u) ? (n_sub + 1u) : n_sub;
    const unsigned int n_super = n_sub >> 3;
    const unsigned int rpb     = 256u / n_sub;
    const unsigned int tid     = threadIdx.x;
    const unsigned int r       = tid / n_sub;
    const unsigned int sb      = tid % n_sub;
    const bool lane_ok = r < rpb;
    const unsigned int row_base = blockIdx.x * rpb * DOWN_R + r;
    const unsigned int rmax = out_dim - 1;
    const unsigned int sbc = lane_ok ? sb : 0u;

    const uint4*    nib = reinterpret_cast<const uint4*>(wbase);
    const uint32_t* h2p = reinterpret_cast<const uint32_t*>(
        wbase + (size_t)out_dim * nsp * 16);
    const uint16_t* smp = reinterpret_cast<const uint16_t*>(
        wbase + (size_t)out_dim * nsp * 16 + (size_t)out_dim * nsp * 8);
    const uint16_t* ddp = reinterpret_cast<const uint16_t*>(
        wbase + (size_t)out_dim * nsp * 16 + (size_t)out_dim * nsp * 8
              + (size_t)out_dim * nsp * 2);

    // --- all loads first ---
    uint4 q[DOWN_R]; uint32_t h2lo[DOWN_R], h2hi[DOWN_R]; uint16_t sm[DOWN_R], db[DOWN_R];
    #pragma unroll
    for (int k = 0; k < DOWN_R; k++) {
        const unsigned int row = min(row_base + k * rpb, rmax);
        const size_t idx = (size_t)row * nsp + sbc;
        q[k]    = nib[idx];
        h2lo[k] = h2p[idx * 2];
        h2hi[k] = h2p[idx * 2 + 1];
        sm[k]   = smp[idx];
        db[k]   = ddp[(size_t)row * n_super + (sbc >> 3)];
    }
    const BlockQ8* xb   = xqs + sbc;
    const float    dx   = xb->d;
    const int*     xq32 = reinterpret_cast<const int*>(xb->qs);
    int xis0 = 0, xis1 = 0;
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        xis0 = __builtin_amdgcn_sdot4(xq32[j],     0x01010101, xis0, false);
        xis1 = __builtin_amdgcn_sdot4(xq32[j + 4], 0x01010101, xis1, false);
    }

    #pragma unroll
    for (int k = 0; k < DOWN_R; k++) {
        const float d = __half2float(*reinterpret_cast<const __half*>(&db[k]));
        const float dsc_lo = d * (float)(int)(int8_t)(sm[k] & 0xFFu);
        const float dsc_hi = d * (float)(int)(int8_t)(sm[k] >> 8);
        const uint32_t qa[4] = { q[k].x, q[k].y, q[k].z, q[k].w };
        int idot0 = 0, idot1 = 0;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            const uint32_t ge = 2 * j;
            const uint32_t go = 2 * j + 1;
            const uint32_t he = ((ge < 4 ? h2lo[k] : h2hi[k]) >> (8 * (ge & 3))) & 0xFFu;
            const uint32_t ho = ((go < 4 ? h2lo[k] : h2hi[k]) >> (8 * (go & 3))) & 0xFFu;
            const uint32_t q6lo = ( qa[j]       & 0x0F0F0F0Fu) | spread2(he);
            const uint32_t q6hi = ((qa[j] >> 4) & 0x0F0F0F0Fu) | spread2(ho);
            idot0 = __builtin_amdgcn_sdot4((int)q6lo, xq32[j],     idot0, false);
            idot1 = __builtin_amdgcn_sdot4((int)q6hi, xq32[j + 4], idot1, false);
        }
        red[k * 256 + tid] = dsc_lo * dx * (float)(idot0 - 32 * xis0)
                           + dsc_hi * dx * (float)(idot1 - 32 * xis1);
    }
    __syncthreads();

    if (lane_ok && sb == 0) {
        #pragma unroll
        for (int k = 0; k < DOWN_R; k++) {
            const unsigned int row = row_base + k * rpb;
            if (row > rmax) break;
            float acc = 0.0f;
            for (unsigned int j = 0; j < n_sub; j++) acc += red[k * 256 + r * n_sub + j];
            yo[row] = acc;
        }
    }
}
