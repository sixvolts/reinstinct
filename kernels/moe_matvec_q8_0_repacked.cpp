// Routed-expert matvec over a slab of repacked Q8_0 experts (the
// `quant::q8_0::repack_for_matvec` layout per expert: a 32-byte int8
// plane per 32-weight sub-block, then an fp16 scale plane). Same launch
// contract as the other `moe_matvec_*_repacked` kernels:
//   grid (ceil(out_dim/8), n_used, n_tok), block 256 = 4 waves x 2 rows.
// `xq` is indexed `tok*xq_tok_stride + slot*xq_slot_stride` (BlockQ8
// units): gate/up share one activation per token (slot stride 0), down
// has one per (token, expert).
//
// `moe_matvec_q8_0_repacked_down_f32` is the row-packed variant for the
// down projection's small in_dim: thread t handles row t / n_sub,
// sub-block t % n_sub, so all 256 lanes stay busy at in_dim = 512..8192
// where the lane-per-sub-block layout above would idle most of a wave.
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

extern "C" __global__
void moe_matvec_q8_0_repacked_f32(const unsigned char* __restrict__ slab,
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
    const uint4*    lo_plane = reinterpret_cast<const uint4*>(wbase);
    const uint4*    hi_plane = reinterpret_cast<const uint4*>(wbase + (size_t)out_dim * nsp * 16);
    const uint16_t* d_plane  = reinterpret_cast<const uint16_t*>(
        wbase + (size_t)out_dim * nsp * 32);

    float acc[ROWS];
    #pragma unroll
    for (int r = 0; r < ROWS; r++) acc[r] = 0.0f;

    for (unsigned int sb = lane; sb < n_sub; sb += 64) {
        const BlockQ8* xb   = xqs + sb;
        const float    dx   = xb->d;
        const int*     xq32 = reinterpret_cast<const int*>(xb->qs);
        #pragma unroll
        for (int r = 0; r < ROWS; r++) {
            const int row = row0 + r;
            if (row >= (int)out_dim) continue;
            const size_t   idx   = (size_t)row * nsp + sb;
            const uint4    wl    = lo_plane[idx];
            const uint4    wh    = hi_plane[idx];
            const uint16_t db    = d_plane[idx];
            const float    dw    = __half2float(*reinterpret_cast<const __half*>(&db));
            const int w[8] = { (int)wl.x, (int)wl.y, (int)wl.z, (int)wl.w,
                               (int)wh.x, (int)wh.y, (int)wh.z, (int)wh.w };
            int idot = 0;
            #pragma unroll
            for (int g = 0; g < 8; g++)
                idot = __builtin_amdgcn_sdot4(w[g], xq32[g], idot, false);
            acc[r] += dw * dx * (float)idot;
        }
    }
    #pragma unroll
    for (int r = 0; r < ROWS; r++) {
        float a = acc[r];
        a = wave64_reduce_add_f32(a);
        if (lane == 0 && (row0 + r) < (int)out_dim) yo[row0 + r] = a;
    }
}

extern "C" __global__
void moe_matvec_q8_0_repacked_down_f32(const unsigned char* __restrict__ slab,
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
    __shared__ float red[256];
    const int tok  = blockIdx.z;
    const int slot = blockIdx.y;
    const int eid  = ids[(size_t)tok * n_used + slot];
    const uint8_t* wbase = slab + (size_t)eid * bytes_per_expert;
    const BlockQ8* xqs = xq + (size_t)tok * xq_tok_stride
                            + (size_t)slot * xq_slot_stride;
    float* yo = y + ((size_t)tok * n_used + slot) * out_dim;

    const unsigned int n_sub = in_dim >> 5;
    const unsigned int nsp   = ((n_sub & (n_sub - 1u)) == 0u) ? (n_sub + 1u) : n_sub;
    const unsigned int rpb   = 256u / n_sub;
    const unsigned int tid   = threadIdx.x;
    const unsigned int r     = tid / n_sub;
    const unsigned int sb    = tid % n_sub;
    const unsigned int row   = blockIdx.x * rpb + r;
    const bool active = (r < rpb) && (row < out_dim);

    const uint4*    lo_plane = reinterpret_cast<const uint4*>(wbase);
    const uint4*    hi_plane = reinterpret_cast<const uint4*>(wbase + (size_t)out_dim * nsp * 16);
    const uint16_t* d_plane  = reinterpret_cast<const uint16_t*>(
        wbase + (size_t)out_dim * nsp * 32);

    float contrib = 0.0f;
    if (active) {
        const BlockQ8* xb   = xqs + sb;
        const float    dx   = xb->d;
        const int*     xq32 = reinterpret_cast<const int*>(xb->qs);
        const size_t   idx   = (size_t)row * nsp + sb;
        const uint4    wl    = lo_plane[idx];
        const uint4    wh    = hi_plane[idx];
        const uint16_t db    = d_plane[idx];
        const float    dw    = __half2float(*reinterpret_cast<const __half*>(&db));
        const int w[8] = { (int)wl.x, (int)wl.y, (int)wl.z, (int)wl.w,
                           (int)wh.x, (int)wh.y, (int)wh.z, (int)wh.w };
        int idot = 0;
        #pragma unroll
        for (int g = 0; g < 8; g++)
            idot = __builtin_amdgcn_sdot4(w[g], xq32[g], idot, false);
        contrib = dw * dx * (float)idot;
    }
    red[tid] = contrib;
    __syncthreads();
    if (active && sb == 0) {
        float acc = 0.0f;
        for (unsigned int k = 0; k < n_sub; k++) acc += red[r * n_sub + k];
        yo[row] = acc;
    }
}
