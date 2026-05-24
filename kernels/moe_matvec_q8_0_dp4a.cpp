// MoE expert matvec — Q8_0 weights, dp4a. Same expert-indexing scheme
// as moe_matvec_q6k_dp4a (see that file's comment for the n_tok /
// xq_tok_stride / xq_slot_stride contract). Used by the 26B-A4B
// layer-29 gate_up which is Q8_0 (rest of the experts are Q6_K).
//
// grid = (ceil(out_dim/ROWS), n_used, n_tok); block = 64.
// Decode callers pass n_tok=1.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include "gfx906_dpp.h"

#define ROWS 2

struct __attribute__((packed)) BlockQ8_0 {
    uint16_t d;
    int8_t   qs[32];
};
static_assert(sizeof(BlockQ8_0) == 34, "BlockQ8_0 must be 34 bytes");

struct __attribute__((packed)) BlockQ8 {
    float  d;
    float  xsum;
    int8_t qs[32];
};
static_assert(sizeof(BlockQ8) == 40, "BlockQ8 must be 40 bytes");

extern "C" __global__
void moe_matvec_q8_0_dp4a_f32(const unsigned char* __restrict__ slab,
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
    const BlockQ8_0* w_blocks =
        reinterpret_cast<const BlockQ8_0*>(slab + (size_t)eid * bytes_per_expert);
    const BlockQ8* xqs = xq + (size_t)tok * xq_tok_stride
                            + (size_t)slot * xq_slot_stride;
    float* yo = y + ((size_t)tok * n_used + slot) * out_dim;

    const int row0 = blockIdx.x * ROWS;
    const int lane = threadIdx.x;
    const unsigned int n_blocks = in_dim >> 5;

    float acc[ROWS];
    #pragma unroll
    for (int r = 0; r < ROWS; r++) acc[r] = 0.0f;

    for (int sb = lane; sb < (int)n_blocks; sb += 64) {
        const BlockQ8* xb = xqs + sb;
        const float dx   = xb->d;
        const int*  xq32 = reinterpret_cast<const int*>(xb->qs);

        #pragma unroll
        for (int r = 0; r < ROWS; r++) {
            const int row = row0 + r;
            if (row >= (int)out_dim) continue;
            const BlockQ8_0* blk = w_blocks + (size_t)row * n_blocks + sb;
            const float dw = __half2float(*reinterpret_cast<const __half*>(&blk->d));
            const uint16_t* qs16 = reinterpret_cast<const uint16_t*>(blk->qs);

            int idot = 0;
            #pragma unroll
            for (int g = 0; g < 8; g++) {
                const uint32_t wq = (uint32_t)qs16[2*g] | ((uint32_t)qs16[2*g+1] << 16);
                idot = __builtin_amdgcn_sdot4((int)wq, xq32[g], idot, false);
            }
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
