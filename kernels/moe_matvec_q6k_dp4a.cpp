// MoE expert matvec — Q6_K weights, dp4a. One launch handles all
// `n_used` routed experts: grid.y selects the expert slot, the expert
// id is read from the device `ids` buffer (no host round-trip), and the
// weight base is offset into the 3D expert slab. The activation can be
// shared across slots (xq_stride = 0, the fused gate_up case) or
// per-slot (xq_stride > 0).
//
// Body is matvec_q6_k_dp4a's Q6_K dp4a with an expert-indexing prologue.
// grid = (ceil(out_dim/8), n_used); block = 64.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#define ROWS 8

struct __attribute__((packed)) BlockQ6_K {
    uint8_t  ql[128];
    uint8_t  qh[64];
    int8_t   scales[16];
    uint16_t d;
};
static_assert(sizeof(BlockQ6_K) == 210, "BlockQ6_K must be 210 bytes");

struct __attribute__((packed)) BlockQ8 {
    float  d;
    float  xsum;
    int8_t qs[32];
};
static_assert(sizeof(BlockQ8) == 40, "BlockQ8 must be 40 bytes");

extern "C" __global__
void moe_matvec_q6k_dp4a_f32(const unsigned char* __restrict__ slab,
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
    const BlockQ6_K* w_blocks =
        reinterpret_cast<const BlockQ6_K*>(slab + (size_t)eid * bytes_per_expert);
    const BlockQ8* xqs = xq + (size_t)slot * xq_stride;
    float* yo = y + (size_t)slot * out_dim;

    const int row0 = blockIdx.x * ROWS;
    const int lane = threadIdx.x;
    const unsigned int n_blocks    = in_dim >> 8;
    const unsigned int n_subblocks = in_dim >> 4;

    float acc[ROWS];
    #pragma unroll
    for (int r = 0; r < ROWS; r++) acc[r] = 0.0f;

    for (int sb = lane; sb < (int)n_subblocks; sb += 64) {
        const int blk_idx     = sb >> 4;
        const int sb_in_blk   = sb & 15;
        const int chunk       = sb_in_blk >> 3;
        const int sb_in_chunk = sb_in_blk & 7;
        const int group       = sb_in_chunk >> 1;
        const int is          = sb_in_chunk & 1;
        const int sc_idx      = chunk * 8 + is + group * 2;
        const int l_start     = is * 16;
        const int qh_shift    = group * 2;
        const int ql_off      = chunk * 64 + (group & 1) * 32 + l_start;
        const int ql_high     = group >> 1;
        const int qh_off      = chunk * 32 + l_start;

        const BlockQ8* xb = xqs + (sb >> 1);
        const float dx   = xb->d;
        const int*  xq32 = reinterpret_cast<const int*>(xb->qs + (sb & 1) * 16);

        int xisum = 0;
        #pragma unroll
        for (int g = 0; g < 4; g++)
            xisum = __builtin_amdgcn_sdot4(xq32[g], 0x01010101, xisum, false);

        #pragma unroll
        for (int r = 0; r < ROWS; r++) {
            const int row = row0 + r;
            if (row >= (int)out_dim) continue;
            const BlockQ6_K* blk = w_blocks + (size_t)row * n_blocks + blk_idx;

            const float d_w   = __half2float(*reinterpret_cast<const __half*>(&blk->d));
            const float scale = (float)blk->scales[sc_idx];

            const uint16_t* ql16 =
                reinterpret_cast<const uint16_t*>(blk->ql + ql_off);
            const uint16_t* qh16 =
                reinterpret_cast<const uint16_t*>(blk->qh + qh_off);

            int idot = 0;
            #pragma unroll
            for (int g = 0; g < 4; g++) {
                const uint32_t qlw =
                    (uint32_t)ql16[2*g] | ((uint32_t)ql16[2*g+1] << 16);
                const uint32_t qhw =
                    (uint32_t)qh16[2*g] | ((uint32_t)qh16[2*g+1] << 16);
                const uint32_t ql_part = ql_high
                    ? ((qlw >> 4) & 0x0F0F0F0Fu)
                    : ( qlw       & 0x0F0F0F0Fu);
                const uint32_t qh_part =
                    ((qhw >> qh_shift) & 0x03030303u) << 4;
                const uint32_t q6 = ql_part | qh_part;
                idot = __builtin_amdgcn_sdot4((int)q6, xq32[g], idot, false);
            }
            acc[r] += d_w * scale * dx * (float)(idot - 32 * xisum);
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
