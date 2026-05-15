// Q6_K matvec via gfx906's v_dot4_i32_i8 (dp4a).
//
// Q6_K is symmetric (q in 0..63, real value q-32, no dmin). Rather than
// byte-wise subtracting 32 (which would borrow across packed bytes) the
// offset is folded out:  sum (q-32)·xi = sum q·xi - 32·sum xi.  Both
// sums are dp4a reductions — the second against a vector of ones.
//
// Lane = one 16-weight sub-block; the activation is the 32-int8 BlockQ8
// (one per 32 elements), indexed by half. ROWS=8 rows per wavefront.
//
// grid = ceil(out_dim/8); block = 64.

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
void matvec_q6_k_dp4a_f32(const BlockQ6_K* __restrict__ w_blocks,
                          const BlockQ8*   __restrict__ xq,
                          float*           __restrict__ y,
                          unsigned int in_dim,
                          unsigned int out_dim)
{
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

        // Activation: 32-int8 block (sb>>1), 16-element half (sb&1).
        const BlockQ8* xb = xq + (sb >> 1);
        const float dx   = xb->d;
        const int*  xq32 = reinterpret_cast<const int*>(xb->qs + (sb & 1) * 16);

        // sum of the 16 int8 activations — shared across rows.
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

            // 210-byte blocks are only 2-byte aligned — read uint16.
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
                const uint32_t q6 = ql_part | qh_part;   // 4 bytes, 0..63
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
        if (lane == 0 && (row0 + r) < (int)out_dim) y[row0 + r] = a;
    }
}
