// Q4_K matvec via gfx906's v_dot4_i32_i8 (dp4a).
//
// The activation is pre-quantized to int8 blocks of 32 (BlockQ8). For
// each 32-weight sub-block the integer dot of the 4-bit weight nibbles
// with the int8 activation is accumulated 4-at-a-time with sdot4, then
// rescaled. This replaces the per-element shift / int->float / fma of
// the f32-dequant matvec — far fewer instructions, so the kernel moves
// from issue-bound toward bandwidth-bound.
//
//   dot = sum_sub [ (d_w·sc)·d_x·<nibbles·int8> - (dmin_w·m)·xsum ]
//
// ROWS=2 output rows per wavefront; grid = ceil(out_dim/ROWS); block = 64.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#define ROWS 2

struct __attribute__((packed)) BlockQ4_K {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[12];
    uint8_t  qs[128];
};
static_assert(sizeof(BlockQ4_K) == 144, "BlockQ4_K must be 144 bytes");

struct __attribute__((packed)) BlockQ8 {
    float  d;
    float  xsum;
    int8_t qs[32];
};
static_assert(sizeof(BlockQ8) == 40, "BlockQ8 must be 40 bytes");

__device__ __forceinline__
void gsm_q4k_dp(int j, const uint8_t* q, uint8_t& sc, uint8_t& m) {
    if (j < 4) { sc = q[j] & 63; m = q[j + 4] & 63; }
    else {
        sc = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        m  = (q[j + 4] >> 4)   | ((q[j]     >> 6) << 4);
    }
}

extern "C" __global__
void matvec_q4_k_dp4a_f32(const BlockQ4_K* __restrict__ w_blocks,
                          const BlockQ8*   __restrict__ xq,
                          float*           __restrict__ y,
                          unsigned int in_dim,
                          unsigned int out_dim)
{
    const int row0 = blockIdx.x * ROWS;
    const int lane = threadIdx.x;
    const unsigned int n_blocks    = in_dim >> 8;
    const unsigned int n_subblocks = in_dim >> 5;

    float acc[ROWS];
    #pragma unroll
    for (int r = 0; r < ROWS; r++) acc[r] = 0.0f;

    for (int sb = lane; sb < (int)n_subblocks; sb += 64) {
        const int blk_idx      = sb >> 3;
        const int sub_in_block = sb & 7;
        const int chunk        = sub_in_block >> 1;
        const int is           = sub_in_block & 1;
        const int qs_off       = chunk * 32;

        const BlockQ8* xb = xq + sb;
        const float dx   = xb->d;
        const float xsum = xb->xsum;
        const int*  xq32 = reinterpret_cast<const int*>(xb->qs);

        #pragma unroll
        for (int r = 0; r < ROWS; r++) {
            const int row = row0 + r;
            if (row >= (int)out_dim) continue;
            const BlockQ4_K* blk = w_blocks + (size_t)row * n_blocks + blk_idx;

            const float d_w    = __half2float(*reinterpret_cast<const __half*>(&blk->d));
            const float dmin_w = __half2float(*reinterpret_cast<const __half*>(&blk->dmin));
            uint8_t sc, m;
            gsm_q4k_dp(sub_in_block, blk->scales, sc, m);

            const uint32_t* qs32 =
                reinterpret_cast<const uint32_t*>(blk->qs + qs_off);

            int idot = 0;
            #pragma unroll
            for (int g = 0; g < 8; g++) {
                const uint32_t wn = (is == 0)
                    ? ( qs32[g]       & 0x0F0F0F0Fu)
                    : ((qs32[g] >> 4) & 0x0F0F0F0Fu);
                idot = __builtin_amdgcn_sdot4((int)wn, xq32[g], idot, false);
            }
            acc[r] += (d_w * (float)sc) * dx * (float)idot
                    - (dmin_w * (float)m) * xsum;
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
