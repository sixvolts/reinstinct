// Q8_0 matvec via gfx906's v_dot4_i32_i8 (dp4a).
//
// Q8_0 is the simplest dp4a case — the weights are already int8 with a
// per-32 f16 scale. The activation is pre-quantized to int8 BlockQ8
// (see quantize_q8.cpp); each 32-weight block is one sdot4 chain.
//
//   dot = sum_block d_w · d_x · <int8 weights · int8 activation>
//
// Q8_0 blocks are 34 bytes, so the int8 payload is only 2-byte aligned —
// it is read as uint16 pairs and recombined. ROWS=8 rows per wavefront.
//
// grid = ceil(out_dim/8); block = 64.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#define ROWS 8

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
void matvec_q8_0_dp4a_f32(const BlockQ8_0* __restrict__ w_blocks,
                          const BlockQ8*   __restrict__ xq,
                          float*           __restrict__ y,
                          unsigned int in_dim,
                          unsigned int out_dim)
{
    const int row0 = blockIdx.x * ROWS;
    const int lane = threadIdx.x;
    const unsigned int n_blocks = in_dim >> 5;       // 32 weights per block

    float acc[ROWS];
    #pragma unroll
    for (int r = 0; r < ROWS; r++) acc[r] = 0.0f;

    for (int sb = lane; sb < (int)n_blocks; sb += 64) {
        const BlockQ8* xb = xq + sb;
        const float dx   = xb->d;
        const int*  xq32 = reinterpret_cast<const int*>(xb->qs);

        #pragma unroll
        for (int r = 0; r < ROWS; r++) {
            const int row = row0 + r;
            if (row >= (int)out_dim) continue;
            const BlockQ8_0* blk = w_blocks + (size_t)row * n_blocks + sb;
            const float dw = __half2float(*reinterpret_cast<const __half*>(&blk->d));
            // qs is 2-byte aligned only — read as uint16 pairs.
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
        a += __shfl_xor(a, 32);
        a += __shfl_xor(a, 16);
        a += __shfl_xor(a,  8);
        a += __shfl_xor(a,  4);
        a += __shfl_xor(a,  2);
        a += __shfl_xor(a,  1);
        if (lane == 0 && (row0 + r) < (int)out_dim) y[row0 + r] = a;
    }
}
