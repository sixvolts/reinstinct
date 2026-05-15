// Row-blocked Q5_K matvec for gfx906 — same idea as the Q4_K rowblock.
//
// ROWS=8 output rows per wavefront; per sub-block each lane touches all
// 8 rows before consuming a result (8-deep memory pipeline). The 32
// low-nibble bytes and the 32 high-bit bytes are each read as 8×uint32
// rather than byte-at-a-time.
//
// grid = ceil(out_dim / 8); block = 64 (one wavefront).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#define ROWS 8

struct __attribute__((packed)) BlockQ5_K {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[12];
    uint8_t  qh[32];
    uint8_t  qs[128];
};
static_assert(sizeof(BlockQ5_K) == 176, "BlockQ5_K must be 176 bytes");

__device__ __forceinline__
void gsm_q5k_rb(int j, const uint8_t* q, uint8_t& sc, uint8_t& m) {
    if (j < 4) { sc = q[j] & 63; m = q[j + 4] & 63; }
    else {
        sc = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        m  = (q[j + 4] >> 4)   | ((q[j]     >> 6) << 4);
    }
}

extern "C" __global__
void matvec_q5_k_rowblock_f32(const BlockQ5_K* __restrict__ w_blocks,
                              const float*     __restrict__ x,
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
        const uint32_t qh_mask = 1u << (chunk * 2 + is);
        const int shift        = (is == 0) ? 0 : 4;
        const float* x_base    = x + (size_t)sb * 32;

        #pragma unroll
        for (int r = 0; r < ROWS; r++) {
            const int row = row0 + r;
            if (row >= (int)out_dim) continue;
            const BlockQ5_K* blk = w_blocks + (size_t)row * n_blocks + blk_idx;

            const float d    = __half2float(*reinterpret_cast<const __half*>(&blk->d));
            const float dmin = __half2float(*reinterpret_cast<const __half*>(&blk->dmin));
            uint8_t sc, m;
            gsm_q5k_rb(sub_in_block, blk->scales, sc, m);
            const float sub_d = d    * (float)sc;
            const float sub_m = dmin * (float)m;

            const uint32_t* qs32 =
                reinterpret_cast<const uint32_t*>(blk->qs + qs_off);
            const uint32_t* qh32 =
                reinterpret_cast<const uint32_t*>(blk->qh);

            float partial = 0.0f;
            #pragma unroll
            for (int g = 0; g < 8; g++) {
                const uint32_t pq = qs32[g];
                const uint32_t ph = qh32[g];
                const float4   xv = *reinterpret_cast<const float4*>(x_base + g * 4);
                #pragma unroll
                for (int b = 0; b < 4; b++) {
                    const uint32_t nib  = (pq >> (b * 8 + shift)) & 0x0F;
                    const uint32_t qhb  = (ph >> (b * 8)) & 0xFF;
                    const float    q5   = (float)(nib + ((qhb & qh_mask) ? 16u : 0u));
                    const float    xval = (&xv.x)[b];
                    partial += (sub_d * q5 - sub_m) * xval;
                }
            }
            acc[r] += partial;
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
