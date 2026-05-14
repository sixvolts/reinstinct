// Wave-cooperative Q5_K matvec for gfx906. Same dequant as
// matvec_q5_k.cpp; reduction via __shfl_xor across 64 lanes.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

struct __attribute__((packed)) BlockQ5_K {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[12];
    uint8_t  qh[32];
    uint8_t  qs[128];
};
static_assert(sizeof(BlockQ5_K) == 176, "BlockQ5_K must be 176 bytes");

__device__ __forceinline__
void get_scale_min_k4_q5w(int j, const uint8_t* q, uint8_t& sc, uint8_t& m) {
    if (j < 4) {
        sc = q[j]     & 63;
        m  = q[j + 4] & 63;
    } else {
        sc = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        m  = (q[j + 4] >> 4)   | ((q[j]     >> 6) << 4);
    }
}

extern "C" __global__
void matvec_q5_k_wave64_f32(const BlockQ5_K* __restrict__ w_blocks,
                            const float*     __restrict__ x,
                            float*           __restrict__ y,
                            unsigned int in_dim,
                            unsigned int out_dim)
{
    const int row = blockIdx.x;
    if (row >= (int)out_dim) return;
    const int lane = threadIdx.x;

    const unsigned int n_blocks    = in_dim >> 8;
    const unsigned int n_subblocks = in_dim >> 5;
    const BlockQ5_K* row_blocks = w_blocks + (size_t)row * n_blocks;

    float acc = 0.0f;
    for (int sb = lane; sb < (int)n_subblocks; sb += 64) {
        const int blk_idx      = sb >> 3;
        const int sub_in_block = sb & 7;
        const int chunk        = sub_in_block >> 1;
        const int is           = sub_in_block & 1;
        const int qh_mask      = 1 << (chunk * 2 + is);
        const int qs_off_blk   = chunk * 32;

        const BlockQ5_K* blk = row_blocks + blk_idx;
        const __half d_h    = *reinterpret_cast<const __half*>(&blk->d);
        const __half dm_h   = *reinterpret_cast<const __half*>(&blk->dmin);
        const float  d      = __half2float(d_h);
        const float  dmin   = __half2float(dm_h);

        uint8_t sc, m;
        get_scale_min_k4_q5w(sub_in_block, blk->scales, sc, m);
        const float sub_d = d    * (float)sc;
        const float sub_m = dmin * (float)m;

        const uint8_t* qs_base = blk->qs + qs_off_blk;
        const uint8_t* qh_base = blk->qh;
        const float*   x_base  = x + (size_t)sb * 32;

        float partial = 0.0f;
        if (is == 0) {
            #pragma unroll
            for (int l = 0; l < 32; l++) {
                const int q_lo = (int)(qs_base[l] & 0x0F)
                               + ((qh_base[l] & qh_mask) ? 16 : 0);
                const float w = sub_d * (float)q_lo - sub_m;
                partial += w * x_base[l];
            }
        } else {
            #pragma unroll
            for (int l = 0; l < 32; l++) {
                const int q_hi = (int)(qs_base[l] >> 4)
                               + ((qh_base[l] & qh_mask) ? 16 : 0);
                const float w = sub_d * (float)q_hi - sub_m;
                partial += w * x_base[l];
            }
        }
        acc += partial;
    }

    acc += __shfl_xor(acc, 32);
    acc += __shfl_xor(acc, 16);
    acc += __shfl_xor(acc,  8);
    acc += __shfl_xor(acc,  4);
    acc += __shfl_xor(acc,  2);
    acc += __shfl_xor(acc,  1);

    if (lane == 0) y[row] = acc;
}
