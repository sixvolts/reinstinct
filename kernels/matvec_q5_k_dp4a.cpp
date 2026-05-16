// Q5_K matvec via gfx906's v_dot4_i32_i8 (dp4a).
//
// Like matvec_q4_k_dp4a but the 5th bit comes from the qh array: per
// 4-weight group the packed 5-bit weights (0..31) are assembled from
// the low nibbles plus the qh bit, then dotted with the int8 activation.
//
//   dot = sum_sub [ (d_w·sc)·d_x·<q5·int8> - (dmin_w·m)·xsum ]
//
// ROWS=2 output rows per wavefront; grid = ceil(out_dim/ROWS); block = 64.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#define ROWS 2

struct __attribute__((packed)) BlockQ5_K {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[12];
    uint8_t  qh[32];
    uint8_t  qs[128];
};
static_assert(sizeof(BlockQ5_K) == 176, "BlockQ5_K must be 176 bytes");

struct __attribute__((packed)) BlockQ8 {
    float  d;
    float  xsum;
    int8_t qs[32];
};
static_assert(sizeof(BlockQ8) == 40, "BlockQ8 must be 40 bytes");

__device__ __forceinline__
void gsm_q5k_dp(int j, const uint8_t* q, uint8_t& sc, uint8_t& m) {
    if (j < 4) { sc = q[j] & 63; m = q[j + 4] & 63; }
    else {
        sc = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        m  = (q[j + 4] >> 4)   | ((q[j]     >> 6) << 4);
    }
}

extern "C" __global__
void matvec_q5_k_dp4a_f32(const BlockQ5_K* __restrict__ w_blocks,
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
        const int qhbitpos     = chunk * 2 + is;
        const int shift        = (is == 0) ? 0 : 4;

        const BlockQ8* xb = xq + sb;
        const float dx   = xb->d;
        const float xsum = xb->xsum;
        const int*  xq32 = reinterpret_cast<const int*>(xb->qs);

        #pragma unroll
        for (int r = 0; r < ROWS; r++) {
            const int row = row0 + r;
            if (row >= (int)out_dim) continue;
            const BlockQ5_K* blk = w_blocks + (size_t)row * n_blocks + blk_idx;

            const float d_w    = __half2float(*reinterpret_cast<const __half*>(&blk->d));
            const float dmin_w = __half2float(*reinterpret_cast<const __half*>(&blk->dmin));
            uint8_t sc, m;
            gsm_q5k_dp(sub_in_block, blk->scales, sc, m);

            const uint32_t* qs32 =
                reinterpret_cast<const uint32_t*>(blk->qs + qs_off);
            const uint32_t* qh32 =
                reinterpret_cast<const uint32_t*>(blk->qh);

            int idot = 0;
            #pragma unroll
            for (int g = 0; g < 8; g++) {
                const uint32_t nib = (qs32[g] >> shift) & 0x0F0F0F0Fu;
                const uint32_t hib =
                    ((qh32[g] >> qhbitpos) & 0x01010101u) << 4;
                const uint32_t q5 = nib | hib;          // 4 bytes, 0..31
                idot = __builtin_amdgcn_sdot4((int)q5, xq32[g], idot, false);
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
