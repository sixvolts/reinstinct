// Fused IQ4_XS dequantize + matvec.
//
// IQ4_XS layout (136 bytes / 256 weights):
//   fp16   d            super-block scale          (offset 0)
//   uint16 scales_h     2 high bits × 8 sub-blocks (offset 2)
//   uint8  scales_l[4]  4 low  bits × 8 sub-blocks (offset 4)
//   uint8  qs[128]      256 nibbles into kvalues_iq4nl[16] (offset 8)
//
// Per sub-block ib (32 weights, 16 qs bytes):
//   ls_lo = (scales_l[ib/2] >> 4*(ib&1)) & 0xF
//   ls_hi = (scales_h >> 2*ib) & 0x3
//   ls    = ls_lo | (ls_hi << 4)              ∈ [0, 63]
//   dl    = d * (ls - 32)
//   For l in [0,16):
//     w[l]      = dl * KVALUES[qs[l] & 0xF]
//     w[l + 16] = dl * KVALUES[qs[l] >> 4]
//
// Note: nibble layout is *not* the same as Q4_K. Here low/high nibbles fill
// adjacent halves of the SAME 32-weight sub-block, not two different ones.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

struct __attribute__((packed)) BlockIQ4_XS {
    uint16_t d;
    uint16_t scales_h;
    uint8_t  scales_l[4];
    uint8_t  qs[128];
};
static_assert(sizeof(BlockIQ4_XS) == 136, "BlockIQ4_XS must be 136 bytes");

// Non-uniform 4-bit codebook shared by IQ4_NL and IQ4_XS.
__device__ static const int8_t KVALUES_IQ4NL[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
       1,   13,  25,  38,  53,  69,  89, 113,
};

extern "C" __global__
void matvec_iq4_xs_f32(const BlockIQ4_XS* __restrict__ w_blocks,  // [out_dim, in_dim/256]
                       const float*       __restrict__ x,          // [in_dim]
                       float*             __restrict__ y,          // [out_dim]
                       unsigned int in_dim,
                       unsigned int out_dim)
{
    extern __shared__ float smem[];
    const int row = blockIdx.x;
    if (row >= (int)out_dim) return;
    const int tid = threadIdx.x;
    const int bs  = blockDim.x;

    const unsigned int n_blocks    = in_dim >> 8;   // /256
    const unsigned int n_subblocks = in_dim >> 5;   // /32
    const BlockIQ4_XS* row_blocks = w_blocks + (size_t)row * (size_t)n_blocks;

    float acc = 0.0f;
    for (int sb = tid; sb < (int)n_subblocks; sb += bs) {
        const int blk_idx = sb >> 3;        // /8 sub-blocks per IQ4_XS block
        const int ib      = sb & 7;         // 0..7

        const BlockIQ4_XS* blk = row_blocks + blk_idx;
        const __half d_h = *reinterpret_cast<const __half*>(&blk->d);
        const float  d   = __half2float(d_h);

        const int ls_lo = (blk->scales_l[ib >> 1] >> (4 * (ib & 1))) & 0x0F;
        const int ls_hi = (blk->scales_h >> (2 * ib)) & 0x3;
        const int ls    = ls_lo | (ls_hi << 4);
        const float dl  = d * (float)(ls - 32);

        const uint8_t* qs_base = blk->qs + ib * 16;
        const float*   x_base  = x + (size_t)sb * 32;

        float partial = 0.0f;
        #pragma unroll
        for (int l = 0; l < 16; l++) {
            const int lo = qs_base[l] & 0x0F;
            const int hi = qs_base[l] >> 4;
            partial += (float)KVALUES_IQ4NL[lo] * x_base[l]
                     + (float)KVALUES_IQ4NL[hi] * x_base[l + 16];
        }
        acc += dl * partial;
    }

    smem[tid] = acc;
    __syncthreads();
    for (int s = bs / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) y[row] = smem[0];
}
