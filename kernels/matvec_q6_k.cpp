// Fused Q6_K dequantize + matvec.
//
// Q6_K layout (210 bytes / 256 weights):
//   uint8 ql[128]      lower 4 bits of each 6-bit quant
//   uint8 qh[64]       upper 2 bits, four 2-bit pairs per byte
//   int8  scales[16]   signed sub-block scales (16 sub-blocks × 16 weights)
//   fp16  d            super-block scale  (offset 208)
//
// Per weight: w = d * scales[sb] * (q6 - 32)  where q6 ∈ [0,63] unsigned.
// Symmetric — no `dmin`.
//
// Within a 256-weight block the dequant walks 2 outer chunks of 128
// weights, 4 groups per chunk, 32 weights per group. The 4 groups in a
// chunk pull from the same `qh[chunk*32 + l]` byte using shifts 0, 2, 4, 6.
// Output positions in chunk c, group g, half is, lane l ∈ [0,16):
//   y[c*128 + g*32 + is*16 + l]
//
// Each thread handles one 16-weight sub-block per iteration; consecutive
// sub-blocks (in output order) all share one scales[] entry.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

struct __attribute__((packed)) BlockQ6_K {
    uint8_t  ql[128];
    uint8_t  qh[64];
    int8_t   scales[16];
    uint16_t d;            // fp16 raw bits
};
static_assert(sizeof(BlockQ6_K) == 210, "BlockQ6_K must be 210 bytes");

extern "C" __global__
void matvec_q6_k_f32(const BlockQ6_K* __restrict__ w_blocks,  // [out_dim, in_dim/256]
                     const float*     __restrict__ x,          // [in_dim]
                     float*           __restrict__ y,          // [out_dim]
                     unsigned int in_dim,
                     unsigned int out_dim)
{
    extern __shared__ float smem[];
    const int row = blockIdx.x;
    if (row >= (int)out_dim) return;
    const int tid = threadIdx.x;
    const int bs  = blockDim.x;

    const unsigned int n_blocks    = in_dim >> 8;   // /256
    const unsigned int n_subblocks = in_dim >> 4;   // /16
    const BlockQ6_K* row_blocks = w_blocks + (size_t)row * (size_t)n_blocks;

    float acc = 0.0f;
    for (int sb = tid; sb < (int)n_subblocks; sb += bs) {
        const int blk_idx     = sb >> 4;        // /16 sub-blocks per Q6_K block
        const int sb_in_blk   = sb & 15;        // 0..15
        const int chunk       = sb_in_blk >> 3; // 0 or 1
        const int sb_in_chunk = sb_in_blk & 7;  // 0..7
        const int group       = sb_in_chunk >> 1; // 0..3
        const int is          = sb_in_chunk & 1;
        const int sc_idx      = chunk * 8 + is + group * 2;
        const int l_start     = is * 16;
        const int qh_shift    = group * 2;
        const int ql_off_blk  = chunk * 64 + (group & 1) * 32;
        const int ql_high     = group >> 1;
        const int qh_off_blk  = chunk * 32;

        const BlockQ6_K* blk = row_blocks + blk_idx;
        const __half d_h = *reinterpret_cast<const __half*>(&blk->d);
        const float  d   = __half2float(d_h);
        const float  ds  = d * (float)blk->scales[sc_idx];

        const float* x_base = x + (size_t)sb * 16;
        float partial = 0.0f;
        #pragma unroll
        for (int li = 0; li < 16; li++) {
            const int l = l_start + li;
            const uint8_t ql_byte = blk->ql[ql_off_blk + l];
            const uint8_t ql_nib  = ql_high ? (ql_byte >> 4) : (ql_byte & 0x0F);
            const uint8_t qh_byte = blk->qh[qh_off_blk + l];
            const uint8_t qh_pair = (qh_byte >> qh_shift) & 0x3;
            const int q = (int)(ql_nib | (qh_pair << 4)) - 32;
            partial += (float)q * x_base[li];
        }
        acc += ds * partial;
    }

    smem[tid] = acc;
    __syncthreads();
    for (int s = bs / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) y[row] = smem[0];
}
