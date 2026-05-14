// Wave-cooperative Q6_K matvec for gfx906 (Wave64).
//
// Same dequant math as matvec_q6_k.cpp, but:
//   - One *wave* (64 threads) per output row, not a 256-thread block.
//     Lower latency per row = more waves resident per CU = better occupancy
//     on the launch-bound output_proj shape (in=1024 → out=248320).
//   - Reduction uses __shfl_xor across the 64 lanes of the wavefront —
//     no shared memory, no __syncthreads.
//   - Each thread owns ONE sub-block (16 weights) per iteration. For
//     in_dim=1024 there are exactly 64 sub-blocks per row, so the work
//     is one sub-block per lane with no inner stride loop. For larger
//     in_dim the lanes stride.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

struct __attribute__((packed)) BlockQ6_K {
    uint8_t  ql[128];
    uint8_t  qh[64];
    int8_t   scales[16];
    uint16_t d;
};
static_assert(sizeof(BlockQ6_K) == 210, "BlockQ6_K must be 210 bytes");

extern "C" __global__
void matvec_q6_k_wave64_f32(const BlockQ6_K* __restrict__ w_blocks,
                            const float*     __restrict__ x,
                            float*           __restrict__ y,
                            unsigned int in_dim,
                            unsigned int out_dim)
{
    const int row = blockIdx.x;
    if (row >= (int)out_dim) return;
    const int lane = threadIdx.x;       // 0..63

    const unsigned int n_blocks    = in_dim >> 8;   // /256
    const unsigned int n_subblocks = in_dim >> 4;   // /16
    const BlockQ6_K* row_blocks = w_blocks + (size_t)row * n_blocks;

    float acc = 0.0f;
    for (int sb = lane; sb < (int)n_subblocks; sb += 64) {
        const int blk_idx     = sb >> 4;        // /16 sub-blocks per Q6_K block
        const int sb_in_blk   = sb & 15;
        const int chunk       = sb_in_blk >> 3;
        const int sb_in_chunk = sb_in_blk & 7;
        const int group       = sb_in_chunk >> 1;
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

    // Wave64 reduction via __shfl_xor — every lane ends up with the sum.
    acc += __shfl_xor(acc, 32);
    acc += __shfl_xor(acc, 16);
    acc += __shfl_xor(acc,  8);
    acc += __shfl_xor(acc,  4);
    acc += __shfl_xor(acc,  2);
    acc += __shfl_xor(acc,  1);

    if (lane == 0) y[row] = acc;
}
