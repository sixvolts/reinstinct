// MoE expert DOWN matvec — Q8_0 weights, row-packed for small in_dim.
//
// Q8_0 analogue of moe_matvec_q5k_down. The down projection has
// in_dim = expert_ff, so n_sub = in_dim/32 is small (gemma 26B-A4B:
// ff_exp 704 -> n_sub 22). The standard moe_matvec_q8_0_dp4a maps
// lane -> sub-block with stride 64, leaving 64 - n_sub lanes idle. Here
// the 256-thread block is mapped (thread -> row, sub-block) so every
// thread is busy: rows_per_block = 256 / n_sub, and each row's n_sub
// partials are summed through LDS. Q8_0 has no super-block scale plane,
// so n_sub need not be a power of two.
//
// grid = (ceil(out_dim / rows_per_block), n_used, n_tok); block 256.
// All current callers pass n_tok=1; see moe_matvec_q6k_dp4a.cpp
// for the n_tok / xq_tok_stride / xq_slot_stride contract.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

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
void moe_matvec_q8_0_down_f32(const unsigned char* __restrict__ slab,
                              const int*       __restrict__ ids,
                              const BlockQ8*   __restrict__ xq,
                              float*           __restrict__ y,
                              unsigned int in_dim,
                              unsigned int out_dim,
                              unsigned int bytes_per_expert,
                              unsigned int xq_tok_stride,
                              unsigned int xq_slot_stride,
                              unsigned int n_used)
{
    __shared__ float red[256];
    const int tok  = blockIdx.z;
    const int slot = blockIdx.y;
    const int eid  = ids[(size_t)tok * n_used + slot];
    const BlockQ8_0* w_blocks =
        reinterpret_cast<const BlockQ8_0*>(slab + (size_t)eid * bytes_per_expert);
    const BlockQ8* xqs = xq + (size_t)tok * xq_tok_stride
                            + (size_t)slot * xq_slot_stride;
    float* yo = y + ((size_t)tok * n_used + slot) * out_dim;

    const unsigned int n_sub = in_dim >> 5;
    const unsigned int rpb   = 256u / n_sub;          // rows per block
    const unsigned int tid   = threadIdx.x;
    const unsigned int r     = tid / n_sub;
    const unsigned int sb    = tid % n_sub;
    const unsigned int row   = blockIdx.x * rpb + r;
    const bool active = (r < rpb) && (row < out_dim);

    float contrib = 0.0f;
    if (active) {
        const BlockQ8* xb   = xqs + sb;
        const float    dx   = xb->d;
        const int*     xq32 = reinterpret_cast<const int*>(xb->qs);

        const BlockQ8_0* blk = w_blocks + (size_t)row * n_sub + sb;
        const float dw = __half2float(*reinterpret_cast<const __half*>(&blk->d));
        // qs is 2-byte aligned only — read as uint16 pairs.
        const uint16_t* qs16 = reinterpret_cast<const uint16_t*>(blk->qs);

        int idot = 0;
        #pragma unroll
        for (int g = 0; g < 8; g++) {
            const uint32_t wq = (uint32_t)qs16[2*g] | ((uint32_t)qs16[2*g+1] << 16);
            idot = __builtin_amdgcn_sdot4((int)wq, xq32[g], idot, false);
        }
        contrib = dw * dx * (float)idot;
    }

    red[tid] = contrib;
    __syncthreads();

    if (active && sb == 0) {
        float acc = 0.0f;
        for (unsigned int k = 0; k < n_sub; k++) acc += red[r * n_sub + k];
        yo[row] = acc;
    }
}
