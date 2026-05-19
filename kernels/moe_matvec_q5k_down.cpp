// MoE expert DOWN matvec — repacked Q5_K, row-packed for small in_dim.
//
// The down projection has in_dim = expert_ff (~512), so n_sub = 16. The
// standard moe_matvec maps lane -> sub-block with stride 64, leaving 48
// of every 64 lanes idle — ~4x worse memory-level parallelism, which
// profiling showed costs ~12% of MoE decode. Here the 256-thread block
// is mapped (thread -> row, sub-block) so every thread is busy:
// rows_per_block = 256 / n_sub, and each row's n_sub partials are
// summed through LDS.
//
// grid = (ceil(out_dim / rows_per_block), n_used, n_tok); block 256.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

struct __attribute__((packed)) BlockQ8 {
    float  d;
    float  xsum;
    int8_t qs[32];
};
static_assert(sizeof(BlockQ8) == 40, "BlockQ8 must be 40 bytes");

// Spread 4 bits (b0..b3) to bit 4 of bytes 0..3.
__device__ __forceinline__ uint32_t spread4(uint32_t h) {
    return ((h & 1u) << 4) | ((h & 2u) << 11) | ((h & 4u) << 18) | ((h & 8u) << 25);
}

extern "C" __global__
void moe_matvec_q5k_down_f32(const unsigned char* __restrict__ slab,
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
    const uint8_t* wbase = slab + (size_t)eid * bytes_per_expert;
    const BlockQ8* xqs = xq + (size_t)tok * xq_tok_stride
                            + (size_t)slot * xq_slot_stride;
    float* yo = y + ((size_t)tok * n_used + slot) * out_dim;

    const unsigned int n_sub   = in_dim >> 5;
    const unsigned int nsp     = ((n_sub & (n_sub - 1u)) == 0u) ? (n_sub + 1u) : n_sub;
    const unsigned int n_super = n_sub >> 3;
    const unsigned int rpb     = 256u / n_sub;          // rows per block
    const unsigned int tid     = threadIdx.x;
    const unsigned int r       = tid / n_sub;
    const unsigned int sb      = tid % n_sub;
    const unsigned int row     = blockIdx.x * rpb + r;
    const bool active = (r < rpb) && (row < out_dim);

    const uint4*    nib = reinterpret_cast<const uint4*>(wbase);
    const uint32_t* qhp = reinterpret_cast<const uint32_t*>(
        wbase + (size_t)out_dim * nsp * 16);
    const uint16_t* smp = reinterpret_cast<const uint16_t*>(
        wbase + (size_t)out_dim * nsp * 16 + (size_t)out_dim * nsp * 4);
    const uint32_t* ddp = reinterpret_cast<const uint32_t*>(
        wbase + (size_t)out_dim * nsp * 16 + (size_t)out_dim * nsp * 4
              + (size_t)out_dim * nsp * 2);

    float contrib = 0.0f;
    if (active) {
        const BlockQ8* xb   = xqs + sb;
        const float    dx   = xb->d;
        const float    xsum = xb->xsum;
        const int*     xq32 = reinterpret_cast<const int*>(xb->qs);

        const uint4    q  = nib[(size_t)row * nsp + sb];
        const uint32_t qh = qhp[(size_t)row * nsp + sb];
        const uint16_t sm = smp[(size_t)row * nsp + sb];
        const uint32_t dd = ddp[(size_t)row * n_super + (sb >> 3)];
        const uint16_t d_bits    = (uint16_t)(dd & 0xFFFF);
        const uint16_t dmin_bits = (uint16_t)(dd >> 16);
        const float dsc  = __half2float(*reinterpret_cast<const __half*>(&d_bits))
                           * (float)(sm & 0xFFu);
        const float deff = __half2float(*reinterpret_cast<const __half*>(&dmin_bits))
                           * (float)(sm >> 8);

        const uint32_t qa[4] = { q.x, q.y, q.z, q.w };
        int idot = 0;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            const uint32_t lo = ( qa[j]       & 0x0F0F0F0Fu)
                | spread4((qh >> (4 * (2 * j)))     & 0xFu);
            const uint32_t hi = ((qa[j] >> 4) & 0x0F0F0F0Fu)
                | spread4((qh >> (4 * (2 * j + 1))) & 0xFu);
            idot = __builtin_amdgcn_sdot4((int)lo, xq32[j],     idot, false);
            idot = __builtin_amdgcn_sdot4((int)hi, xq32[j + 4], idot, false);
        }
        contrib = dsc * dx * (float)idot - deff * xsum;
    }

    red[tid] = contrib;
    __syncthreads();

    if (active && sb == 0) {
        float acc = 0.0f;
        for (unsigned int k = 0; k < n_sub; k++) acc += red[r * n_sub + k];
        yo[row] = acc;
    }
}
