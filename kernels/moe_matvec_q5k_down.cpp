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
// Each workgroup covers DOWN_R groups of rows_per_block rows: every
// thread issues the loads for its DOWN_R rows (clamped, not branched)
// before the first dot, and all partials reduce through LDS in one
// pass. One group per workgroup made ~1000 workgroups of a single
// short memory round trip each — launch- and latency-bound (Q5_K at
// in_dim 512: 144 GB/s). The runtime prepends `#define DOWN_R` so its
// grid and the kernel agree.
//
// grid = (ceil(out_dim / (rows_per_block * DOWN_R)), n_used, n_tok); block 256.

// No default: a kernel compiled with fewer groups than its launcher's
// grid assumes would leave rows unwritten.
#ifndef DOWN_R
#error "moe down kernel: the runtime must #define DOWN_R (see down_src)"
#endif

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

struct __attribute__((packed)) BlockQ8 {
    float  d;
    float  xsum;
    int8_t qs[32];
};
static_assert(sizeof(BlockQ8) == 40, "BlockQ8 must be 40 bytes");

// Spread 4 bits (b0..b3) to bit 4 of bytes 0..3 — one multiply, as in
// matvec_q5k_repacked.
__device__ __forceinline__ uint32_t spread4(uint32_t h) {
    return ((h & 0xFu) * 0x02040810u) & 0x10101010u;
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
    __shared__ float red[DOWN_R * 256];
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
    const unsigned int rpb     = 256u / n_sub;          // rows per group
    const unsigned int tid     = threadIdx.x;
    const unsigned int r       = tid / n_sub;
    const unsigned int sb      = tid % n_sub;
    const bool lane_ok = r < rpb;
    const unsigned int row_base = blockIdx.x * rpb * DOWN_R + r;
    const unsigned int rmax = out_dim - 1;

    const uint4*    nib = reinterpret_cast<const uint4*>(wbase);
    const uint32_t* qhp = reinterpret_cast<const uint32_t*>(
        wbase + (size_t)out_dim * nsp * 16);
#ifdef Q5_1_SCALES
    // Q5_1 (quant::q5_1::repack_for_matvec): raw fp16 d|m per sub-block.
    const uint32_t* dmp = reinterpret_cast<const uint32_t*>(
        wbase + (size_t)out_dim * nsp * 16 + (size_t)out_dim * nsp * 4);
#else
    const uint16_t* smp = reinterpret_cast<const uint16_t*>(   // v2: sc|m per sub-block
        wbase + (size_t)out_dim * nsp * 16 + (size_t)out_dim * nsp * 4);
    const uint32_t* ddp = reinterpret_cast<const uint32_t*>(   // v2: d|dmin per superblock
        wbase + (size_t)out_dim * nsp * 16 + (size_t)out_dim * nsp * 4
              + (size_t)out_dim * nsp * 2);
#endif
    const unsigned int sbc = lane_ok ? sb : 0u;                // idle lanes read sub-block 0

    // --- all loads first ---
    uint4 q[DOWN_R]; uint32_t qh[DOWN_R];
#ifdef Q5_1_SCALES
    uint32_t dm[DOWN_R];
#else
    uint16_t sm[DOWN_R]; uint32_t dd[DOWN_R];
#endif
    #pragma unroll
    for (int k = 0; k < DOWN_R; k++) {
        const unsigned int row = min(row_base + k * rpb, rmax);
        const size_t idx = (size_t)row * nsp + sbc;
        q[k]  = nib[idx];
        qh[k] = qhp[idx];
#ifdef Q5_1_SCALES
        dm[k] = dmp[idx];
#else
        sm[k] = smp[idx];
        dd[k] = ddp[(size_t)row * n_super + (sbc >> 3)];
#endif
    }
    const BlockQ8* xb   = xqs + sbc;
    const float    dx   = xb->d;
    const float    xsum = xb->xsum;
    const int*     xq32 = reinterpret_cast<const int*>(xb->qs);

    #pragma unroll
    for (int k = 0; k < DOWN_R; k++) {
#ifdef Q5_1_SCALES
        const uint16_t d_bits = (uint16_t)(dm[k] & 0xFFFF);
        const uint16_t m_bits = (uint16_t)(dm[k] >> 16);
        const float dsc  =  __half2float(*reinterpret_cast<const __half*>(&d_bits));
        const float deff = -__half2float(*reinterpret_cast<const __half*>(&m_bits));
#else
        const uint16_t d_bits    = (uint16_t)(dd[k] & 0xFFFF);
        const uint16_t dmin_bits = (uint16_t)(dd[k] >> 16);
        const float dsc  = __half2float(*reinterpret_cast<const __half*>(&d_bits))
                           * (float)(sm[k] & 0xFFu);
        const float deff = __half2float(*reinterpret_cast<const __half*>(&dmin_bits))
                           * (float)(sm[k] >> 8);
#endif
        const uint32_t qa[4] = { q[k].x, q[k].y, q[k].z, q[k].w };
        int idot = 0;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            const uint32_t lo = ( qa[j]       & 0x0F0F0F0Fu)
                | spread4(qh[k] >> (4 * (2 * j)));
            const uint32_t hi = ((qa[j] >> 4) & 0x0F0F0F0Fu)
                | spread4(qh[k] >> (4 * (2 * j + 1)));
            idot = __builtin_amdgcn_sdot4((int)lo, xq32[j],     idot, false);
            idot = __builtin_amdgcn_sdot4((int)hi, xq32[j + 4], idot, false);
        }
        red[k * 256 + tid] = dsc * dx * (float)idot - deff * xsum;
    }
    __syncthreads();

    if (lane_ok && sb == 0) {
        #pragma unroll
        for (int k = 0; k < DOWN_R; k++) {
            const unsigned int row = row_base + k * rpb;
            if (row > rmax) break;
            float acc = 0.0f;
            for (unsigned int j = 0; j < n_sub; j++) acc += red[k * 256 + r * n_sub + j];
            yo[row] = acc;
        }
    }
}
