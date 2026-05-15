// Row-blocked Q4_K matvec for gfx906.
//
// The wave64 matvec processes one output row per wavefront: it issues a
// load, consumes it, issues the next — memory latency barely hidden, so
// it sustains only ~13% of HBM bandwidth on Gemma's large matmuls.
//
// This variant gives each wavefront ROWS=8 output rows. The inner loop
// is over sub-blocks; for each sub-block every lane touches all 8 rows
// before consuming any result, so 8 independent global loads are in
// flight at once — an 8-deep memory pipeline per lane. The activation
// slice x[sb·32 .. +32] is loaded once and reused across the 8 rows.
//
// The 32 packed weight nibbles are read as 8×uint32 (not 32 byte loads)
// and the 32 activations as 8×float4 — the byte-at-a-time version was
// load-issue bound, not bandwidth bound.
//
// grid = ceil(out_dim / 8); block = 64 (one wavefront).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#define ROWS 8

struct __attribute__((packed)) BlockQ4_K {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[12];
    uint8_t  qs[128];
};
static_assert(sizeof(BlockQ4_K) == 144, "BlockQ4_K must be 144 bytes");

__device__ __forceinline__
void gsm_q4k_rb(int j, const uint8_t* q, uint8_t& sc, uint8_t& m) {
    if (j < 4) { sc = q[j] & 63; m = q[j + 4] & 63; }
    else {
        sc = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        m  = (q[j + 4] >> 4)   | ((q[j]     >> 6) << 4);
    }
}

extern "C" __global__
void matvec_q4_k_rowblock_f32(const BlockQ4_K* __restrict__ w_blocks,
                              const float*     __restrict__ x,
                              float*           __restrict__ y,
                              unsigned int in_dim,
                              unsigned int out_dim)
{
    const int row0 = blockIdx.x * ROWS;
    const int lane = threadIdx.x;          // 0..63
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
        const float* x_base    = x + (size_t)sb * 32;

        // Process all ROWS rows for this sub-block. The per-row block
        // pointers are far apart, so these loads pipeline.
        #pragma unroll
        for (int r = 0; r < ROWS; r++) {
            const int row = row0 + r;
            if (row >= (int)out_dim) continue;
            const BlockQ4_K* blk = w_blocks + (size_t)row * n_blocks + blk_idx;

            const __half d_h  = *reinterpret_cast<const __half*>(&blk->d);
            const __half dm_h = *reinterpret_cast<const __half*>(&blk->dmin);
            const float  d    = __half2float(d_h);
            const float  dmin = __half2float(dm_h);
            uint8_t sc, m;
            gsm_q4k_rb(sub_in_block, blk->scales, sc, m);
            const float sub_d = d    * (float)sc;
            const float sub_m = dmin * (float)m;
            // qs is 4-byte aligned (struct offset 16, stride 144,
            // qs_off a multiple of 32) — read 32 nibbles as 8×uint32.
            const uint32_t* qs32 =
                reinterpret_cast<const uint32_t*>(blk->qs + qs_off);
            const int shift = (is == 0) ? 0 : 4;

            float partial = 0.0f;
            #pragma unroll
            for (int g = 0; g < 8; g++) {
                const uint32_t packed = qs32[g];
                const float4   xv     = *reinterpret_cast<const float4*>(x_base + g * 4);
                const float q0 = (float)((packed >>  shift)        & 0x0F);
                const float q1 = (float)((packed >> ( 8 + shift))  & 0x0F);
                const float q2 = (float)((packed >> (16 + shift))  & 0x0F);
                const float q3 = (float)((packed >> (24 + shift))  & 0x0F);
                partial += (sub_d * q0 - sub_m) * xv.x;
                partial += (sub_d * q1 - sub_m) * xv.y;
                partial += (sub_d * q2 - sub_m) * xv.z;
                partial += (sub_d * q3 - sub_m) * xv.w;
            }
            acc[r] += partial;
        }
    }

    // Per-row wavefront reduction.
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
