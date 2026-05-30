// Q5_K dp4a matvec for K=2..4 activation rows — batched variant of
// matvec_q5_k_dp4a, used by spec-decode verify's lm_head (1 weight @
// [hidden, vocab] read against K batched logits rows).
//
// Same dp4a math as the K=1 version, but reads each weight superblock
// once and dots it against all N input rows. Per-thread accumulator is
// [ROWS][N_ROWS_MAX] floats. Output `y` is [n_rows, out_dim] strided.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include "gfx906_dpp.h"

#define ROWS         2     // output rows per wavefront
#define N_ROWS_MAX   4     // batch upper bound. Kept symmetric with the
                           // per-layer batched kquant kernels (see
                           // matvec_q4k_repacked_batched.cpp); going to 8
                           // here doesn't help in practice because the
                           // per-layer GEMMs cap K at 4 anyway — verify(K≥5)
                           // falls through to MMQ in prefill.rs and that's
                           // ~3.3× slower than batched, swamping any LM-head
                           // win. Bump both together if you ever revisit.

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
void gsm_q5k_dpb(int j, const uint8_t* q, uint8_t& sc, uint8_t& m) {
    if (j < 4) { sc = q[j] & 63; m = q[j + 4] & 63; }
    else {
        sc = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        m  = (q[j + 4] >> 4)   | ((q[j]     >> 6) << 4);
    }
}

extern "C" __global__
void matvec_q5_k_dp4a_batched_f32(const BlockQ5_K* __restrict__ w_blocks,
                                  const BlockQ8*   __restrict__ xq,
                                  float*           __restrict__ y,
                                  unsigned int in_dim,
                                  unsigned int out_dim,
                                  unsigned int n_rows)
{
    const int row0 = blockIdx.x * ROWS;
    const int lane = threadIdx.x;
    const unsigned int n_blocks    = in_dim >> 8;
    const unsigned int n_subblocks = in_dim >> 5;

    float acc[ROWS][N_ROWS_MAX];
    #pragma unroll
    for (int r = 0; r < ROWS; r++)
        #pragma unroll
        for (int b = 0; b < N_ROWS_MAX; b++) acc[r][b] = 0.0f;

    for (int sb = lane; sb < (int)n_subblocks; sb += 64) {
        const int blk_idx      = sb >> 3;
        const int sub_in_block = sb & 7;
        const int chunk        = sub_in_block >> 1;
        const int is           = sub_in_block & 1;
        const int qs_off       = chunk * 32;
        const int qhbitpos     = chunk * 2 + is;
        const int shift        = (is == 0) ? 0 : 4;

        // Loop-swap: hoist per-(sb, r) q5 chunks + scales, then dp4a
        // phase reads each activation row once per (sb, b). At ROWS=2
        // the per-thread state stays small enough not to blow occupancy.
        uint32_t q5buf_all[ROWS][8];
        float    dsc_all[ROWS], deff_all[ROWS];
        bool     row_valid[ROWS];
        #pragma unroll
        for (int r = 0; r < ROWS; r++) {
            const int row = row0 + r;
            row_valid[r] = (row < (int)out_dim);
            if (!row_valid[r]) continue;

            const BlockQ5_K* blk = w_blocks + (size_t)row * n_blocks + blk_idx;
            const float d_w    = __half2float(*reinterpret_cast<const __half*>(&blk->d));
            const float dmin_w = __half2float(*reinterpret_cast<const __half*>(&blk->dmin));
            uint8_t sc, m;
            gsm_q5k_dpb(sub_in_block, blk->scales, sc, m);
            dsc_all[r]  = d_w    * (float)sc;
            deff_all[r] = dmin_w * (float)m;

            const uint32_t* qs32 =
                reinterpret_cast<const uint32_t*>(blk->qs + qs_off);
            const uint32_t* qh32 =
                reinterpret_cast<const uint32_t*>(blk->qh);
            #pragma unroll
            for (int g = 0; g < 8; g++) {
                const uint32_t nib = (qs32[g] >> shift) & 0x0F0F0F0Fu;
                const uint32_t hib = ((qh32[g] >> qhbitpos) & 0x01010101u) << 4;
                q5buf_all[r][g] = nib | hib;
            }
        }

        for (unsigned int b = 0; b < n_rows; b++) {
            const BlockQ8* xb = xq + (size_t)b * n_subblocks + sb;
            const float dx   = xb->d;
            const float xsum = xb->xsum;
            const int*  xq32 = reinterpret_cast<const int*>(xb->qs);

            #pragma unroll
            for (int r = 0; r < ROWS; r++) {
                if (!row_valid[r]) continue;
                int idot = 0;
                #pragma unroll
                for (int g = 0; g < 8; g++) {
                    idot = __builtin_amdgcn_sdot4((int)q5buf_all[r][g], xq32[g], idot, false);
                }
                acc[r][b] += dsc_all[r] * dx * (float)idot - deff_all[r] * xsum;
            }
        }
    }

    #pragma unroll
    for (int r = 0; r < ROWS; r++) {
        for (unsigned int b = 0; b < n_rows; b++) {
            float a = acc[r][b];
            a = wave64_reduce_add_f32(a);
            if (lane == 0 && (row0 + r) < (int)out_dim) {
                y[(size_t)b * out_dim + (row0 + r)] = a;
            }
        }
    }
}
