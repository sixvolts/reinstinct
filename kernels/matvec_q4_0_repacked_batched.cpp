// Q4_0 matvec for K=2..4 activation rows (MTP spec-decode verify).
//
// The Q4_0 counterpart of matvec_q4k_repacked_batched, and it exists for
// the same reason: the plain repacked matvec is K=1 (decode) and the MMQ
// GEMM is BN=64 (prefill), so spec-decode verify sits in the gap where
// MMQ wastes ~94% of every workgroup. This reads each weight block ONCE
// and dots it against N_ROWS activation rows, amortising the weight
// stream that bounds an HBM-bound matvec at small N.
//
// Layout is identical to matvec_q4_0_repacked (see q4_0::repack_for_matvec):
//   * nibble plane — 16 bytes per block, block-major.
//   * d plane      — 2 bytes per block, fp16.
// No superblock plane and no 6-bit scale/min pair: Q4_0's per-block fp16
// scale and constant −8 offset collapse to `dw · dx · (idot − 8·xqsum)`,
// where xqsum sums the *quantized* activations — see
// matvec_q4_0_repacked.cpp for why the exact BlockQ8::xsum is wrong here.
//
// Activation `xq` is [n_rows, n_sub, BlockQ8] (40-byte BlockQ8, per-row
// per-block d/xsum/qs[32]). Output `y` is [n_rows, out_dim].
//
// Two instantiations, differing only in (ROWS, N_ROWS_MAX):
//
//   `_f32`   — ROWS=2, N_ROWS_MAX=4.  4 waves x 2 = 8 out_dim rows/WG,
//              grid.x = ceil(out_dim / 8). The tuned K<=4 spec-decode path.
//   `batched16_f32` — ROWS=1, N_ROWS_MAX=16. 4 waves x 1 = 4 out_dim rows/WG,
//              grid.x = ceil(out_dim / 4). For DFlash, whose block size is
//              16. Dropping ROWS keeps ROWS*N_ROWS_MAX accumulators at 16
//              instead of 32, so register pressure stays near the 4-row
//              kernel while the weight is still streamed once for all 16
//              activation rows.
//
// The caller must match grid.x to the variant.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include "gfx906_dpp.h"


struct __attribute__((packed)) BlockQ8 {
    float  d;
    float  xsum;
    int8_t qs[32];
};
static_assert(sizeof(BlockQ8) == 40, "BlockQ8 must be 40 bytes");

template<int ROWS, int N_ROWS_MAX>
__device__ __forceinline__
void mv_q4_0_batched_impl(const uint8_t* __restrict__ wbase,
                                      const BlockQ8* __restrict__ xq,
                                      float*         __restrict__ y,
                                      unsigned int in_dim,
                                      unsigned int out_dim,
                                      unsigned int n_rows)
{
    const int wave = threadIdx.x >> 6;          // 0..3
    const int lane = threadIdx.x & 63;
    const int row0 = blockIdx.x * (ROWS * 4) + wave * ROWS;
    const unsigned int n_sub = in_dim >> 5;
    const unsigned int nsp = ((n_sub & (n_sub - 1u)) == 0u) ? (n_sub + 1u) : n_sub;

    const uint4*    nib = reinterpret_cast<const uint4*>(wbase);
    const uint16_t* dp  = reinterpret_cast<const uint16_t*>(
        wbase + (size_t)out_dim * nsp * 16);

    // Per-row, per-batch-row accumulator. (n_rows ≤ N_ROWS_MAX is a
    // host-side invariant from MAX_VERIFY_K.)
    float acc[ROWS][N_ROWS_MAX];
    #pragma unroll
    for (int r = 0; r < ROWS; r++)
        #pragma unroll
        for (int b = 0; b < N_ROWS_MAX; b++) acc[r][b] = 0.0f;

    for (unsigned int sb = lane; sb < n_sub; sb += 64) {
        // Load the per-(sb, r) weight metadata once for ALL r up front,
        // then loop b outermost so each activation row is read exactly
        // once per sb.
        uint32_t qa_all[ROWS][4];
        float    dw_all[ROWS];
        bool     row_valid[ROWS];
        #pragma unroll
        for (int r = 0; r < ROWS; r++) {
            const int row = row0 + r;
            row_valid[r] = (row < (int)out_dim);
            if (!row_valid[r]) continue;

            const uint4    q  = nib[(size_t)row * nsp + sb];
            const uint16_t db = dp[(size_t)row * nsp + sb];
            dw_all[r] = __half2float(*reinterpret_cast<const __half*>(&db));
            qa_all[r][0] = q.x; qa_all[r][1] = q.y;
            qa_all[r][2] = q.z; qa_all[r][3] = q.w;
        }

        for (unsigned int b = 0; b < n_rows; b++) {
            const BlockQ8* xb   = xq + (size_t)b * n_sub + sb;
            const float    dx   = xb->d;
            const int*     xq32 = reinterpret_cast<const int*>(xb->qs);
            int xqsum = 0;
            #pragma unroll
            for (int g = 0; g < 8; g++)
                xqsum = __builtin_amdgcn_sdot4(0x01010101, xq32[g], xqsum, false);

            #pragma unroll
            for (int r = 0; r < ROWS; r++) {
                if (!row_valid[r]) continue;
                int idot = 0;
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    idot = __builtin_amdgcn_sdot4(
                        (int)( qa_all[r][j]       & 0x0F0F0F0Fu), xq32[j],     idot, false);
                    idot = __builtin_amdgcn_sdot4(
                        (int)((qa_all[r][j] >> 4) & 0x0F0F0F0Fu), xq32[j + 4], idot, false);
                }
                acc[r][b] += dw_all[r] * dx * (float)(idot - 8 * xqsum);
            }
        }
    }

    // Wave-local reduce + write. n_rows · ROWS reductions per WG.
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

extern "C" __global__
void matvec_q4_0_repacked_batched_f32(const uint8_t* __restrict__ wbase,
                                      const BlockQ8* __restrict__ xq,
                                      float*         __restrict__ y,
                                      unsigned int in_dim,
                                      unsigned int out_dim,
                                      unsigned int n_rows)
{
    mv_q4_0_batched_impl<2, 4>(wbase, xq, y, in_dim, out_dim, n_rows);
}

// Wide variant for DFlash, whose block size is 16. ROWS drops to 1 so a
// thread carries ROWS*N_ROWS_MAX = 16 accumulators rather than 32, keeping
// register pressure near the 4-row kernel's while reading each weight
// sub-block once for all 16 activation rows. Weight traffic is then
// out_dim*in_dim total, the same as a single matvec.
extern "C" __global__
void matvec_q4_0_repacked_batched16_f32(const uint8_t* __restrict__ wbase,
                                      const BlockQ8* __restrict__ xq,
                                      float*         __restrict__ y,
                                      unsigned int in_dim,
                                      unsigned int out_dim,
                                      unsigned int n_rows)
{
    mv_q4_0_batched_impl<1, 16>(wbase, xq, y, in_dim, out_dim, n_rows);
}
