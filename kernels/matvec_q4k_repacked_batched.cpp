// Q4_K matvec for K=2..8 activation rows (MTP spec-decode verify).
//
// The existing repacked-Q4_K matvec is K=1 (decode); the MMQ GEMM is
// BN=64 (prefill). Spec-decode verify sits in the K=2..8 gap and
// MMQ wastes ~94% of every workgroup at K=4. This kernel reads each
// weight sub-block ONCE and dots it against N_ROWS quantised
// activation rows, amortising the weight stream — which is exactly
// what HBM-bound matvec needs at small N.
//
// Layout is identical to matvec_q4k_repacked (see q4_k::repack_for_matvec):
//   * nibble plane — 16 bytes per sub-block, sub-block-major.
//   * scale plane — 2 bytes per sub-block (6-bit sc | m).
//   * superblock plane — 4 bytes per 256-weight superblock (fp16 d, dmin).
//
// Activation `xq` is [n_rows, n_sub, BlockQ8] (40-byte BlockQ8, per-row
// per-sub-block d/xsum/qs[32]). Output `y` is [n_rows, out_dim].
//
// 256-thread workgroup, 4 waves × ROWS=2 rows of output each (8
// out_dim rows / WG). Each thread carries ROWS·N_ROWS_MAX accumulators
// in VGPRs (32 floats max at N_ROWS=8). grid.x = ceil(out_dim / 8).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#define ROWS         2     // output rows per wavefront
#define N_ROWS_MAX   4     // K upper bound; the verify-cost-vs-K experiment
                           // (commit: HEAD~1 of this revert) confirmed K=8
                           // verify is 177ms vs K=4 108ms — linear in K with
                           // ~62ms fixed cost. Tree drafting at any reasonable
                           // shape loses against chain K=4 because the extra
                           // verify cost outweighs the token-gain from branching.
                           // Keeping the K≤4 cap to preserve the 9% occupancy
                           // win we got from N_ROWS_MAX=4.

struct __attribute__((packed)) BlockQ8 {
    float  d;
    float  xsum;
    int8_t qs[32];
};
static_assert(sizeof(BlockQ8) == 40, "BlockQ8 must be 40 bytes");

extern "C" __global__
void matvec_q4k_repacked_batched_f32(const uint8_t* __restrict__ wbase,
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
    const uint16_t* smp = reinterpret_cast<const uint16_t*>(
        wbase + (size_t)out_dim * nsp * 16);
    const uint32_t* ddp = reinterpret_cast<const uint32_t*>(
        wbase + (size_t)out_dim * nsp * 16 + (size_t)out_dim * nsp * 2);
    const unsigned int n_super = n_sub >> 3;

    // Per-row, per-batch-row accumulator. (n_rows ≤ N_ROWS_MAX is a
    // host-side invariant from MAX_VERIFY_K.)
    float acc[ROWS][N_ROWS_MAX];
    #pragma unroll
    for (int r = 0; r < ROWS; r++)
        #pragma unroll
        for (int b = 0; b < N_ROWS_MAX; b++) acc[r][b] = 0.0f;

    for (unsigned int sb = lane; sb < n_sub; sb += 64) {
        // Load the per-(sb, r) weight metadata once for ALL r values up
        // front, then loop b outermost in the dp4a phase so each
        // activation row is read exactly ONCE per sb (was ROWS times in
        // the previous structure, e.g. read twice with ROWS=2).
        uint32_t qa_all[ROWS][4];
        float    dsc_all[ROWS], deff_all[ROWS];
        bool     row_valid[ROWS];
        #pragma unroll
        for (int r = 0; r < ROWS; r++) {
            const int row = row0 + r;
            row_valid[r] = (row < (int)out_dim);
            if (!row_valid[r]) continue;

            const uint4    q  = nib[(size_t)row * nsp + sb];
            const uint16_t sm = smp[(size_t)row * nsp + sb];
            const uint32_t dd = ddp[(size_t)row * n_super + (sb >> 3)];
            const uint16_t d_bits    = (uint16_t)(dd & 0xFFFF);
            const uint16_t dmin_bits = (uint16_t)(dd >> 16);
            dsc_all[r]  = __half2float(*reinterpret_cast<const __half*>(&d_bits))
                          * (float)(sm & 0xFFu);
            deff_all[r] = __half2float(*reinterpret_cast<const __half*>(&dmin_bits))
                          * (float)(sm >> 8);
            qa_all[r][0] = q.x; qa_all[r][1] = q.y;
            qa_all[r][2] = q.z; qa_all[r][3] = q.w;
        }

        // Now loop activations once each, doing ROWS dp4a accumulations.
        // Tried staging xq into LDS once per WG so the 4 waves share the
        // load instead of each fetching the same super-blocks from HBM
        // — turned out 3-4% SLOWER because the 26 KB LDS alloc per WG
        // dropped occupancy from ~8 → 2 WGs/CU. L1 caching across waves
        // is already cheap enough that LDS sharing doesn't pay.
        for (unsigned int b = 0; b < n_rows; b++) {
            const BlockQ8* xb   = xq + (size_t)b * n_sub + sb;
            const float    dx   = xb->d;
            const float    xsum = xb->xsum;
            const int*     xq32 = reinterpret_cast<const int*>(xb->qs);

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
                acc[r][b] += dsc_all[r] * dx * (float)idot - deff_all[r] * xsum;
            }
        }
    }

    // Wave-local reduce + write. n_rows · ROWS reductions per WG.
    #pragma unroll
    for (int r = 0; r < ROWS; r++) {
        for (unsigned int b = 0; b < n_rows; b++) {
            float a = acc[r][b];
            a += __shfl_xor(a, 32);
            a += __shfl_xor(a, 16);
            a += __shfl_xor(a,  8);
            a += __shfl_xor(a,  4);
            a += __shfl_xor(a,  2);
            a += __shfl_xor(a,  1);
            if (lane == 0 && (row0 + r) < (int)out_dim) {
                y[(size_t)b * out_dim + (row0 + r)] = a;
            }
        }
    }
}

