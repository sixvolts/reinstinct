// IQ4_XS batched matvec — derived from the Q4_0 kernel of the same name.
//
// The repacked layout is byte-identical to Q4_0's (quant::iq4_xs::
// repack_for_matvec): a 16-byte nibble plane per 32-weight sub-block and
// an fp16 scale plane, here with the super-block d and 6-bit sub-scale
// pre-folded into `dl = d * (ls - 32)`. Two differences from Q4_0:
//   * each nibble maps through the 16-entry IQ4_NL codebook before the
//     dot, instead of the linear `n - 8`;
//   * the codebook values are already signed, so there is no constant
//     offset and no `8 * xqsum` term.
// Everything else — tiling, launch contract, entry points — is Q4_0's.
//
// Layout:
//   slab + 0                    : out_dim × nsp × 16  nibble bytes
//   slab + out_dim × nsp × 16   : out_dim × nsp × 2   fp16 dl
//   nsp = (n_sub power of two) ? n_sub + 1 : n_sub   // anti-alias

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include "gfx906_dpp.h"


// IQ4_NL / IQ4_XS codebook — the 16 int8 values a nibble indexes, packed
// four per 32-bit word (little-endian): entries 0-3, 4-7, 8-11, 12-15.
//   {-127,-104,-83,-65,-49,-35,-22,-10, 1,13,25,38,53,69,89,113}
// Overridable: IQ3_S repacks into this same layout with the codebook
// {-15,-13,...,15} and compiles this source with the four words
// predefined (quant::iq3_s::kernel_source).
#ifndef IQ4NL_KV_0_3
#define IQ4NL_KV_0_3   0xbfad9881u
#define IQ4NL_KV_4_7   0xf6eaddcfu
#define IQ4NL_KV_8_11  0x26190d01u
#define IQ4NL_KV_12_15 0x71594535u
#endif

// Map 4 nibbles (one per byte of `n4`, masked to 0x0F0F0F0F) to 4 int8
// codebook values packed for sdot4 — in registers, via v_perm_b32.
//
// A scalar table lookup here was catastrophic: 8 constant-memory byte
// loads per sdot4 inside the compute-bound MMQ loop made prefill 5.8x
// slower than the Q8_0 transcode it was meant to beat. v_perm selects
// bytes from an 8-byte pair per selector byte, so the low 3 bits of each
// nibble pick within a half-table, and bit 3 chooses which half.
__device__ __forceinline__ int iq4xs_lut4(uint32_t n4) {
    const uint32_t sel = n4 & 0x07070707u;
    // perm(a, b, sel): selector 0-3 -> bytes of b, 4-7 -> bytes of a.
    const uint32_t lo = __builtin_amdgcn_perm(IQ4NL_KV_4_7,   IQ4NL_KV_0_3,  sel);
    const uint32_t hi = __builtin_amdgcn_perm(IQ4NL_KV_12_15, IQ4NL_KV_8_11, sel);
    const uint32_t m  = ((n4 >> 3) & 0x01010101u) * 0xFFu;   // 0xFF where nibble >= 8
    return (int)((hi & m) | (lo & ~m));
}

struct __attribute__((packed)) BlockQ8 {
    float  d;
    float  xsum;
    int8_t qs[32];
};
static_assert(sizeof(BlockQ8) == 40, "BlockQ8 must be 40 bytes");

template<int ROWS, int N_ROWS_MAX>
__device__ __forceinline__
void mv_iq4xs_batched_impl(const uint8_t* __restrict__ wbase,
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

            #pragma unroll
            for (int r = 0; r < ROWS; r++) {
                if (!row_valid[r]) continue;
                int idot = 0;
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    idot = __builtin_amdgcn_sdot4(
                        iq4xs_lut4(qa_all[r][j] & 0x0F0F0F0Fu), xq32[j],     idot, false);
                    idot = __builtin_amdgcn_sdot4(
                        iq4xs_lut4((qa_all[r][j] >> 4) & 0x0F0F0F0Fu), xq32[j + 4], idot, false);
                }
                acc[r][b] += dw_all[r] * dx * (float)idot;
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
void matvec_iq4xs_repacked_batched_f32(const uint8_t* __restrict__ wbase,
                                      const BlockQ8* __restrict__ xq,
                                      float*         __restrict__ y,
                                      unsigned int in_dim,
                                      unsigned int out_dim,
                                      unsigned int n_rows)
{
    mv_iq4xs_batched_impl<2, 4>(wbase, xq, y, in_dim, out_dim, n_rows);
}

// Wide variant for DFlash, whose block size is 16. ROWS drops to 1 so a
// thread carries ROWS*N_ROWS_MAX = 16 accumulators rather than 32, keeping
// register pressure near the 4-row kernel's while reading each weight
// sub-block once for all 16 activation rows. Weight traffic is then
// out_dim*in_dim total, the same as a single matvec.
extern "C" __global__
void matvec_iq4xs_repacked_batched16_f32(const uint8_t* __restrict__ wbase,
                                      const BlockQ8* __restrict__ xq,
                                      float*         __restrict__ y,
                                      unsigned int in_dim,
                                      unsigned int out_dim,
                                      unsigned int n_rows)
{
    mv_iq4xs_batched_impl<1, 16>(wbase, xq, y, in_dim, out_dim, n_rows);
}
