// Q4_0 matvec via v_dot4_i32_i8, repacked two-plane weight layout (see
// quant::q4_0::repack_for_matvec).
//
// Same shape as the Q4_K kernel, minus the scale machinery. Q4_0 carries
// one fp16 scale per 32-weight block and a constant −8 offset, so where
// Q4_K unpacks a 6-bit (sc, m) pair and forms dsc = d·sc / deff = dmin·m
// per sub-block, this reads one fp16 and folds the constant offset into
// the integer accumulator:
//
//   w_i = d·(q_i − 8),  q_i an unsigned nibble
//   sum_i w_i·x_i = d·dx·sum_i (q_i − 8)·xq_i
//
// The −8 is applied against the *quantized* activation sum, not the
// exact one BlockQ8::xsum carries. That matters: writing this as
// `dx·idot − 8·xsum` mixes quantized activations in the dot with exact
// ones in the offset, so each activation's quantization error enters
// weighted by q_i ∈ [0,15] — mean 7.5 — instead of cancelling. Measured
// on the oracle test that costs 40× accuracy (6.5e-3 vs 1.6e-4 rel_l2),
// which compounds over 60 layers into visibly degenerate output. Summing
// the int8 activations with a second sdot4 against a vector of ones keeps
// both terms in the same quantized domain, so the error is weighted by
// (q_i − 8) — zero mean — and cancels. The extra sdot4s are hoisted out
// of the row loop and this kernel is HBM-bound, so they are free.
//
// The nibbles need no permutation on the way in: Q4_0 already stores
// byte k as weight k (low) and weight k+16 (high), which is exactly the
// order uint32 j must carry for sdot4 — so the repack is a plain copy
// and this inner loop is character-for-character the Q4_K one.
//
// Layout:
//   slab + 0                    : out_dim × nsp × 16  nibble bytes
//   slab + out_dim × nsp × 16   : out_dim × nsp × 2   fp16 d
//   nsp = (n_blocks power of two) ? n_blocks + 1 : n_blocks   // anti-alias
//
// Lane l reads block l's 16 bytes (one uint4) — consecutive lanes,
// consecutive memory — so the weight sweep is fully coalesced.
//
// Entry points mirror the Q4_K set; the caller picks per matvec (see
// gemma4 / qwen35 launch_matvec):
//   * `_f32`    — 4 waves, ROWS=2, grid = ceil(out_dim/8). Default.
//   * `_r1_f32` — 4 waves, ROWS=1, grid = ceil(out_dim/4). Wins at small
//                 out_dim, where ROWS=2 leaves the CUs under-filled.
//   * `_r4_f32` — 4 waves, ROWS=4, grid = ceil(out_dim/16).

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

template<int ROWS>
__device__ __forceinline__
void mv_q4_0_repacked(const uint8_t* __restrict__ wbase,
                      const BlockQ8* __restrict__ xq,
                      float*         __restrict__ y,
                      unsigned int in_dim,
                      unsigned int out_dim)
{
    const int wave = threadIdx.x >> 6;          // 0..3
    const int lane = threadIdx.x & 63;
    const int row0 = blockIdx.x * (ROWS * 4) + wave * ROWS;
    const unsigned int n_sub = in_dim >> 5;
    // Must match quant::q4_0::repacked_n_sub_padded.
    const unsigned int nsp = ((n_sub & (n_sub - 1u)) == 0u) ? (n_sub + 1u) : n_sub;

    const uint4*    nib = reinterpret_cast<const uint4*>(wbase);
    const uint16_t* dp  = reinterpret_cast<const uint16_t*>(
        wbase + (size_t)out_dim * nsp * 16);

    float acc[ROWS];
    #pragma unroll
    for (int r = 0; r < ROWS; r++) acc[r] = 0.0f;

    for (unsigned int sb = lane; sb < n_sub; sb += 64) {
        const BlockQ8* xb   = xq + sb;
        const float    dx   = xb->d;
        const int*     xq32 = reinterpret_cast<const int*>(xb->qs);
        // sum of the 32 int8 activations — row-independent, so hoisted.
        int xqsum = 0;
        #pragma unroll
        for (int g = 0; g < 8; g++)
            xqsum = __builtin_amdgcn_sdot4(0x01010101, xq32[g], xqsum, false);

        #pragma unroll
        for (int r = 0; r < ROWS; r++) {
            const int row = row0 + r;
            if (row >= (int)out_dim) continue;

            const uint4    q  = nib[(size_t)row * nsp + sb];
            const uint16_t db = dp[(size_t)row * nsp + sb];
            const float    dw = __half2float(*reinterpret_cast<const __half*>(&db));

            const uint32_t qa[4] = { q.x, q.y, q.z, q.w };
            int idot = 0;
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                idot = __builtin_amdgcn_sdot4((int)( qa[j]       & 0x0F0F0F0Fu),
                                              xq32[j],     idot, false);
                idot = __builtin_amdgcn_sdot4((int)((qa[j] >> 4) & 0x0F0F0F0Fu),
                                              xq32[j + 4], idot, false);
            }
            // One integer accumulator for the whole block: the −8 offset
            // applies to all 32 weights, so it collapses to 8·xqsum.
            acc[r] += dw * dx * (float)(idot - 8 * xqsum);
        }
    }

    #pragma unroll
    for (int r = 0; r < ROWS; r++) {
        float a = wave64_reduce_add_f32(acc[r]);
        if (lane == 0 && (row0 + r) < (int)out_dim) y[row0 + r] = a;
    }
}

extern "C" __global__
void matvec_q4_0_repacked_f32(const uint8_t* __restrict__ wbase,
                              const BlockQ8* __restrict__ xq,
                              float*         __restrict__ y,
                              unsigned int in_dim,
                              unsigned int out_dim)
{
    mv_q4_0_repacked<2>(wbase, xq, y, in_dim, out_dim);
}

extern "C" __global__
void matvec_q4_0_repacked_r1_f32(const uint8_t* __restrict__ wbase,
                                 const BlockQ8* __restrict__ xq,
                                 float*         __restrict__ y,
                                 unsigned int in_dim,
                                 unsigned int out_dim)
{
    mv_q4_0_repacked<1>(wbase, xq, y, in_dim, out_dim);
}

extern "C" __global__
void matvec_q4_0_repacked_r4_f32(const uint8_t* __restrict__ wbase,
                                 const BlockQ8* __restrict__ xq,
                                 float*         __restrict__ y,
                                 unsigned int in_dim,
                                 unsigned int out_dim)
{
    mv_q4_0_repacked<4>(wbase, xq, y, in_dim, out_dim);
}
