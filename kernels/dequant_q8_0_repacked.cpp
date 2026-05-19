// Bulk Q8_0 → fp16 dequant for the repacked two-plane layout used by
// matvec_q8_0_repacked. Mirrors dequant_q8_0_f16 but reads from the
// separated qs / d planes (no struct stride).
//
// Layout (matches src/quant/q8_0.rs::repack_for_matvec):
//   slab + 0                    : out_dim × nsp × 32  qs bytes
//   slab + out_dim × nsp × 32   : out_dim × nsp × 2   fp16 d
//   nsp = (n_blocks is pow2) ? n_blocks + 1 : n_blocks
//
// Output is dense [out_dim × in_dim] fp16 (no padding) — the padded
// sub-block is skipped during the row→nsp index calculation.
//
// grid = (out_dim × n_blocks_per_row,); block = 32. Same launch shape
// as dequant_q{4,5,6}k_repacked so the prefill dispatcher can reuse
// the 32-thread / one-block-per-sub-block convention.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

extern "C" __global__
void dequant_q8_0_repacked_f16(const unsigned char* __restrict__ slab,
                               __half*              __restrict__ out,
                               unsigned int in_dim,
                               unsigned int out_dim)
{
    const unsigned int n_blocks = in_dim >> 5;
    const unsigned int nsp = ((n_blocks & (n_blocks - 1u)) == 0u)
                             ? (n_blocks + 1u) : n_blocks;
    const unsigned int idx = blockIdx.x;
    const unsigned int row = idx / n_blocks;
    const unsigned int blk = idx - row * n_blocks;
    if (row >= out_dim) return;
    const int i = (int)threadIdx.x;
    if (i >= 32) return;

    const int8_t*   qs_plane = reinterpret_cast<const int8_t*>(slab);
    const uint16_t* d_plane  = reinterpret_cast<const uint16_t*>(
        slab + (size_t)out_dim * nsp * 32);

    const size_t pidx = (size_t)row * nsp + blk;
    const int8_t   q = qs_plane[pidx * 32 + i];
    const uint16_t db = d_plane[pidx];
    const float    d  = __half2float(*reinterpret_cast<const __half*>(&db));

    out[(size_t)row * in_dim + blk * 32 + i] = __float2half(d * (float)q);
}
