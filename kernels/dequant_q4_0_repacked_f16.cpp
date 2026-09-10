// Bulk Q4_0 → fp16 dequant for the repacked two-plane layout used by
// matvec_q4_0_repacked. Mirrors dequant_q4_0_f16 but reads from the
// separated nibble / d planes (no 18-byte struct stride).
//
// Layout (matches src/quant/q4_0.rs::repack_for_matvec):
//   slab + 0                    : out_dim × nsp × 16  nibble bytes
//   slab + out_dim × nsp × 16   : out_dim × nsp × 2   fp16 d
//   nsp = (n_blocks is pow2) ? n_blocks + 1 : n_blocks
//
// Output is dense [out_dim × in_dim] fp16 (no padding) — the padded
// block is skipped by the row→nsp index calculation.
//
// grid = (out_dim × n_blocks_per_row,); block = 32, matching the
// one-block-per-sub-block convention the prefill dispatcher uses for
// every other repacked dequant.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

extern "C" __global__
void dequant_q4_0_repacked_f16(const unsigned char* __restrict__ slab,
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

    const uint8_t*  nib_plane = reinterpret_cast<const uint8_t*>(slab);
    const uint16_t* d_plane   = reinterpret_cast<const uint16_t*>(
        slab + (size_t)out_dim * nsp * 16);

    const size_t pidx = (size_t)row * nsp + blk;
    const uint16_t db = d_plane[pidx];
    const float    d  = __half2float(*reinterpret_cast<const __half*>(&db));

    const int is_high = i >> 4;
    const int k       = i & 15;
    const uint8_t byte = nib_plane[pidx * 16 + k];
    const int nib = is_high ? (byte >> 4) : (byte & 0x0F);

    out[(size_t)row * in_dim + blk * 32 + i] = __float2half(d * (float)(nib - 8));
}
