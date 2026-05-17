// Dequantize a repacked-Q4_K weight (quant::q4_k::repack_for_matvec
// layout) to fp16, for the rocBLAS-GEMM prefill path.
//
// The repacked planes have a per-row stride of `nsp = (in_dim/32)|1`
// sub-blocks (odd, to avoid HBM channel aliasing). Sub-block `gidx`
// belongs to row `gidx/n_sub`, position `gidx%n_sub`, and dequantizes
// into output weights [gidx*32, gidx*32+32) — the natural [out_dim,
// in_dim] fp16 layout the GEMM expects.
//
// grid = out_dim * (in_dim/32); block = 32.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

extern "C" __global__
void dequant_q4k_repacked_f16(const uint8_t* __restrict__ wbase,
                              __half*        __restrict__ out,
                              unsigned int   in_dim,
                              unsigned int   out_dim)
{
    const unsigned int gidx = blockIdx.x;       // global sub-block index
    const unsigned int k    = threadIdx.x;      // weight within sub-block
    const unsigned int n_sub = in_dim >> 5;
    const unsigned int nsp   = ((n_sub & (n_sub - 1u)) == 0u) ? (n_sub + 1u) : n_sub;

    const unsigned int row = gidx / n_sub;
    const unsigned int sb  = gidx % n_sub;
    if (row >= out_dim) return;

    const uint8_t*  nib = wbase + (size_t)(row * nsp + sb) * 16;
    const uint32_t* scl = reinterpret_cast<const uint32_t*>(
        wbase + (size_t)out_dim * nsp * 16 + (size_t)(row * nsp + sb) * 4);

    const uint16_t dsc_bits  = (uint16_t)(*scl & 0xFFFF);
    const uint16_t deff_bits = (uint16_t)(*scl >> 16);
    const float dsc  = __half2float(*reinterpret_cast<const __half*>(&dsc_bits));
    const float deff = __half2float(*reinterpret_cast<const __half*>(&deff_bits));

    const uint8_t nibble = (k < 16) ? (nib[k] & 0x0F) : (nib[k - 16] >> 4);
    out[(size_t)gidx * 32 + k] = __float2half(dsc * (float)nibble - deff);
}
