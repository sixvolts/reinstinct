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

    const unsigned int n_super = n_sub >> 3;
    const uint8_t*  nib = wbase + (size_t)(row * nsp + sb) * 16;
    // v2 scales: sc|m (u8 each) per sub-block, d|dmin (fp16) per superblock.
    const uint16_t* smp = reinterpret_cast<const uint16_t*>(
        wbase + (size_t)out_dim * nsp * 16 + (size_t)(row * nsp + sb) * 2);
    const uint32_t* ddp = reinterpret_cast<const uint32_t*>(
        wbase + (size_t)out_dim * nsp * 16 + (size_t)out_dim * nsp * 2
              + (size_t)(row * n_super + (sb >> 3)) * 4);

    const uint16_t sm = *smp;
    const uint16_t d_bits    = (uint16_t)(*ddp & 0xFFFF);
    const uint16_t dmin_bits = (uint16_t)(*ddp >> 16);
    const float dsc  = __half2float(*reinterpret_cast<const __half*>(&d_bits))
                       * (float)(sm & 0xFFu);
    const float deff = __half2float(*reinterpret_cast<const __half*>(&dmin_bits))
                       * (float)(sm >> 8);

    const uint8_t nibble = (k < 16) ? (nib[k] & 0x0F) : (nib[k - 16] >> 4);
    out[(size_t)gidx * 32 + k] = __float2half(dsc * (float)nibble - deff);
}
