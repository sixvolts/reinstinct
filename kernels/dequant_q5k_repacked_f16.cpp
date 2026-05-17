// Dequantize a repacked-Q5_K weight (quant::q5_k::repack_for_matvec) to
// fp16 for the rocBLAS-GEMM prefill path.
//
// grid = out_dim * (in_dim/32); block = 32.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

extern "C" __global__
void dequant_q5k_repacked_f16(const uint8_t* __restrict__ wbase,
                              __half*        __restrict__ out,
                              unsigned int   in_dim,
                              unsigned int   out_dim)
{
    const unsigned int gidx = blockIdx.x;
    const unsigned int k    = threadIdx.x;          // 0..31
    const unsigned int n_sub = in_dim >> 5;
    const unsigned int nsp = ((n_sub & (n_sub - 1u)) == 0u) ? (n_sub + 1u) : n_sub;

    const unsigned int row = gidx / n_sub;
    const unsigned int sb  = gidx % n_sub;
    if (row >= out_dim) return;
    const size_t idx = (size_t)row * nsp + sb;

    const uint8_t*  nib = wbase + idx * 16;
    const uint32_t* qhp = reinterpret_cast<const uint32_t*>(
        wbase + (size_t)out_dim * nsp * 16 + idx * 4);
    const uint32_t* scl = reinterpret_cast<const uint32_t*>(
        wbase + (size_t)out_dim * nsp * 16 + (size_t)out_dim * nsp * 4 + idx * 4);

    const uint16_t dsc_bits  = (uint16_t)(*scl & 0xFFFF);
    const uint16_t deff_bits = (uint16_t)(*scl >> 16);
    const float dsc  = __half2float(*reinterpret_cast<const __half*>(&dsc_bits));
    const float deff = __half2float(*reinterpret_cast<const __half*>(&deff_bits));

    const uint32_t nibble = (k < 16) ? (nib[k] & 0x0F) : (nib[k - 16] >> 4);
    const unsigned int bit = (k < 16)
        ? (8u * (k / 4) + (k % 4))
        : (8u * ((k - 16) / 4) + 4u + ((k - 16) % 4));
    const uint32_t q5 = nibble | (((*qhp >> bit) & 1u) << 4);

    out[(size_t)gidx * 32 + k] = __float2half(dsc * (float)q5 - deff);
}
