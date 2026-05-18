// Dequantize a repacked-Q6_K weight (quant::q6_k::repack_for_matvec) to
// fp16 for the rocBLAS-GEMM prefill path.
//
// grid = out_dim * (in_dim/32); block = 32.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

extern "C" __global__
void dequant_q6k_repacked_f16(const uint8_t* __restrict__ wbase,
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

    const unsigned int n_super = n_sub >> 3;
    const uint8_t*  nib = wbase + idx * 16;
    const uint32_t* h2p = reinterpret_cast<const uint32_t*>(
        wbase + (size_t)out_dim * nsp * 16 + idx * 8);
    // v2 scales: sc_lo|sc_hi (int8) per sub-block, d (fp16) per superblock.
    const uint16_t* smp = reinterpret_cast<const uint16_t*>(
        wbase + (size_t)out_dim * nsp * 16 + (size_t)out_dim * nsp * 8 + idx * 2);
    const uint16_t* ddp = reinterpret_cast<const uint16_t*>(
        wbase + (size_t)out_dim * nsp * 16 + (size_t)out_dim * nsp * 8
              + (size_t)out_dim * nsp * 2 + ((size_t)row * n_super + (sb >> 3)) * 2);

    const uint16_t sm = *smp;
    const int sc = (k < 16) ? (int)(int8_t)(sm & 0xFFu) : (int)(int8_t)(sm >> 8);
    const float dsc = __half2float(*reinterpret_cast<const __half*>(ddp)) * (float)sc;

    const uint32_t nibble = (k < 16) ? (nib[k] & 0x0F) : (nib[k - 16] >> 4);

    // high-2-bit field: dp4a group g, weight b within group → byte g,
    // bits 2b..2b+1. group 2j covers k=4j+b, group 2j+1 covers 16+4j+b.
    const unsigned int kk = (k < 16) ? k : (k - 16);
    const unsigned int bg = 2u * (kk / 4) + (k < 16 ? 0u : 1u);
    const unsigned int bo = 2u * (kk % 4);
    const uint32_t hword = h2p[bg >> 2];
    const uint32_t high2 = ((hword >> (8u * (bg & 3u))) >> bo) & 3u;

    const int q6 = (int)(nibble | (high2 << 4));
    out[(size_t)gidx * 32 + k] = __float2half(dsc * (float)(q6 - 32));
}
