// Bulk IQ4_XS → fp16 dequant for the repacked two-plane layout used by
// matvec_iq4xs_repacked. Layout is Q4_0's (quant::iq4_xs::repack_for_matvec):
//   slab + 0                    : out_dim × nsp × 16  nibble bytes
//   slab + out_dim × nsp × 16   : out_dim × nsp × 2   fp16 dl (pre-folded)
// The value is dl * codebook[nibble]. Dense [out_dim × in_dim] fp16 out;
// grid = (out_dim × n_sub_per_row,), block = 32, like the other repacked
// dequants so the prefill dispatcher can reuse the launch shape.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

__constant__ int8_t IQ4NL_KV[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10,
       1,   13,  25,  38,  53,  69,  89, 113,
};

extern "C" __global__
void dequant_iq4xs_repacked_f16(const unsigned char* __restrict__ slab,
                                __half*              __restrict__ out,
                                unsigned int in_dim,
                                unsigned int out_dim)
{
    const unsigned int n_sub = in_dim >> 5;
    const unsigned int nsp = ((n_sub & (n_sub - 1u)) == 0u) ? (n_sub + 1u) : n_sub;
    const unsigned int idx = blockIdx.x;
    const unsigned int row = idx / n_sub;
    const unsigned int sb  = idx - row * n_sub;
    if (row >= out_dim) return;
    const int i = (int)threadIdx.x;
    if (i >= 32) return;

    const uint8_t*  nib_plane = reinterpret_cast<const uint8_t*>(slab);
    const uint16_t* d_plane   = reinterpret_cast<const uint16_t*>(
        slab + (size_t)out_dim * nsp * 16);

    const size_t pidx = (size_t)row * nsp + sb;
    const uint16_t db = d_plane[pidx];
    const float    dl = __half2float(*reinterpret_cast<const __half*>(&db));
    const int is_high = i >> 4;
    const int k       = i & 15;
    const uint8_t byte = nib_plane[pidx * 16 + k];
    const int nib = is_high ? (byte >> 4) : (byte & 0x0F);
    out[(size_t)row * in_dim + sb * 32 + i] = __float2half(dl * (float)IQ4NL_KV[nib]);
}
