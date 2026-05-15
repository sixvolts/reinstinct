// Quantize an f32 activation vector to int8 blocks of 32 for the dp4a
// matvec path. Each block stores a per-block scale `d`, the plain sum
// of the original values `xsum` (used for the Q4_K/Q5_K dmin term),
// and 32 signed int8 quants.
//
// grid = in_dim / 32; block = 32 (one thread per element).

#include <hip/hip_runtime.h>
#include <stdint.h>

struct __attribute__((packed)) BlockQ8 {
    float  d;
    float  xsum;
    int8_t qs[32];
};
static_assert(sizeof(BlockQ8) == 40, "BlockQ8 must be 40 bytes");

extern "C" __global__
void quantize_q8_f32(const float*  __restrict__ x,
                     BlockQ8*      __restrict__ out,
                     unsigned int  in_dim)
{
    const unsigned int blk = blockIdx.x;
    const int l = threadIdx.x;                  // 0..31
    const float v = x[blk * 32 + l];

    float amax = fabsf(v);
    float sum  = v;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor(amax, o));
        sum += __shfl_xor(sum, o);
    }

    const float d   = amax > 0.0f ? amax / 127.0f : 1.0f;
    const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    int q = (int)rintf(v * inv);
    q = max(-127, min(127, q));

    out[blk].qs[l] = (int8_t)q;
    if (l == 0) { out[blk].d = d; out[blk].xsum = sum; }
}
