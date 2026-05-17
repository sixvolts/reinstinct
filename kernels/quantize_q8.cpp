// Quantize an f32 activation vector to int8 blocks of 32 for the dp4a
// matvec path. Each block stores a per-block scale `d`, the plain sum
// of the original values `xsum` (used for the Q4_K/Q5_K dmin term),
// and 32 signed int8 quants.
//
// grid = (ceil(in_dim/256), n_vec); block = 256 — a full wavefront ×4,
// each block doing 8 sub-blocks of 32. The earlier block=32 launch was
// a half-wavefront on gfx906 (Wave64) and spawned in_dim/32 tiny
// blocks; this amortizes the dispatch over 8× the work per block.
// blockIdx.y selects the vector for the batched (n_vec > 1) case.

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
    const unsigned int vec = blockIdx.y;
    x   += (size_t)vec * in_dim;
    out += (size_t)vec * (in_dim >> 5);

    const unsigned int n_sub = in_dim >> 5;                        // total sub-blocks
    const unsigned int blk = blockIdx.x * 8 + (threadIdx.x >> 5);  // this thread's sub-block
    const int l = threadIdx.x & 31;                                // 0..31 within it
    if (blk >= n_sub) return;

    const float v = x[blk * 32 + l];
    float amax = fabsf(v);
    float sum  = v;
    // Reduction stays within the 32-aligned lane group.
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
