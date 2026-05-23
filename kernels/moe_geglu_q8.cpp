// Fused MoE GeGLU + Q8 quantize:
//
//   For each (slot, sub_block):
//     v[i] = gelu(gu[gate]) * gu[up]    for i in sub_block of 32
//     out[sb].qs[i] = round(v[i] * 127 / max(|v|))
//     out[sb].d, out[sb].xsum            (per-block scale + sum)
//
// Replaces moe_geglu(gu, act) + quantize_q8(act, xq8, ff_exp, n_slot).
// Saves one launch per layer + the HBM round-trip of act through fp32
// (kept in registers between geglu and quantize).
//
// Layout matches quantize_q8: block=256 = 8 sub-blocks × 32 lanes.
// blockIdx.x strides sub-block groups, blockIdx.y selects the slot.

#include <hip/hip_runtime.h>
#include <stdint.h>

struct __attribute__((packed)) BlockQ8 {
    float  d;
    float  xsum;
    int8_t qs[32];
};
static_assert(sizeof(BlockQ8) == 40, "BlockQ8 must be 40 bytes");

__device__ __forceinline__ float gelu_tanh(float x) {
    const float k = 0.7978845608028654f;          // sqrt(2/pi)
    const float c = x + 0.044715f * x * x * x;
    return 0.5f * x * (1.0f + tanhf(k * c));
}

extern "C" __global__
void moe_geglu_q8_f32(const float* __restrict__ gu,    // [n_slot, 2*ff_exp]
                      BlockQ8*     __restrict__ out,   // [n_slot, ff_exp/32]
                      unsigned int ff_exp)
{
    const unsigned int slot  = blockIdx.y;
    const unsigned int n_sub = ff_exp >> 5;
    const unsigned int blk   = blockIdx.x * 8u + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (blk >= n_sub) return;

    const float* g = gu + (size_t)slot * 2u * ff_exp;
    BlockQ8*     o = out + (size_t)slot * n_sub;

    const int idx = blk * 32 + lane;
    const float gate = g[idx];
    const float up   = g[ff_exp + idx];
    const float v    = gelu_tanh(gate) * up;

    float amax = fabsf(v);
    float vsum = v;
    #pragma unroll
    for (int o2 = 16; o2 > 0; o2 >>= 1) {
        amax = fmaxf(amax, __shfl_xor(amax, o2));
        vsum += __shfl_xor(vsum, o2);
    }

    const float inv = amax > 0.0f ? 127.0f / amax : 0.0f;
    int q = (int)rintf(v * inv);
    q = max(-127, min(127, q));

    o[blk].qs[lane] = (int8_t)q;
    if (lane == 0) {
        o[blk].d    = amax > 0.0f ? amax / 127.0f : 1.0f;
        o[blk].xsum = vsum;
    }
}
