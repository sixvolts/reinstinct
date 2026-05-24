// Fused RMSNorm + Q8 quantize:
//
//   For each sub-block sb of 32 elements:
//     v[i] = x[i] * rsqrt(mean(x^2) + eps) * w[i]
//     out[sb].qs[i] = round(v[i] * 127 / max(|v|))
//     out[sb].d     = max(|v|) / 127
//     out[sb].xsum  = sum(v)        (used by Q4_K/Q5_K dmin term)
//
// Replaces the two-kernel sequence:
//   launch_rmsnorm(x, w, normed, n);
//   launch_quantize_q8(normed, out_q8, n, 1);
//
// Saves one launch per pair AND the round-trip of normalized values
// through HBM (rmsnorm writes `normed` only for quantize to read it
// back). Used in the MoE decode path's pre-expert preparation.
//
// Block layout: single WG per vector (blockIdx.x = 0, blockIdx.y =
// vec). 512 threads = 16 wavefronts × 32 lanes. The 32-lane group in
// each wave naturally maps to one sub-block of 32 quants — the amax
// and sum reductions stay within a wave via __shfl_xor.
//
// Numerics: rmsnorm reduction uses parallel-tree (same order as
// rmsnorm_f32), so rrms is bit-identical to running the two kernels
// separately. The quantize half is bit-identical to quantize_q8_f32
// — same per-block amax + sum + round.

#include <hip/hip_runtime.h>
#include <stdint.h>
#include "gfx906_dpp.h"

struct __attribute__((packed)) BlockQ8 {
    float  d;
    float  xsum;
    int8_t qs[32];
};
static_assert(sizeof(BlockQ8) == 40, "BlockQ8 must be 40 bytes");

extern "C" __global__
void rmsnorm_q8_f32(const float* __restrict__ x,    // [n]
                    const float* __restrict__ w,    // [n]
                    BlockQ8*     __restrict__ out,  // [n/32]
                    unsigned int n,
                    float        eps)
{
    extern __shared__ float smem[];
    const int tid = threadIdx.x;
    const int bs  = blockDim.x;

    // Per-vector batching (blockIdx.y = vec idx; grid_y = n_vec).
    x   += (size_t)blockIdx.y * n;
    out += (size_t)blockIdx.y * (n >> 5);

    // --- Phase 1: full-vector reduction → rrms ---
    float sum = 0.0f;
    for (int i = tid; i < (int)n; i += bs) {
        float v = x[i];
        sum += v * v;
    }
    smem[tid] = sum;
    __syncthreads();
    for (int s = bs / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        smem[0] = rsqrtf(smem[0] / (float)n + eps);
    }
    __syncthreads();
    const float rrms = smem[0];

    // --- Phase 2: per-sub-block normalize + quantize ---
    // Each 32-lane wavefront slice handles one sub-block at a time.
    // n_warps = bs / 32; stride sub-blocks across warps.
    const unsigned int n_sub  = n >> 5;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;
    const int n_warps = bs >> 5;

    for (unsigned int sb = warp_id; sb < n_sub; sb += n_warps) {
        const int idx = sb * 32 + lane;
        const float xv = x[idx];
        const float wv = w[idx];
        const float v  = xv * rrms * wv;

        float amax = fabsf(v);
        float vsum = v;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            amax = fmaxf(amax, __shfl_xor(amax, o));
            vsum += __shfl_xor(vsum, o);
        }

        const float inv = amax > 0.0f ? 127.0f * fast_rcp_f32(amax) : 0.0f;
        int q = (int)rintf(v * inv);
        q = max(-127, min(127, q));

        out[sb].qs[lane] = (int8_t)q;
        if (lane == 0) {
            out[sb].d    = amax > 0.0f ? amax / 127.0f : 1.0f;
            out[sb].xsum = vsum;
        }
    }
}
