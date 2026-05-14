// RMSNorm (no-shift): y[i] = x[i] * rsqrt(mean(x^2) + eps) * w[i]
//
// Matches src/cpu/ops.rs `rmsnorm`. Note that for the Qwen3_5 model the
// GGUF norm weights already have +1 baked in by convert_hf_to_gguf.py
// (Qwen3NextModel.modify_tensors) — so this kernel uses w[i] directly,
// not 1+w[i]. The Qwen3_5RMSNormGated variant is a separate kernel.
//
// One block per vector. Block size is configurable (256 default); each
// thread strides over n with stride blockDim.x, so any n ≤ 65536 fits
// in one launch. Reduction uses dynamic shared memory of blockDim.x floats.

#include <hip/hip_runtime.h>

extern "C" __global__
void rmsnorm_f32(const float* __restrict__ x,
                 const float* __restrict__ w,
                 float*       __restrict__ y,
                 unsigned int n,
                 float        eps)
{
    extern __shared__ float smem[];
    const int tid = threadIdx.x;
    const int bs  = blockDim.x;

    // Per-thread partial sum of squares.
    float sum = 0.0f;
    for (int i = tid; i < (int)n; i += bs) {
        float v = x[i];
        sum += v * v;
    }

    // Tree reduction in shared memory. Differs from CPU in summation order
    // (parallel tree vs sequential) — bit-identity is not expected, but
    // accumulated relative error stays well below 1e-5 for n ≤ 16k.
    smem[tid] = sum;
    __syncthreads();
    for (int s = bs / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }

    // Broadcast rrms via shared mem.
    if (tid == 0) {
        float mean_sq = smem[0] / (float)n;
        smem[0] = rsqrtf(mean_sq + eps);
    }
    __syncthreads();
    const float rrms = smem[0];

    for (int i = tid; i < (int)n; i += bs) {
        y[i] = x[i] * rrms * w[i];
    }
}
