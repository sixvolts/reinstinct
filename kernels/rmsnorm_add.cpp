// Fused RMSNorm + residual add:
//
//   y[i] += rmsnorm(x)[i] * w[i] = y[i] + x[i] * rsqrt(mean(x^2) + eps) * w[i]
//
// Replaces the two-kernel sequence:
//   launch_rmsnorm(x, w, normed, n);  // normed = rmsnorm(x) * w
//   launch_add(y, normed, n);          // y += normed
//
// One launch instead of two — saves the ~9 μs per-call dispatch cost on
// the gemma4 decode path, which calls this pattern 4× per layer × 30+
// layers per forward.
//
// Block layout matches `rmsnorm_f32`: one block per vector, configurable
// block size, dynamic shared memory of blockDim.x floats for the tree
// reduction. blockIdx.y stripes per-vector for the batched variant.
//
// Numerics: reduction is parallel-tree (same order as rmsnorm_f32), so
// the rrms value is bit-identical to running rmsnorm_f32 standalone.
// The final `y += x * rrms * w` is one FMA per element; against the
// previous two-step (`normed = x * rrms * w` then `y += normed`), the
// only difference is whether the intermediate normed is rounded once or
// twice. Both produce the same fp32 result for normal inputs.

#include <hip/hip_runtime.h>

extern "C" __global__
void rmsnorm_add_f32(const float* __restrict__ x,    // [n]
                     const float* __restrict__ w,    // [n]
                     float*       __restrict__ y,    // [n]  in/out, += normed*w
                     unsigned int n,
                     float        eps)
{
    extern __shared__ float smem[];
    const int tid = threadIdx.x;
    const int bs  = blockDim.x;

    x += (size_t)blockIdx.y * n;
    y += (size_t)blockIdx.y * n;

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
        float mean_sq = smem[0] / (float)n;
        smem[0] = rsqrtf(mean_sq + eps);
    }
    __syncthreads();
    const float rrms = smem[0];

    for (int i = tid; i < (int)n; i += bs) {
        y[i] += x[i] * rrms * w[i];
    }
}
