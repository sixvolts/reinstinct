// Per-head decay and beta for the Gated Delta-Net recurrence:
//
//   beta[h]  = sigmoid(b[h])
//   g[h]     = ssm_a[h] * softplus(a[h] + dt_bias[h])      ssm_a stored as -exp(A_log)
//   decay[h] = exp(g[h])                                    ∈ (0, 1]
//
// One thread per head. n_heads is small (16 for Qwen 3.5 0.8B), so this
// is launched as a single-block kernel.

#include <hip/hip_runtime.h>

__device__ __forceinline__ float softplus_stable(float x) {
    // log(1 + exp(x)). For numerical stability:
    //   x > 0:  x + log(1 + exp(-x))
    //   x ≤ 0:  log(1 + exp(x))
    return (x > 0.0f) ? x + __logf(1.0f + __expf(-x))
                      :     __logf(1.0f + __expf(x));
}

extern "C" __global__
void gdn_decay_beta_f32(const float* __restrict__ a,         // [n_heads]
                        const float* __restrict__ b,         // [n_heads]
                        const float* __restrict__ ssm_a,     // [n_heads]
                        const float* __restrict__ dt_bias,   // [n_heads]
                        float*       __restrict__ decay,     // [n_heads]
                        float*       __restrict__ beta,      // [n_heads]
                        unsigned int n_heads)
{
    const unsigned int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= n_heads) return;

    const float ax = a[h] + dt_bias[h];
    const float sp = softplus_stable(ax);
    const float g  = ssm_a[h] * sp;
    decay[h] = __expf(g);
    beta[h]  = 1.0f / (1.0f + __expf(-b[h]));
}
