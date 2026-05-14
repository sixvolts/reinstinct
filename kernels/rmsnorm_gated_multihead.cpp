// Per-head Qwen3_5RMSNormGated:
//
//   y[h, i] = (x[h, i] * rsqrt(mean(x[h]²) + eps)) * w[i] * silu(z[h, i])
//
// Used at the end of the GDN block (`ssm_norm` weight, `z` gate from the
// `attn_gate` projection). Different from `rmsnorm_multihead` only in the
// extra per-element silu(z) factor.

#include <hip/hip_runtime.h>

extern "C" __global__
void rmsnorm_gated_multihead_f32(const float* __restrict__ x,    // [n_heads, head_dim]
                                 const float* __restrict__ z,    // [n_heads, head_dim]
                                 const float* __restrict__ w,    // [head_dim]
                                 float*       __restrict__ y,    // [n_heads, head_dim]
                                 unsigned int n_heads,
                                 unsigned int head_dim,
                                 float        eps)
{
    extern __shared__ float smem[];
    const int h   = blockIdx.x;
    if (h >= (int)n_heads) return;
    const int tid = threadIdx.x;
    const int bs  = blockDim.x;

    const float* x_h = x + (size_t)h * head_dim;
    const float* z_h = z + (size_t)h * head_dim;
    float*       y_h = y + (size_t)h * head_dim;

    float sum = 0.0f;
    for (int i = tid; i < (int)head_dim; i += bs) {
        float v = x_h[i];
        sum += v * v;
    }
    smem[tid] = sum;
    __syncthreads();
    for (int s = bs / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        float mean_sq = smem[0] / (float)head_dim;
        smem[0] = rsqrtf(mean_sq + eps);
    }
    __syncthreads();
    const float rrms = smem[0];

    for (int i = tid; i < (int)head_dim; i += bs) {
        const float zg     = z_h[i];
        const float silu_z = zg / (1.0f + __expf(-zg));
        y_h[i] = x_h[i] * rrms * w[i] * silu_z;
    }
}
