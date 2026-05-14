// Per-head L2 normalization with optional uniform scale.
//
//   For each head h:
//     y[h, i] = (x[h, i] / sqrt(Σ_i x[h,i]² + eps)) * scale
//
// Matches `cpu::ops::l2norm` (sum, NOT mean) plus a follow-up scalar
// multiply. Combining the two into one kernel avoids an extra launch
// for the Q-side `1/√head_dim` scaling in the GDN block (`scale = 1`
// for K, `scale = head_dim^-0.5` for Q).

#include <hip/hip_runtime.h>

extern "C" __global__
void l2norm_multihead_f32(const float* __restrict__ x,    // [n_heads, head_dim]
                          float*       __restrict__ y,    // [n_heads, head_dim]
                          unsigned int n_heads,
                          unsigned int head_dim,
                          float        eps,
                          float        scale)
{
    extern __shared__ float smem[];
    const int h   = blockIdx.x;
    if (h >= (int)n_heads) return;
    const int tid = threadIdx.x;
    const int bs  = blockDim.x;

    const float* x_h = x + (size_t)h * head_dim;
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
        smem[0] = rsqrtf(smem[0] + eps) * scale;
    }
    __syncthreads();
    const float k = smem[0];

    for (int i = tid; i < (int)head_dim; i += bs) {
        y_h[i] = x_h[i] * k;
    }
}
