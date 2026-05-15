// Fused per-head L2 normalization for Q and K in one launch.
//
// The GDN block L2-normalizes Q (with a 1/√head_dim scale) and K
// (scale 1) separately. They have the same head count and head_dim,
// so a single 2D-grid launch — blockIdx.y picks the side — replaces
// two launches of l2norm_multihead.

#include <hip/hip_runtime.h>

extern "C" __global__
void l2norm_qk_f32(const float* __restrict__ q_in,    // [n_heads, head_dim]
                   float*       __restrict__ q_out,
                   const float* __restrict__ k_in,    // [n_heads, head_dim]
                   float*       __restrict__ k_out,
                   unsigned int n_heads,
                   unsigned int head_dim,
                   float        eps,
                   float        q_scale)
{
    extern __shared__ float smem[];
    const int side = blockIdx.y;        // 0 = Q, 1 = K
    const int h    = blockIdx.x;
    if (h >= (int)n_heads) return;
    const int tid = threadIdx.x;
    const int bs  = blockDim.x;

    const float* x_h = (side == 0 ? q_in  : k_in)  + (size_t)h * head_dim;
    float*       y_h = (side == 0 ? q_out : k_out) + (size_t)h * head_dim;
    const float scale = (side == 0) ? q_scale : 1.0f;

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
