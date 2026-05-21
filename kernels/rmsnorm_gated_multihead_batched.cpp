// Batched per-head Qwen3_5RMSNormGated. Same per-row math as
// rmsnorm_gated_multihead.cpp; just adds grid.y = n_rows so prefill
// issues one launch instead of n_rows. Per-row state is independent.

#include <hip/hip_runtime.h>

extern "C" __global__
void rmsnorm_gated_multihead_batched_f32(const float* __restrict__ x_batch,  // [n_rows, n_heads, head_dim]
                                         const float* __restrict__ z_batch,  // [n_rows, n_heads, head_dim] (gate)
                                         const float* __restrict__ w,        // [head_dim]
                                         float*       __restrict__ y_batch,  // [n_rows, n_heads, head_dim]
                                         unsigned int n_heads,
                                         unsigned int head_dim,
                                         float        eps,
                                         unsigned int n_rows,
                                         unsigned int row_stride)  // floats per row (typically n_heads*head_dim)
{
    extern __shared__ float smem[];
    const int h = blockIdx.x;
    const int r = blockIdx.y;
    if (h >= (int)n_heads || r >= (int)n_rows) return;
    const int tid = threadIdx.x;
    const int bs  = blockDim.x;

    const float* x_h = x_batch + (size_t)r * row_stride + (size_t)h * head_dim;
    const float* z_h = z_batch + (size_t)r * row_stride + (size_t)h * head_dim;
    float*       y_h = y_batch + (size_t)r * row_stride + (size_t)h * head_dim;

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
