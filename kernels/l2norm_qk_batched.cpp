// Batched per-head L2 normalization for Q and K. Same per-row math as
// l2norm_qk.cpp; just adds an outer loop over n_rows so prefill issues
// one launch instead of n_rows. Each (row, head, side ∈ {Q, K}) is an
// independent workgroup — no cross-row state.
//
// grid = (n_heads, 2, n_rows); each block reduces over head_dim and
// writes one normalized [head_dim] vector.

#include <hip/hip_runtime.h>

extern "C" __global__
void l2norm_qk_batched_f32(const float* __restrict__ q_in_batch,    // [n_rows, n_heads, head_dim] (input stride may differ from output)
                           float*       __restrict__ q_out_batch,
                           const float* __restrict__ k_in_batch,    // [n_rows, n_heads, head_dim]
                           float*       __restrict__ k_out_batch,
                           unsigned int n_heads,
                           unsigned int head_dim,
                           float        eps,
                           float        q_scale,
                           unsigned int n_rows,
                           unsigned int q_in_row_stride,
                           unsigned int q_out_row_stride,
                           unsigned int k_in_row_stride,
                           unsigned int k_out_row_stride)
{
    extern __shared__ float smem[];
    const int side = blockIdx.y;        // 0 = Q, 1 = K
    const int h    = blockIdx.x;
    const int r    = blockIdx.z;
    if (h >= (int)n_heads || r >= (int)n_rows) return;
    const int tid = threadIdx.x;
    const int bs  = blockDim.x;

    const float* x_h;
    float*       y_h;
    if (side == 0) {
        x_h = q_in_batch  + (size_t)r * q_in_row_stride  + (size_t)h * head_dim;
        y_h = q_out_batch + (size_t)r * q_out_row_stride + (size_t)h * head_dim;
    } else {
        x_h = k_in_batch  + (size_t)r * k_in_row_stride  + (size_t)h * head_dim;
        y_h = k_out_batch + (size_t)r * k_out_row_stride + (size_t)h * head_dim;
    }
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
    if (tid == 0) smem[0] = rsqrtf(smem[0] + eps) * scale;
    __syncthreads();
    const float k = smem[0];

    for (int i = tid; i < (int)head_dim; i += bs) {
        y_h[i] = x_h[i] * k;
    }
}
