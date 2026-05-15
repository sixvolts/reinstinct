// Causal depthwise Conv1D streaming step with SiLU fused into the output.
//
// Identical to conv1d_step.cpp except the final write is silu(acc)
// instead of acc. The convolution still operates on raw input values
// (the rolling history stores raw x_new, not the activated output),
// so this exactly matches "conv1d_step then silu_inplace" — it just
// saves a kernel launch.

#include <hip/hip_runtime.h>

extern "C" __global__
void conv1d_step_silu_f32(const float* __restrict__ x_new,    // [n_channels]
                          const float* __restrict__ w,        // [n_channels, K]
                          float*       __restrict__ history,  // [n_channels, K-1]  in/out
                          float*       __restrict__ y,        // [n_channels]
                          unsigned int n_channels,
                          unsigned int kernel_size)
{
    const unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= n_channels) return;

    const unsigned int hist_w = kernel_size - 1;
    const float* w_ch = w + (size_t)ch * kernel_size;
    float*       h_ch = history + (size_t)ch * hist_w;

    float acc = w_ch[kernel_size - 1] * x_new[ch];
    for (int k = 0; k < (int)hist_w; k++) {
        acc += w_ch[k] * h_ch[k];
    }
    // Fused SiLU: y = conv * sigmoid(conv).
    y[ch] = acc / (1.0f + __expf(-acc));

    for (int k = 0; k + 1 < (int)hist_w; k++) {
        h_ch[k] = h_ch[k + 1];
    }
    if (hist_w >= 1) {
        h_ch[hist_w - 1] = x_new[ch];
    }
}
