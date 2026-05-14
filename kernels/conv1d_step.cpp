// Causal depthwise Conv1D, single-step streaming variant.
//
// For each of n_channels channels independently:
//   y[ch] = w[ch, K-1] * x_new[ch] + Σ_{k=0..K-2} w[ch, k] * history[ch, k]
//   shift history left by 1 and append x_new[ch] at the end
//
// Equivalent to running PyTorch Conv1d(padding=K-1) over the implicit
// concatenation [history | x_new] and taking the last output.
//
// Weight layout matches GGUF ssm_conv1d.weight: w[ch * K + k], i.e. each
// channel's K weights are contiguous.
//
// One thread per channel. With Qwen 3.5 0.8B: n_channels = 6144, K = 4 →
// history is [6144, 3] floats and we touch ~6144 * 4 = 24K floats per call.

#include <hip/hip_runtime.h>

extern "C" __global__
void conv1d_step_f32(const float* __restrict__ x_new,    // [n_channels]
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

    // y = w[K-1] * x_new + Σ_k w[k] * h[k]
    float acc = w_ch[kernel_size - 1] * x_new[ch];
    for (int k = 0; k < (int)hist_w; k++) {
        acc += w_ch[k] * h_ch[k];
    }
    y[ch] = acc;

    // Shift history left by 1, append x_new[ch] at the end.
    for (int k = 0; k + 1 < (int)hist_w; k++) {
        h_ch[k] = h_ch[k + 1];
    }
    if (hist_w >= 1) {
        h_ch[hist_w - 1] = x_new[ch];
    }
}
