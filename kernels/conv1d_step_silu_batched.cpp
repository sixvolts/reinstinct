// Batched causal depthwise Conv1D streaming step + fused SiLU. Folds
// the prefill's `for r in 0..n { conv1d_step_silu }` loop into a single
// launch — each thread owns one channel and loops over the n rows in
// registers, threading the rolling history through the loop body.
//
// Memory traffic is identical to per-row launches (each row's input is
// still read once, each row's output written once); the win is killing
// the host-side launch overhead (n_rows × per-launch HIP cost), which
// dominates the GDN prefill block at the typical n ≥ 64.
//
// Channel layout is unchanged — `x_new_batch` is row-major
// `[n_rows, n_channels]`, `y_batch` is the same shape. `history` is
// `[n_channels, kernel_size - 1]` and is read/written in place.

#include <hip/hip_runtime.h>

#define MAX_HIST_W 3        // qwen 3.5/3.6 use kernel_size=4 → hist_w=3

extern "C" __global__
void conv1d_step_silu_batched_f32(const float* __restrict__ x_new_batch, // [n_rows, n_channels]
                                  const float* __restrict__ w,           // [n_channels, K]
                                  float*       __restrict__ history,     // [n_channels, K-1]
                                  float*       __restrict__ y_batch,     // [n_rows, n_channels]
                                  unsigned int n_channels,
                                  unsigned int kernel_size,
                                  unsigned int n_rows)
{
    const unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= n_channels) return;

    const unsigned int hist_w = kernel_size - 1;
    const float* w_ch = w + (size_t)ch * kernel_size;
    float*       h_ch = history + (size_t)ch * hist_w;

    // Pull the rolling history into registers — small (≤ MAX_HIST_W).
    float hist[MAX_HIST_W];
    #pragma unroll
    for (int k = 0; k < MAX_HIST_W; k++) {
        hist[k] = (k < (int)hist_w) ? h_ch[k] : 0.0f;
    }
    const float wK = w_ch[kernel_size - 1];

    // Cache the K-1 history weights too (also small).
    float w_hist[MAX_HIST_W];
    #pragma unroll
    for (int k = 0; k < MAX_HIST_W; k++) {
        w_hist[k] = (k < (int)hist_w) ? w_ch[k] : 0.0f;
    }

    for (unsigned int r = 0; r < n_rows; r++) {
        const float x = x_new_batch[(size_t)r * n_channels + ch];
        float acc = wK * x;
        #pragma unroll
        for (int k = 0; k < MAX_HIST_W; k++) {
            if (k < (int)hist_w) acc += w_hist[k] * hist[k];
        }
        // Fused SiLU.
        y_batch[(size_t)r * n_channels + ch] = acc / (1.0f + __expf(-acc));

        // Shift register history + insert this row's x.
        #pragma unroll
        for (int k = 0; k + 1 < MAX_HIST_W; k++) {
            if (k + 1 < (int)hist_w) hist[k] = hist[k + 1];
        }
        if (hist_w >= 1) hist[hist_w - 1] = x;
    }

    // Write the final history back to global memory for the next call.
    #pragma unroll
    for (int k = 0; k < MAX_HIST_W; k++) {
        if (k < (int)hist_w) h_ch[k] = hist[k];
    }
}
