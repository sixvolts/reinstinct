// Batched fp16-weight GEMM: y[r][j] = Σ_i (fp16→fp32) w[j*in_dim + i] * x[r][i].
//
// The multi-row counterpart of `matvec_f16.cpp`, and the fallback GEMM for
// every weight dtype that has no repacked int8 MMQ kernel: the caller
// dequantizes such a weight to fp16 first and hands the result here. That
// covers the small F16/F32 projection tensors Unsloth UD files leave
// unquantized (Qwen's GDN ssm_alpha/beta among them).
//
// This exists so the engine never needs rocBLAS. rocBLAS builds from ROCm
// 7.x ship no gfx906 Tensile kernels and `abort()` the process on handle
// creation, which would take out the very hardware this engine targets.
//
// Activations and outputs stay fp32 throughout — unlike the rocBLAS path,
// which had to narrow x to fp16 on the way in and widen y on the way out.
//
// One block per (output column, row tile). Each block streams one weight
// row from HBM and dots it against up to NR_TILE activation rows, so the
// weight traffic — the dominant term — is amortized NR_TILE ways.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#define NR_TILE 8

extern "C" __global__
void gemm_f16_rows_f32(const uint16_t* __restrict__ w_bits,  // fp16 bits, [out_dim, in_dim]
                       const float*    __restrict__ x,       // [n_rows, in_dim]
                       float*          __restrict__ y,       // [n_rows, out_dim]
                       unsigned int in_dim,
                       unsigned int out_dim,
                       unsigned int n_rows)
{
    extern __shared__ float smem[];          // NR_TILE * blockDim.x floats
    const int col = blockIdx.x;
    const int r0  = blockIdx.y * NR_TILE;
    if (col >= (int)out_dim || r0 >= (int)n_rows) return;

    const int tid = threadIdx.x;
    const int bs  = blockDim.x;
    int nr = (int)n_rows - r0;
    if (nr > NR_TILE) nr = NR_TILE;

    const uint16_t* wrow = w_bits + (size_t)col * in_dim;

    float acc[NR_TILE];
#pragma unroll
    for (int k = 0; k < NR_TILE; ++k) acc[k] = 0.0f;

    for (int i = tid; i < (int)in_dim; i += bs) {
        const __half h = *reinterpret_cast<const __half*>(&wrow[i]);
        const float  wv = __half2float(h);
        for (int k = 0; k < nr; ++k)
            acc[k] += wv * x[(size_t)(r0 + k) * in_dim + i];
    }

    for (int k = 0; k < nr; ++k) smem[k * bs + tid] = acc[k];
    __syncthreads();
    for (int s = bs / 2; s > 0; s >>= 1) {
        if (tid < s)
            for (int k = 0; k < nr; ++k) smem[k * bs + tid] += smem[k * bs + tid + s];
        __syncthreads();
    }
    if (tid == 0)
        for (int k = 0; k < nr; ++k)
            y[(size_t)(r0 + k) * out_dim + col] = smem[k * bs];
}
