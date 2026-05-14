// FP32 matvec: y[j] = sum_i w[j, i] * x[i]
//
// W is row-major [out_dim, in_dim]; one block per output row.
// Block size is configurable (caller picks 256 by default); each thread
// strides over in_dim with stride blockDim.x and accumulates a partial
// dot product. Tree reduction in shared memory finishes the row.
//
// This kernel intentionally has no LDS staging of x or w — it's a
// correctness baseline, not the perf path. Real decode will use
// fused dequant+GEMV variants per quant type.

#include <hip/hip_runtime.h>

extern "C" __global__
void matvec_f32(const float* __restrict__ w,    // [out_dim, in_dim]
                const float* __restrict__ x,    // [in_dim]
                float*       __restrict__ y,    // [out_dim]
                unsigned int in_dim,
                unsigned int out_dim)
{
    extern __shared__ float smem[];
    const int row = blockIdx.x;
    if (row >= (int)out_dim) return;
    const int tid = threadIdx.x;
    const int bs  = blockDim.x;

    const float* wrow = w + (size_t)row * (size_t)in_dim;
    float acc = 0.0f;
    for (int i = tid; i < (int)in_dim; i += bs) {
        acc += wrow[i] * x[i];
    }

    smem[tid] = acc;
    __syncthreads();
    for (int s = bs / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) y[row] = smem[0];
}
