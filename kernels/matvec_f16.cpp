// FP16 matvec: y[j] = Σ_i (fp16→fp32) w[j*in_dim + i] * x[i].
//
// Activation x stays fp32; weights live as raw fp16 (uint16_t bits)
// and are converted in registers per element. Used for the small
// projection tensors in Unsloth UD-Q4_K_XL files where the GGUF
// converter stores fp16 instead of a K-quant.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

extern "C" __global__
void matvec_f16_f32(const uint16_t* __restrict__ w_bits,  // fp16 raw bits, [out_dim, in_dim]
                    const float*    __restrict__ x,       // [in_dim]
                    float*          __restrict__ y,       // [out_dim]
                    unsigned int in_dim,
                    unsigned int out_dim)
{
    extern __shared__ float smem[];
    const int row = blockIdx.x;
    if (row >= (int)out_dim) return;
    const int tid = threadIdx.x;
    const int bs  = blockDim.x;

    const uint16_t* wrow = w_bits + (size_t)row * in_dim;
    float acc = 0.0f;
    for (int i = tid; i < (int)in_dim; i += bs) {
        const __half h = *reinterpret_cast<const __half*>(&wrow[i]);
        acc += __half2float(h) * x[i];
    }
    smem[tid] = acc;
    __syncthreads();
    for (int s = bs / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) y[row] = smem[0];
}
