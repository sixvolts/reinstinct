// fp32 ↔ fp16 element-wise conversion. Used to feed/drain rocBLAS
// HGEMM in the batched prefill path (activations are fp32 in our
// pipeline; HGEMM wants fp16 on both inputs and the output).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

extern "C" __global__
void cvt_f32_to_f16(const float* __restrict__ in,
                    __half*      __restrict__ out,
                    unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = __float2half(in[i]);
}

extern "C" __global__
void cvt_f16_to_f32(const __half* __restrict__ in,
                    float*        __restrict__ out,
                    unsigned int n)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = __half2float(in[i]);
}
