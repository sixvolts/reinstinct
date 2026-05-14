// Wave-cooperative fp16 matvec for gfx906.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

extern "C" __global__
void matvec_f16_wave64_f32(const uint16_t* __restrict__ w_bits,
                           const float*    __restrict__ x,
                           float*          __restrict__ y,
                           unsigned int in_dim,
                           unsigned int out_dim)
{
    const int row = blockIdx.x;
    if (row >= (int)out_dim) return;
    const int lane = threadIdx.x;

    const uint16_t* wrow = w_bits + (size_t)row * in_dim;
    float acc = 0.0f;
    for (int i = lane; i < (int)in_dim; i += 64) {
        const __half h = *reinterpret_cast<const __half*>(&wrow[i]);
        acc += __half2float(h) * x[i];
    }

    acc += __shfl_xor(acc, 32);
    acc += __shfl_xor(acc, 16);
    acc += __shfl_xor(acc,  8);
    acc += __shfl_xor(acc,  4);
    acc += __shfl_xor(acc,  2);
    acc += __shfl_xor(acc,  1);

    if (lane == 0) y[row] = acc;
}
