// Fused Q8_0 dequantize + matvec.
//
//   y[j] = sum over blocks b of d[j,b] * sum_i (qs[j,b,i] * x[b*32 + i])
//
// W is laid out exactly as on disk: an array of BlockQ8_0 structs of shape
// [out_dim, in_dim/32]. Each block is 34 bytes — fp16 scale `d` followed
// by 32 signed int8 quants.
//
// The point of this kernel vs `matvec_f32(dequant(W), x)` is that we never
// materialise the dequantised fp32 weights — saves ~4× memory bandwidth
// (and ~4× VRAM if W is resident) for an effective 8.5 bpw vs 32 bpw.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

struct __attribute__((packed)) BlockQ8_0 {
    uint16_t d;        // raw fp16 bits (matches GGUF on-disk layout)
    int8_t   qs[32];
};
static_assert(sizeof(BlockQ8_0) == 34, "BlockQ8_0 must be 34 bytes");

extern "C" __global__
void matvec_q8_0_f32(const BlockQ8_0* __restrict__ w_blocks,  // [out_dim, in_dim/32]
                     const float*     __restrict__ x,         // [in_dim]
                     float*           __restrict__ y,         // [out_dim]
                     unsigned int in_dim,
                     unsigned int out_dim)
{
    extern __shared__ float smem[];
    const int row = blockIdx.x;
    if (row >= (int)out_dim) return;
    const int tid = threadIdx.x;
    const int bs  = blockDim.x;

    const unsigned int n_blocks = in_dim >> 5;  // /32
    const BlockQ8_0* row_blocks = w_blocks + (size_t)row * (size_t)n_blocks;

    float acc = 0.0f;
    for (int b = tid; b < (int)n_blocks; b += bs) {
        const BlockQ8_0* blk = row_blocks + b;
        // Reinterpret the raw u16 as fp16 and convert to fp32 in registers.
        const __half d_h = *reinterpret_cast<const __half*>(&blk->d);
        const float d    = __half2float(d_h);

        const float* xb = x + (size_t)b * 32;
        float partial = 0.0f;
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            partial += (float)blk->qs[i] * xb[i];
        }
        acc += d * partial;
    }

    smem[tid] = acc;
    __syncthreads();
    for (int s = bs / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) y[row] = smem[0];
}
