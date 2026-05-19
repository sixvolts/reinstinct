// fp32 matvec for gfx906 — 256-thread block (4 wavefronts) per output
// row.
//
// matvec_f32_wave64 uses one 64-thread wavefront per row, so a matvec
// with few rows (e.g. the 128-expert MoE router: out_dim 128) launches
// only ~128 wavefronts — about 2 per CU, far too few to hide HBM
// latency. A 256-thread block quadruples resident wavefronts; the four
// waves reduce through a 4-element LDS slot.

#include <hip/hip_runtime.h>

extern "C" __global__
void matvec_f32_b256(const float* __restrict__ w,
                     const float* __restrict__ x,
                     float*       __restrict__ y,
                     unsigned int in_dim,
                     unsigned int out_dim)
{
    __shared__ float wred[4];
    const int row = blockIdx.x;
    if (row >= (int)out_dim) return;
    const int tid  = threadIdx.x;        // 0..255
    const int lane = tid & 63;
    const int wave = tid >> 6;           // 0..3

    const float* wrow = w + (size_t)row * (size_t)in_dim;
    float acc = 0.0f;
    for (int i = tid; i < (int)in_dim; i += 256) {
        acc += wrow[i] * x[i];
    }

    acc += __shfl_xor(acc, 32);
    acc += __shfl_xor(acc, 16);
    acc += __shfl_xor(acc,  8);
    acc += __shfl_xor(acc,  4);
    acc += __shfl_xor(acc,  2);
    acc += __shfl_xor(acc,  1);

    if (lane == 0) wred[wave] = acc;
    __syncthreads();
    if (tid == 0) y[row] = wred[0] + wred[1] + wred[2] + wred[3];
}
