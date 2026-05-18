// Shared-expert sigmoid gate (Qwen MoE): scales the shared-expert
// output in place by sigmoid(dot(hidden, gate_w)).
//
//   sh_out[i] *= sigmoid( sum_k hidden[k]·gate_w[k] )
//
// One workgroup per token (grid.x = n_tok; decode launches grid.x = 1),
// block = 256, dynamic shared mem of blockDim.x floats for the
// dot-product reduction. `gate_w` is shared across all tokens.

#include <hip/hip_runtime.h>

extern "C" __global__
void moe_shexp_gate_f32(float*       __restrict__ sh_out,   // [n] in/out
                        const float* __restrict__ hidden,   // [n]
                        const float* __restrict__ gate_w,   // [n]
                        unsigned int n)
{
    extern __shared__ float smem[];
    const int tid = threadIdx.x;
    const int bs  = blockDim.x;

    // One block per token; gate_w is shared across tokens.
    sh_out += (size_t)blockIdx.x * n;
    hidden += (size_t)blockIdx.x * n;

    float d = 0.0f;
    for (int i = tid; i < (int)n; i += bs) d += hidden[i] * gate_w[i];
    smem[tid] = d;
    __syncthreads();
    for (int s = bs / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    const float g = 1.0f / (1.0f + __expf(-smem[0]));
    for (int i = tid; i < (int)n; i += bs) sh_out[i] *= g;
}
