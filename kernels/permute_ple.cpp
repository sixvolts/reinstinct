// Permute the per-layer-embedding tensor from token-major
// [P][n_layer][np] to layer-major [n_layer][P][np], so each layer's
// [P, np] slice is contiguous for the batched per-layer PLE block in
// the Gemma 4 E4B prefill.

#include <hip/hip_runtime.h>
#include <stdint.h>

extern "C" __global__
void permute_ple_f32(const float* __restrict__ src,
                     float*       __restrict__ dst,
                     unsigned int P, unsigned int n_layer, unsigned int np)
{
    const unsigned int t = blockIdx.y;     // token
    const unsigned int l = blockIdx.z;     // layer
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= np || t >= P || l >= n_layer) return;
    dst[((size_t)l * P + t) * np + i] = src[((size_t)t * n_layer + l) * np + i];
}
