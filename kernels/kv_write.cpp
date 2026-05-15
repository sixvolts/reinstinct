// Append a normed K (or V) vector into the per-layer KV cache at row
// `*pos_ptr`. Replaces a pos-offset hipMemcpy — as a kernel reading the
// position from a device buffer, the KV write becomes capturable into a
// parametric HIP graph (the same graph serves every decode position).
//
// grid = ceil(kv_dim / 256); block = 256.

#include <hip/hip_runtime.h>

extern "C" __global__
void kv_write_f32(const float*        __restrict__ src,
                  float*              __restrict__ dst,
                  const unsigned int* __restrict__ pos_ptr,
                  unsigned int kv_dim)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= kv_dim) return;
    dst[(size_t)(*pos_ptr) * kv_dim + i] = src[i];
}
