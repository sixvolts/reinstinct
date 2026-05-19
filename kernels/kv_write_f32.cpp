// Write one decode token's K (or V) row into the f32 KV cache at the
// device-resident position `*pos_ptr`. Replaces a host-offset memcpy so
// the qwen decode forward is parametric in position — capturable once
// into a HIP graph and replayed for every step.
//
//   cache[(*pos_ptr) * kv_dim + i] = src[i]   for i in [0, kv_dim)

#include <hip/hip_runtime.h>

extern "C" __global__
void kv_write_f32(const float*        __restrict__ src,      // [kv_dim]
                  float*              __restrict__ cache,    // [max_seq, kv_dim]
                  const unsigned int* __restrict__ pos_ptr,
                  unsigned int kv_dim)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= kv_dim) return;
    cache[(size_t)(*pos_ptr) * kv_dim + i] = src[i];
}
