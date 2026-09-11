// Gather one block's residual-stream output into a slot of the DFlash
// target-context buffer.
//
// DFlash conditions its drafter on hidden states from a fixed set of
// target layers, concatenated along the FEATURE axis per position:
//
//   ctx[t] = [ H(l0)[t] ; H(l1)[t] ; ... ; H(l_{n-1})[t] ]
//
// so the buffer is [P, n_taps * hidden] and each tapped layer owns a
// contiguous `hidden`-wide column slice of every row. That is a strided
// scatter, not a flat copy — hence a kernel rather than a memcpy, which
// also keeps it capture-safe inside the prefill HIP graph.
//
// grid = (ceil(hidden/256), P); block = 256.

#include <hip/hip_runtime.h>
#include <stdint.h>

extern "C" __global__
void tap_copy_f32(const float* __restrict__ src,   // [p, hidden] residual stream
                  float*       __restrict__ dst,   // [p, n_taps * hidden]
                  unsigned int hidden,
                  unsigned int dst_stride,         // n_taps * hidden
                  unsigned int slot,               // which tap, 0..n_taps-1
                  unsigned int p)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int t = blockIdx.y;
    if (i >= hidden || t >= p) return;
    dst[(size_t)t * dst_stride + (size_t)slot * hidden + i] = src[(size_t)t * hidden + i];
}
