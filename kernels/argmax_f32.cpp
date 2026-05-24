// Single-block argmax over a logits vector. Output is one `int2`-like
// (idx, packed val as int) pair on device — caller D2Hs 8 bytes instead
// of the whole vocab.
//
// Saves the per-step D2H of the full vocab during spec-decode draft
// loops: at qwen 27B vocab ≈ 152K × fp32 = 608 KB per draft step, × K
// × rounds. The host-side argmax is then on those 152K floats. Doing
// argmax on-device replaces both with a one-block reduce + an 8-byte
// D2H, freeing the CPU and PCIe lane for actual work.
//
// Tolerates NaN/+inf/-inf in the input (treated as -inf), matching
// our host argmax in sampling.rs.

#include <hip/hip_runtime.h>
#include <math.h>

extern "C" __global__
void argmax_f32(const float* __restrict__ logits,
                unsigned int n,
                int* __restrict__ out_idx,
                float* __restrict__ out_val)
{
    extern __shared__ float smem[];
    int*   smem_idx = reinterpret_cast<int*>(smem + blockDim.x);   // overlay
    const int t  = threadIdx.x;
    const int bs = blockDim.x;

    // Per-thread argmax over a strided slice. NaN/+inf get masked out
    // by the is_finite filter — argmax over the finite tail.
    float best_v = -INFINITY;
    int   best_i = 0;
    for (int i = t; i < (int)n; i += bs) {
        float v = logits[i];
        if (!isfinite(v)) continue;
        if (v > best_v) { best_v = v; best_i = i; }
    }
    smem[t]     = best_v;
    smem_idx[t] = best_i;
    __syncthreads();

    // Tree reduce. Ties broken by the lower index (idx-min tiebreak so
    // the reduction order doesn't change argmax).
    for (int s = bs >> 1; s > 0; s >>= 1) {
        if (t < s) {
            const float rv = smem[t + s];
            const int   ri = smem_idx[t + s];
            const float lv = smem[t];
            const int   li = smem_idx[t];
            if (rv > lv || (rv == lv && ri < li)) {
                smem[t]     = rv;
                smem_idx[t] = ri;
            }
        }
        __syncthreads();
    }

    if (t == 0) {
        *out_idx = smem_idx[0];
        *out_val = smem[0];
    }
}
