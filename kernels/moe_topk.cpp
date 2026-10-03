// MoE router: softmax over the expert logits, then top-k selection
// with the selected weights renormalised to sum to 1.
//
// One workgroup per token (grid.x = n_tok; decode launches grid.x = 1).
//
// Softmax is monotonic and the renormalisation over the k selected
// cancels its denominator, so the kernel ranks the raw logits and only
// exponentiates the k winners:
//   w_k = exp(l_k - l_max) / sum_{j selected} exp(l_j - l_max).
// (The old clamp of the selected-probability sum at 2^-14 never binds:
// the top k of n hold at least k/n of the mass.)
//
// Top-k is a parallel rank (crossport L6, fork 4a8dee414): every
// expert counts the experts that beat it — larger logit, then lower
// id — against all keys in LDS, and the ones with rank < n_used write
// themselves to that slot. The (logit, id) pair is folded into one
// 64-bit key so each comparison is a single unsigned compare. This
// replaces softmax block reductions plus n_used argmax rounds, each a
// tree with a barrier per level: 20 -> 7 us per call on gfx906.
//
// out_ids[k]     = expert index of the k-th largest logit
// out_weights[k] = its softmax probability, renormalised over the k used

#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>

// Order-preserving map from float to uint32 (NaN already replaced).
__device__ __forceinline__ uint32_t ordered_bits(float v) {
    const uint32_t u = __float_as_uint(v);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

extern "C" __global__ __launch_bounds__(256)
void moe_topk_f32(const float* __restrict__ logits,
                  int n_expert, int n_used,
                  int*   __restrict__ out_ids,
                  float* __restrict__ out_weights)
{
    extern __shared__ __attribute__((aligned(16))) uint64_t keys[];   // n_expert keys (read 16 B at a time)
    __shared__ float chosen[64];             // n_used <= 64
    const int t  = threadIdx.x;
    const int nt = blockDim.x;

    logits      += (size_t)blockIdx.x * n_expert;
    out_ids     += (size_t)blockIdx.x * n_used;
    out_weights += (size_t)blockIdx.x * n_used;

    // key = ordered logit bits above the complemented id, so a larger
    // key is a larger logit, then a lower id. A NaN logit (numerical
    // blowup upstream) becomes -inf so it can never be selected ahead
    // of a real value.
    for (int i = t; i < n_expert; i += nt) {
        float v = logits[i];
        if (isnan(v)) v = -INFINITY;
        keys[i] = ((uint64_t)ordered_bits(v) << 32) | (uint32_t)(~i);
    }
    __syncthreads();

    for (int i = t; i < n_expert; i += nt) {
        const uint64_t k = keys[i];
        int rank = 0;
        // two keys per 16-byte LDS read, 8 reads in flight per step
        const ulonglong2* k2 = reinterpret_cast<const ulonglong2*>(keys);
        int j = 0;
        for (; j + 16 <= n_expert; j += 16) {
            ulonglong2 q[8];
            #pragma unroll
            for (int u = 0; u < 8; u++) q[u] = k2[(j >> 1) + u];
            #pragma unroll
            for (int u = 0; u < 8; u++) rank += (q[u].x > k) + (q[u].y > k);
        }
        for (; j < n_expert; j++) rank += keys[j] > k;
        if (rank < n_used) {
            out_ids[rank] = i;
            const uint32_t o = (uint32_t)(k >> 32);
            chosen[rank] = __uint_as_float((o & 0x80000000u) ? (o & 0x7FFFFFFFu) : ~o);
        }
    }
    __syncthreads();

    // renormalised softmax over the selected logits (chosen[0] is the max).
    if (t == 0) {
        const float m = chosen[0];
        float wsum = 0.0f;
        for (int k = 0; k < n_used; k++) {
            // equal-to-max counts as 1 so an all -inf (or +inf) row stays finite
            const float e = chosen[k] == m ? 1.0f : expf(chosen[k] - m);
            chosen[k] = e;
            wsum += e;
        }
        const float inv = 1.0f / wsum;
        for (int k = 0; k < n_used; k++) out_weights[k] = chosen[k] * inv;
    }
}
