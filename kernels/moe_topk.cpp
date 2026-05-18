// MoE router: softmax over the expert logits, then top-k selection
// with the selected weights renormalised to sum to 1.
//
// One workgroup per token (grid.x = n_tok; decode launches grid.x = 1).
// The softmax is computed cooperatively; the top-k scan runs on thread 0
// (n_expert is small — 128 — so 8 serial argmax passes is negligible and
// avoids a fiddly parallel reduction).
//
// out_ids[k]     = expert index of the k-th largest probability
// out_weights[k] = its softmax probability, renormalised over the k used

#include <hip/hip_runtime.h>
#include <math.h>

extern "C" __global__
void moe_topk_f32(const float* __restrict__ logits,
                  int n_expert, int n_used,
                  int*   __restrict__ out_ids,
                  float* __restrict__ out_weights)
{
    extern __shared__ float probs[];        // n_expert floats
    __shared__ float red[256];
    const int t  = threadIdx.x;
    const int nt = blockDim.x;

    // One block per token: offset logits / outputs by the token index.
    logits      += (size_t)blockIdx.x * n_expert;
    out_ids     += (size_t)blockIdx.x * n_used;
    out_weights += (size_t)blockIdx.x * n_used;

    // load + block-max
    float local_max = -INFINITY;
    for (int i = t; i < n_expert; i += nt) {
        probs[i] = logits[i];
        local_max = fmaxf(local_max, logits[i]);
    }
    red[t] = local_max;
    __syncthreads();
    for (int s = nt >> 1; s > 0; s >>= 1) {
        if (t < s) red[t] = fmaxf(red[t], red[t + s]);
        __syncthreads();
    }
    const float m = red[0];
    __syncthreads();

    // exp + block-sum
    float local_sum = 0.0f;
    for (int i = t; i < n_expert; i += nt) {
        const float e = expf(probs[i] - m);
        probs[i] = e;
        local_sum += e;
    }
    red[t] = local_sum;
    __syncthreads();
    for (int s = nt >> 1; s > 0; s >>= 1) {
        if (t < s) red[t] += red[t + s];
        __syncthreads();
    }
    const float sum = red[0];
    __syncthreads();
    for (int i = t; i < n_expert; i += nt) probs[i] /= sum;
    __syncthreads();

    // top-k on thread 0
    if (t == 0) {
        float wsum = 0.0f;
        for (int k = 0; k < n_used; k++) {
            int   best = 0;
            float bv   = -1.0f;
            for (int i = 0; i < n_expert; i++) {
                if (probs[i] > bv) { bv = probs[i]; best = i; }
            }
            out_ids[k]     = best;
            out_weights[k] = bv;
            wsum += bv;
            probs[best] = -1.0f;             // exclude from later passes
        }
        if (wsum < 6.103515625e-5f) wsum = 6.103515625e-5f;
        for (int k = 0; k < n_used; k++) out_weights[k] /= wsum;
    }
}
