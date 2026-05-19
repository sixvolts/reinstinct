// MoE router: softmax over the expert logits, then top-k selection
// with the selected weights renormalised to sum to 1.
//
// One workgroup per token (grid.x = n_tok; decode launches grid.x = 1).
// Both the softmax and the top-k run cooperatively across the block:
// the top-k is `n_used` rounds of a parallel block-wide argmax (the
// previous serial scan on thread 0 alone cost ~74 us/layer at decode —
// it left 255 threads idle and exposed LDS latency on a dependent
// chain).
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
    __shared__ int   ridx[256];
    __shared__ float chosen[64];            // n_used <= 64
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
    const float inv_sum = 1.0f / red[0];
    __syncthreads();
    for (int i = t; i < n_expert; i += nt) probs[i] *= inv_sum;
    __syncthreads();

    // top-k: n_used rounds of a parallel block-wide argmax.
    for (int k = 0; k < n_used; k++) {
        float bv = -1.0f; int bi = 0;
        for (int i = t; i < n_expert; i += nt) {
            const float v = probs[i];
            if (v > bv) { bv = v; bi = i; }
        }
        red[t] = bv; ridx[t] = bi;
        __syncthreads();
        for (int s = nt >> 1; s > 0; s >>= 1) {
            if (t < s && red[t + s] > red[t]) {
                red[t]  = red[t + s];
                ridx[t] = ridx[t + s];
            }
            __syncthreads();
        }
        if (t == 0) {
            out_ids[k]  = ridx[0];
            chosen[k]   = red[0];
            probs[ridx[0]] = -1.0f;        // exclude from later rounds
        }
        __syncthreads();
    }

    // renormalise the selected weights over their sum.
    if (t == 0) {
        float wsum = 0.0f;
        for (int k = 0; k < n_used; k++) wsum += chosen[k];
        if (wsum < 6.103515625e-5f) wsum = 6.103515625e-5f;
        for (int k = 0; k < n_used; k++) out_weights[k] = chosen[k] / wsum;
    }
}
