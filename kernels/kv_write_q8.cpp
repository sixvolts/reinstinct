// Quantize a normed K (or V) vector and append it to the int8 KV cache
// at row *pos_ptr. Each (token, head) gets a symmetric int8 quant(one
// f32 scale per head). Halves-then-quarters the KV cache vs f32, and
// lets the attention kernel dot K with a quantized Q via dp4a.
//
// grid = n_kv (one workgroup per head); block = 256.

#include <hip/hip_runtime.h>
#include <math.h>

extern "C" __global__
void kv_write_q8_f32(const float*        __restrict__ src,      // [n_kv, head_dim]
                     signed char*        __restrict__ dst_q,    // [max_seq, n_kv, head_dim]
                     float*              __restrict__ dst_s,    // [max_seq, n_kv]
                     const unsigned int* __restrict__ pos_ptr,
                     unsigned int n_kv,
                     unsigned int head_dim)
{
    const unsigned int h   = blockIdx.x;
    const unsigned int pos = *pos_ptr;
    const int tid = threadIdx.x;
    const int bs  = blockDim.x;
    const float* sh = src + (size_t)h * head_dim;

    __shared__ float red[256];
    float a = 0.0f;
    for (int i = tid; i < (int)head_dim; i += bs) a = fmaxf(a, fabsf(sh[i]));
    red[tid] = a;
    __syncthreads();
    for (int s = bs >> 1; s > 0; s >>= 1) {
        if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
        __syncthreads();
    }
    const float amax  = red[0];
    const float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
    const float inv   = amax > 0.0f ? 127.0f / amax : 0.0f;

    signed char* dq = dst_q + ((size_t)pos * n_kv + h) * head_dim;
    for (int i = tid; i < (int)head_dim; i += bs) {
        int q = (int)rintf(sh[i] * inv);
        q = max(-127, min(127, q));
        dq[i] = (signed char)q;
    }
    if (tid == 0) dst_s[(size_t)pos * n_kv + h] = scale;
}
