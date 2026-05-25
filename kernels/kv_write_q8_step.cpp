// Decode-step single-token write to an int8 KV cache slot.
//
// Per (n_kv head): compute the amax across head_dim, derive a per-(slot,
// head) fp32 scale, quantize, write 128-thread cooperative.
//
// Same shape as `kv_quant_prefill_f32` but for a single position
// (decode-time). Used by SuperQuant when writes land in the Warm tier
// (the only "active" tier in the 2-tier design — Hot fp16 was removed
// per user request 2026-05-25).
//
// Layout:
//   src   [n_kv, head_dim]                            fp32
//   dst_q [max_seq, n_kv, head_dim]                   int8  (slot `pos`)
//   dst_s [max_seq, n_kv]                             fp32  (slot `pos`)
// Grid: (n_kv). Block: 256 threads.

#include <hip/hip_runtime.h>
#include <math.h>

extern "C" __global__
void kv_write_q8_step_f32(const float* __restrict__ src,
                          signed char* __restrict__ dst_q,
                          float*       __restrict__ dst_s,
                          unsigned int n_kv,
                          unsigned int head_dim,
                          unsigned int pos,
                          unsigned int max_seq)
{
    const unsigned int h = blockIdx.x;
    if (h >= n_kv || pos >= max_seq) return;
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
    const float scale = (amax > 0.0f) ? (amax / 127.0f) : 1.0f;
    const float inv   = (amax > 0.0f) ? (127.0f / amax) : 0.0f;

    const size_t row_off = (size_t)pos * n_kv * head_dim + (size_t)h * head_dim;
    signed char* dq = dst_q + row_off;
    for (int i = tid; i < (int)head_dim; i += bs) {
        int q = (int)rintf(sh[i] * inv);
        q = max(-127, min(127, q));
        dq[i] = (signed char)q;
    }
    if (tid == 0) dst_s[(size_t)pos * n_kv + h] = scale;
}
