// fp16 → int8 KV-tier demotion. Per-(token,head) symmetric quant with
// one fp32 scale per (token,head) — same shape as our existing int8 KV
// (matches kv_quant_prefill_f32's output but takes fp16 input).
//
// Used by SuperQuantKvCache to demote the oldest tokens of the Hot tier
// (fp16) into the head of the Warm tier (int8) at turn boundaries or
// when the Hot window slides forward.
//
// Layout:
//   src   [n_demote, n_kv, head_dim]                  fp16
//   dst_q [n_demote, n_kv, head_dim]                  int8
//   dst_s [n_demote, n_kv]                            fp32
//
// Grid: (n_demote, n_kv). Block: 256 threads (covers head_dim by stride).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <math.h>

extern "C" __global__
void kv_promote_fp16_to_q8_f32(const __half* __restrict__ src,
                                signed char* __restrict__ dst_q,
                                float*       __restrict__ dst_s,
                                unsigned int n_kv,
                                unsigned int head_dim)
{
    const unsigned int p = blockIdx.x;     // demote position
    const unsigned int h = blockIdx.y;     // head
    const int tid = threadIdx.x;
    const int bs  = blockDim.x;
    if (h >= n_kv) return;

    const size_t row_off = ((size_t)p * n_kv + h) * head_dim;
    const __half* sh = src + row_off;

    __shared__ float red[256];
    float a = 0.0f;
    for (int i = tid; i < (int)head_dim; i += bs) {
        a = fmaxf(a, fabsf(__half2float(sh[i])));
    }
    red[tid] = a;
    __syncthreads();
    for (int s = bs >> 1; s > 0; s >>= 1) {
        if (tid < s) red[tid] = fmaxf(red[tid], red[tid + s]);
        __syncthreads();
    }
    const float amax = red[0];
    const float scale = (amax > 0.0f) ? (amax / 127.0f) : 1.0f;
    const float inv   = (amax > 0.0f) ? (127.0f / amax) : 0.0f;

    signed char* dq = dst_q + row_off;
    for (int i = tid; i < (int)head_dim; i += bs) {
        int q = (int)rintf(__half2float(sh[i]) * inv);
        q = max(-127, min(127, q));
        dq[i] = (signed char)q;
    }
    if (tid == 0) dst_s[(size_t)p * n_kv + h] = scale;
}
