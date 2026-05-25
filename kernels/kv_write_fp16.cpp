// Decode-step single-token write to a fp16 KV cache slot (Hot tier).
// Source is fp32 [n_kv * head_dim] from the model's K/V projection;
// destination is fp16 cache [max_seq, n_kv * head_dim] at slot `pos`.
//
// One thread per element (head_dim wide × n_kv tall). Grid: (n_kv).
// Block: head_dim threads (capped at 1024) — for head_dim ≤ 1024 the
// kernel covers the whole row in one launch; larger needs a stride loop.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

extern "C" __global__
void kv_write_fp16_step_f32(const float* __restrict__ src,    // [n_kv * head_dim]
                            __half*      __restrict__ cache,  // [max_seq, n_kv * head_dim]
                            unsigned int n_kv,
                            unsigned int head_dim,
                            unsigned int pos,
                            unsigned int max_seq)
{
    const unsigned int head = blockIdx.x;
    if (head >= n_kv || pos >= max_seq) return;
    const int tid = threadIdx.x;
    const int bs  = blockDim.x;

    const size_t src_off = (size_t)head * head_dim;
    const size_t dst_off = (size_t)pos * n_kv * head_dim + head * head_dim;
    for (int i = tid; i < (int)head_dim; i += bs) {
        cache[dst_off + i] = __float2half(src[src_off + i]);
    }
}
