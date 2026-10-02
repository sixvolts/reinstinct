// Batched per-(token,head) int8 quantization of the prefill's freshly
// projected K or V, written straight into the decode KV cache.
//
// The decode path appends one int8 (token,head) row at a time via
// kv_write_q8; this is the same symmetric per-head quant (one f32 scale
// per (token,head)), batched over all P prompt tokens. Writing the
// first P rows of a [max_seq, n_kv, head_dim] cache lets a batched
// prefill hand decode a populated cache to continue from.
//
//   src   : [P, n_kv, head_dim] f32
//   dst_q : [max_seq, n_kv, head_dim] int8   (rows 0..P written)
//   dst_s : [max_seq, n_kv]           f32    (rows 0..P written)
//   grid = (n_kv, P); block = 256.

#include <hip/hip_runtime.h>
#include <math.h>
#include "gfx906_dpp.h"

extern "C" __global__
void kv_quant_prefill_f32(const float* __restrict__ src,
                          signed char* __restrict__ dst_q,
                          float*       __restrict__ dst_s,
                          unsigned int n_kv,
                          unsigned int head_dim)
{
    const unsigned int h = blockIdx.x;
    const unsigned int p = blockIdx.y;
    const int tid = threadIdx.x;
    const int bs  = blockDim.x;
    const float* sh = src + ((size_t)p * n_kv + h) * head_dim;

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
    const float inv   = amax > 0.0f ? 127.0f * fast_rcp_f32(amax) : 0.0f;

    signed char* dq = dst_q + ((size_t)p * n_kv + h) * head_dim;
    for (int i = tid; i < (int)head_dim; i += bs) {
        int q = (int)rintf(sh[i] * inv);
        q = max(-127, min(127, q));
        dq[i] = (signed char)q;
    }
    if (tid == 0) dst_s[(size_t)p * n_kv + h] = scale;
}

// Variant that reads `base_pos` from a device-resident uint32, then
// writes K = grid.y rows at slots [base_pos, base_pos + K). Used by
// verify_forward when captured into a HIP graph — the per-call
// base_pos lives in `v_base_pos` and gets updated host-side between
// graph replays without re-capturing.
extern "C" __global__
void kv_quant_prefill_offset_f32(const float* __restrict__ src,
                                 signed char* __restrict__ dst_q_base,
                                 float*       __restrict__ dst_s_base,
                                 const unsigned int* __restrict__ base_pos_ptr,
                                 unsigned int n_kv,
                                 unsigned int head_dim)
{
    const unsigned int h = blockIdx.x;
    const unsigned int p = blockIdx.y;
    const int tid = threadIdx.x;
    const int bs  = blockDim.x;
    const float* sh = src + ((size_t)p * n_kv + h) * head_dim;

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
    const float inv   = amax > 0.0f ? 127.0f * fast_rcp_f32(amax) : 0.0f;

    const unsigned int base_pos = *base_pos_ptr;
    const size_t row_idx = (size_t)(base_pos + p);
    signed char* dq = dst_q_base + (row_idx * n_kv + h) * head_dim;
    for (int i = tid; i < (int)head_dim; i += bs) {
        int q = (int)rintf(sh[i] * inv);
        q = max(-127, min(127, q));
        dq[i] = (signed char)q;
    }
    if (tid == 0) dst_s_base[row_idx * n_kv + h] = scale;
}

// Inverse of kv_quant_prefill_f32 over a row range: int8 cache rows
// [row0, row0 + grid.y) back to f32. Lets a prefill that continues
// from a restored prefix (base_pos > 0) hand the prefill attention
// kernel the cached prefix as f32 — the same int8 values decode reads.
//
//   src_q : [max_seq, n_kv, head_dim] int8
//   src_s : [max_seq, n_kv]           f32
//   dst   : [grid.y, n_kv, head_dim]  f32   (row r = cache row row0 + r)
//   grid = (n_kv, n_rows); block = 256.
extern "C" __global__
void kv_dequant_rows_f32(const signed char* __restrict__ src_q,
                         const float*       __restrict__ src_s,
                         float*             __restrict__ dst,
                         unsigned int n_kv,
                         unsigned int head_dim,
                         unsigned int row0)
{
    const unsigned int h = blockIdx.x;
    const unsigned int r = blockIdx.y;
    const size_t src_row = (size_t)(row0 + r) * n_kv + h;
    const float s = src_s[src_row];
    const signed char* sq = src_q + src_row * head_dim;
    float* d = dst + ((size_t)r * n_kv + h) * head_dim;
    for (unsigned int i = threadIdx.x; i < head_dim; i += blockDim.x)
        d[i] = (float)sq[i] * s;
}
