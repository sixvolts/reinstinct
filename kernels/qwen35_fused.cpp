// Fused elementwise / norm / quantize kernels for the qwen35 decode
// step. Each one replaces two to four launches that were each a few
// microseconds of work behind ~5-8 µs of dispatch, ramp and drain — at
// ~1650 launches per token that overhead was ~10% of a 27B's decode.
//
// Every kernel that produces a matvec input writes BOTH the fp32 vector
// (for the non-dp4a fallbacks: F16 weights, dp4a disabled) and its
// BlockQ8 quantisation (what the repacked int8 matvecs read), so the
// launch that follows never needs a separate quantize.
#include <hip/hip_runtime.h>
#include <stdint.h>
#include "gfx906_dpp.h"

struct __attribute__((packed)) BlockQ8 {
    float  d;
    float  xsum;
    int8_t qs[32];
};
static_assert(sizeof(BlockQ8) == 40, "BlockQ8 must be 40 bytes");

// Quantise one 32-value sub-block held one value per lane of a 32-lane
// group (lanes 0-31 or 32-63 of a wave64; xor-shuffles below 32 stay
// inside the group).
__device__ __forceinline__ void q8_store_sub(BlockQ8* __restrict__ out, int lane, float v) {
    float amax = fabsf(v);
    float vsum = v;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor(amax, o));
        vsum += __shfl_xor(vsum, o);
    }
    const float inv = amax > 0.0f ? 127.0f * fast_rcp_f32(amax) : 0.0f;
    int q = (int)rintf(v * inv);
    q = max(-127, min(127, q));
    out->qs[lane] = (int8_t)q;
    if (lane == 0) { out->d = amax > 0.0f ? amax / 127.0f : 1.0f; out->xsum = vsum; }
}

// Block-wide sum of `v` (block = blockDim.x threads, `red` has blockDim.x
// floats). Returns the total to every thread.
__device__ __forceinline__ float block_sum(float v, float* red) {
    const int tid = threadIdx.x, bs = blockDim.x;
    red[tid] = v;
    __syncthreads();
    for (int s = bs / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    const float t = red[0];
    __syncthreads();
    return t;
}

// hidden (+= add)  ->  y = rmsnorm(hidden) * w  ->  q8(y).
// One workgroup of 1024 threads (16 waves — the vector's memory traffic
// needs the parallelism; at 256 threads this was a 35 µs kernel).
// Thread t holds elements t, t+1024, ... and their `w` in registers, so
// 32-lane group j holds sub-blocks j, j+32, ... whole and quantises them
// from registers: one global read of the vector and of w, no LDS
// staging, and the sum of squares reduces with a DPP wave reduction
// plus one LDS exchange instead of a ten-step barrier tree.
// `add` may be null. Replaces add_inplace + rmsnorm + quantize_q8.
#define ARQ_THREADS 1024
#define ARQ_KMAX    8           // n <= 8192
extern "C" __global__ __launch_bounds__(ARQ_THREADS)
void add_rmsnorm_q8_f32(float*       __restrict__ hidden,
                        const float* __restrict__ add,
                        const float* __restrict__ w,
                        float*       __restrict__ y,
                        BlockQ8*     __restrict__ out,
                        unsigned int n,
                        float        eps)
{
    __shared__ float red[ARQ_THREADS / 64];
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int nk   = (int)n / ARQ_THREADS;          // elements per thread
    float v[ARQ_KMAX], wv[ARQ_KMAX];
    float sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < ARQ_KMAX; k++) {
        if (k < nk) {
            const int i = k * ARQ_THREADS + tid;
            float x = hidden[i];
            wv[k] = w[i];
            if (add) { x += add[i]; hidden[i] = x; }
            v[k] = x;
            sum += x * x;
        }
    }
    sum = wave64_reduce_add_f32(sum);
    if ((tid & 63) == 0) red[tid >> 6] = sum;
    __syncthreads();
    float tot = 0.0f;
    #pragma unroll
    for (int i = 0; i < ARQ_THREADS / 64; i++) tot += red[i];
    const float rrms = rsqrtf(tot / (float)n + eps);
    #pragma unroll
    for (int k = 0; k < ARQ_KMAX; k++) {
        if (k < nk) {
            const int i = k * ARQ_THREADS + tid;
            const float o = v[k] * rrms * wv[k];
            y[i] = o;
            q8_store_sub(out + (i >> 5), lane, o);
        }
    }
}

// out = silu(gate) * up, also quantised. Replaces swiglu + quantize_q8.
// Grid: ceil(n/256) blocks of 256 = 8 sub-blocks each.
extern "C" __global__
void swiglu_q8_f32(const float* __restrict__ gate,
                   const float* __restrict__ up,
                   float*       __restrict__ out,
                   BlockQ8*     __restrict__ q8,
                   unsigned int n)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int sb = i >> 5;
    if (i >= n) return;
    const float g = gate[i];
    const float v = g / (1.0f + __expf(-g)) * up[i];
    out[i] = v;
    q8_store_sub(q8 + sb, threadIdx.x & 31, v);
}

// x *= sigmoid(gate), also quantised. Replaces sigmoid_mul + quantize_q8.
extern "C" __global__
void sigmoid_mul_q8_f32(float*       __restrict__ x,
                        const float* __restrict__ gate,
                        BlockQ8*     __restrict__ q8,
                        unsigned int n)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = gate[i];
    const float v = x[i] / (1.0f + __expf(-g));
    x[i] = v;
    q8_store_sub(q8 + (i >> 5), threadIdx.x & 31, v);
}

// GDN: causal conv1d + SiLU over the [Q | K | V] projection, with the
// per-head L2 norm of Q (scaled by q_scale) and K fused in. Grid =
// 2*n_k_heads + n_heads blocks of head_dim threads: Q heads, then K
// heads (normalised into q_out / k_out), then V heads (SiLU output into
// conv_out's V region, where the recurrent step reads it). Replaces
// conv1d_step_silu + l2norm_qk.
extern "C" __global__
void conv1d_l2norm_f32(const float* __restrict__ x_new,     // [conv_dim]
                       const float* __restrict__ w,         // [conv_dim, K]
                       float*       __restrict__ history,   // [conv_dim, K-1] in/out
                       float*       __restrict__ conv_out,  // [conv_dim] (V region written)
                       float*       __restrict__ q_out,     // [n_k_heads, head_dim]
                       float*       __restrict__ k_out,     // [n_k_heads, head_dim]
                       unsigned int n_k_heads,
                       unsigned int head_dim,
                       unsigned int key_dim,
                       unsigned int kernel_size,
                       float        eps,
                       float        q_scale)
{
    extern __shared__ float smem[];
    const unsigned int b = blockIdx.x;
    const int tid = threadIdx.x;
    unsigned int base; int side;          // 0 = Q, 1 = K, 2 = V
    if (b < n_k_heads)          { side = 0; base = b * head_dim; }
    else if (b < 2 * n_k_heads) { side = 1; base = key_dim + (b - n_k_heads) * head_dim; }
    else                        { side = 2; base = 2 * key_dim + (b - 2 * n_k_heads) * head_dim; }
    const unsigned int ch = base + tid;
    const unsigned int hist_w = kernel_size - 1;
    const float* w_ch = w + (size_t)ch * kernel_size;
    float*       h_ch = history + (size_t)ch * hist_w;
    float acc = w_ch[kernel_size - 1] * x_new[ch];
    for (int k = 0; k < (int)hist_w; k++) acc += w_ch[k] * h_ch[k];
    const float y = acc / (1.0f + __expf(-acc));
    for (int k = 0; k + 1 < (int)hist_w; k++) h_ch[k] = h_ch[k + 1];
    if (hist_w >= 1) h_ch[hist_w - 1] = x_new[ch];
    if (side == 2) { conv_out[ch] = y; return; }
    const float ss = block_sum(y * y, smem);
    const float scale = rsqrtf(ss + eps) * (side == 0 ? q_scale : 1.0f);
    const unsigned int h = (side == 0) ? b : b - n_k_heads;
    (side == 0 ? q_out : k_out)[(size_t)h * head_dim + tid] = y * scale;
}

// GDN: per-head gated RMSNorm (y = rmsnorm(x) * w * silu(z)), also
// quantised for the ssm_out matvec. Grid = n_heads blocks of head_dim
// threads (head_dim a multiple of 32). Replaces rmsnorm_gated_multihead
// + quantize_q8.
extern "C" __global__
void rmsnorm_gated_q8_f32(const float* __restrict__ x,
                          const float* __restrict__ z,
                          const float* __restrict__ w,
                          float*       __restrict__ y,
                          BlockQ8*     __restrict__ q8,
                          unsigned int n_heads,
                          unsigned int head_dim,
                          float        eps)
{
    extern __shared__ float smem[];
    const unsigned int h = blockIdx.x;
    const int tid = threadIdx.x;
    const size_t i = (size_t)h * head_dim + tid;
    const float xv = x[i];
    const float rrms = rsqrtf(block_sum(xv * xv, smem) / (float)head_dim + eps);
    const float zg = z[i];
    const float v = xv * rrms * w[tid] * (zg / (1.0f + __expf(-zg)));
    y[i] = v;
    q8_store_sub(q8 + (i >> 5), tid & 31, v);
}

// Per-head RMSNorm then RoPE on `vals` held in LDS; writes head `h` of
// `dst` (row stride head_dim). Threads: head_dim.
__device__ __forceinline__
void norm_rope_head(float v, const float* __restrict__ w, float* smem,
                    const float* __restrict__ cr, const float* __restrict__ sr,
                    unsigned int head_dim, unsigned int rotary_dim, float eps,
                    float* __restrict__ dst)
{
    float* vals = smem;
    float* red  = smem + head_dim;
    const int tid = threadIdx.x;
    const float rrms = rsqrtf(block_sum(v * v, red) / (float)head_dim + eps);
    vals[tid] = v * rrms * w[tid];
    __syncthreads();
    const unsigned int half = rotary_dim >> 1;
    if ((unsigned)tid < half) {
        const float a = vals[tid], b = vals[tid + half];
        dst[tid]        = a * cr[tid]        - b * sr[tid];
        dst[tid + half] = b * cr[tid + half] + a * sr[tid + half];
    } else if ((unsigned)tid >= rotary_dim) {
        dst[tid] = vals[tid];
    }
}

// Full attention, Q side: split the [Q | gate] projection per head,
// RMSNorm + RoPE the Q half into q_out, copy the gate half to gate_out.
// Grid = n_heads blocks of head_dim threads; LDS = 2*head_dim floats.
// Replaces split_q_gate + rmsnorm_multihead + rope.
extern "C" __global__
void q_split_norm_rope_f32(const float* __restrict__ q_raw,    // [n_heads, 2*head_dim]
                           float*       __restrict__ q_out,    // [n_heads, head_dim]
                           float*       __restrict__ gate_out, // [n_heads, head_dim]
                           const float* __restrict__ w,        // [head_dim]
                           const float* __restrict__ cos,      // [max_seq, rotary_dim]
                           const float* __restrict__ sin,
                           const unsigned int* __restrict__ pos_ptr,
                           unsigned int head_dim,
                           unsigned int rotary_dim,
                           float        eps)
{
    extern __shared__ float smem[];
    const unsigned int h = blockIdx.x;
    const int tid = threadIdx.x;
    const float* src = q_raw + (size_t)h * 2 * head_dim;
    gate_out[(size_t)h * head_dim + tid] = src[head_dim + tid];
    const unsigned int pos = *pos_ptr;
    norm_rope_head(src[tid], w, smem, cos + (size_t)pos * rotary_dim, sin + (size_t)pos * rotary_dim,
                   head_dim, rotary_dim, eps, q_out + (size_t)h * head_dim);
}

// Full attention, K/V side: RMSNorm + RoPE each K head straight into
// the K cache row for this position, and copy V into the V cache row.
// Grid = 2*n_kv_heads blocks (K heads, then V heads) of head_dim
// threads. Replaces rmsnorm_multihead + rope + 2x kv_write.
extern "C" __global__
void k_norm_rope_kv_write_f32(const float* __restrict__ k_raw,   // [n_kv_heads, head_dim]
                              const float* __restrict__ v_raw,
                              const float* __restrict__ w,       // [head_dim]
                              const float* __restrict__ cos,
                              const float* __restrict__ sin,
                              const unsigned int* __restrict__ pos_ptr,
                              float*       __restrict__ k_cache, // [max_seq, kv_dim]
                              float*       __restrict__ v_cache,
                              unsigned int n_kv_heads,
                              unsigned int head_dim,
                              unsigned int rotary_dim,
                              float        eps)
{
    extern __shared__ float smem[];
    const unsigned int b = blockIdx.x;
    const int tid = threadIdx.x;
    const unsigned int pos = *pos_ptr;
    const size_t kv_dim = (size_t)n_kv_heads * head_dim;
    if (b >= n_kv_heads) {
        const unsigned int h = b - n_kv_heads;
        v_cache[pos * kv_dim + (size_t)h * head_dim + tid] = v_raw[(size_t)h * head_dim + tid];
        return;
    }
    norm_rope_head(k_raw[(size_t)b * head_dim + tid], w, smem,
                   cos + (size_t)pos * rotary_dim, sin + (size_t)pos * rotary_dim,
                   head_dim, rotary_dim, eps, k_cache + pos * kv_dim + (size_t)b * head_dim);
}
