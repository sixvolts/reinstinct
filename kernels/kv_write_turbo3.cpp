// Decode-step single-token write to a turbo3 KV cache slot.
//
// Input:  src[n_kv * head_dim] f32 — one token's K (or V) projection
// Output: write into k_cache or v_cache at slot `pos`, packed turbo3
//
// Grid: (n_kv, groups_per_head) — one block per (head, rotation group).
// Block: 128 threads — one per value of the 128-element rotation group.
//
// Reuses the encode pipeline from turbo3_quantize.cpp but bakes in the
// per-(token,head,group) destination address: dst slot is at
// `cache_base + pos * (n_kv * head_dim/32 * 16) + head * (head_dim/32 * 16)`.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

constexpr int ROT_GROUP        = 128;
constexpr int BLOCK_SIZE       = 32;
constexpr int BYTES_PER_BLOCK  = 16;
constexpr int BLOCKS_PER_GROUP = ROT_GROUP / BLOCK_SIZE;
constexpr int BYTES_PER_GROUP  = BLOCKS_PER_GROUP * BYTES_PER_BLOCK;

__device__ __constant__ float CENTROIDS_W3[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};

__device__ __forceinline__ uint8_t classify_w3(float v) {
    uint8_t idx = 0;
    idx += (uint8_t)(v >= -0.154259f);
    idx += (uint8_t)(v >= -0.091775f);
    idx += (uint8_t)(v >= -0.043589f);
    idx += (uint8_t)(v >=  0.0f);
    idx += (uint8_t)(v >=  0.043589f);
    idx += (uint8_t)(v >=  0.091775f);
    idx += (uint8_t)(v >=  0.154259f);
    return idx;
}

extern "C" __global__ __launch_bounds__(128, 4)
void kv_write_turbo3_step_f32(const float*       __restrict__ src,    // [n_kv * head_dim]
                              const int8_t*      __restrict__ signs1, // [128]
                              const int8_t*      __restrict__ signs2,
                              unsigned char*     __restrict__ cache,  // [max_seq, n_kv, head_dim/32, 16]
                              unsigned int n_kv,
                              unsigned int head_dim,
                              unsigned int pos,
                              unsigned int max_seq)
{
    const unsigned int groups_per_head = head_dim / ROT_GROUP;
    const unsigned int head = blockIdx.x;
    const unsigned int grp  = blockIdx.y;
    if (head >= n_kv || grp >= groups_per_head || pos >= max_seq) return;

    const int lane = threadIdx.x;

    __shared__ float x[ROT_GROUP];
    __shared__ float red[ROT_GROUP];
    __shared__ uint8_t codes[ROT_GROUP];

    const float v_in = src[head * head_dim + grp * ROT_GROUP + lane];
    x[lane]   = v_in;
    red[lane] = v_in * v_in;
    __syncthreads();

    for (int s = ROT_GROUP >> 1; s > 0; s >>= 1) {
        if (lane < s) red[lane] += red[lane + s];
        __syncthreads();
    }
    const float grp_norm = sqrtf(red[0]);
    const float inv_norm = (grp_norm > 1e-10f) ? (1.0f / grp_norm) : 0.0f;

    x[lane] = v_in * inv_norm * (float)signs1[lane];
    __syncthreads();

    for (int h = 1; h < ROT_GROUP; h *= 2) {
        if ((lane & h) == 0) {
            const float a = x[lane];
            const float b = x[lane + h];
            x[lane]     = a + b;
            x[lane + h] = a - b;
        }
        __syncthreads();
    }
    const float scale = 1.0f / sqrtf((float)ROT_GROUP);
    const float rotated = x[lane] * scale * (float)signs2[lane];
    __syncthreads();

    const uint8_t code = classify_w3(rotated);
    codes[lane] = code;
    red[lane]   = CENTROIDS_W3[code] * CENTROIDS_W3[code];
    __syncthreads();
    for (int s = ROT_GROUP >> 1; s > 0; s >>= 1) {
        if (lane < s) red[lane] += red[lane + s];
        __syncthreads();
    }
    const float recon_norm = sqrtf(red[0]);
    const float corrected  = (recon_norm > 1e-10f) ? (grp_norm / recon_norm) : grp_norm;

    // Destination: cache[pos][head][grp][...] — 64 contiguous bytes.
    const size_t dst_off = ((size_t)pos * n_kv + head) * groups_per_head * BYTES_PER_GROUP
                         + grp * BYTES_PER_GROUP;
    unsigned char* dst_grp = cache + dst_off;

    if (lane < BLOCKS_PER_GROUP * 2) {
        const int blk_idx  = lane >> 1;
        const int byte_off = lane & 1;
        const __half norm_h = __float2half(corrected);
        const uint16_t nb = *reinterpret_cast<const uint16_t*>(&norm_h);
        dst_grp[blk_idx * BYTES_PER_BLOCK + byte_off]
            = (byte_off == 0) ? (nb & 0xFF) : ((nb >> 8) & 0xFF);
    }
    if (lane < BLOCKS_PER_GROUP * 8) {
        const int blk_idx  = lane >> 3;
        const int byte_idx = lane & 7;
        const int base = blk_idx * BLOCK_SIZE + byte_idx * 4;
        unsigned char b = 0;
        b |= (codes[base + 0] & 0x3) << 0;
        b |= (codes[base + 1] & 0x3) << 2;
        b |= (codes[base + 2] & 0x3) << 4;
        b |= (codes[base + 3] & 0x3) << 6;
        dst_grp[blk_idx * BYTES_PER_BLOCK + 2 + byte_idx] = b;
    }
    if (lane < BLOCKS_PER_GROUP * 4) {
        const int blk_idx  = lane >> 2;
        const int byte_idx = lane & 3;
        const int base = blk_idx * BLOCK_SIZE + byte_idx * 8;
        unsigned char b = 0;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            const unsigned char hi = (codes[base + i] >> 2) & 0x1;
            b |= hi << i;
        }
        dst_grp[blk_idx * BYTES_PER_BLOCK + 10 + byte_idx] = b;
    }
    if (lane < BLOCKS_PER_GROUP * 2) {
        const int blk_idx  = lane >> 1;
        const int byte_off = (lane & 1) + 14;
        dst_grp[blk_idx * BYTES_PER_BLOCK + byte_off] = 0;
    }
}
