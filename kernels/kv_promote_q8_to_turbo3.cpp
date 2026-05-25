// int8 → turbo3 KV-tier demotion. Reads int8 [N, n_kv, head_dim] +
// per-(token,head) fp32 scale, dequants to fp32, then runs the standard
// turbo3 encode pipeline (RHT + Lloyd-Max + L2-preserving norm correction).
//
// Used by SuperQuantKvCache to demote the oldest tokens of the Warm tier
// (int8) into the head of the Cold tier (turbo3).
//
// Grid: (n_demote * n_kv, groups_per_head) — one block per (slot, head,
// rotation group). Block: 128 threads — one per value of the 128-element
// rotation group.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

constexpr int ROT_GROUP        = 128;
constexpr int BLOCK_SIZE       = 32;
constexpr int BYTES_PER_BLOCK  = 16;
constexpr int BLOCKS_PER_GROUP = ROT_GROUP / BLOCK_SIZE;
constexpr int BYTES_PER_GROUP  = BLOCKS_PER_GROUP * BYTES_PER_BLOCK;

__device__ __constant__ float CENTROIDS_P3[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};

__device__ __forceinline__ uint8_t classify_p3(float v) {
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
void kv_promote_q8_to_turbo3_f32(const signed char* __restrict__ src_q,
                                 const float*       __restrict__ src_s,
                                 const int8_t*      __restrict__ signs1,
                                 const int8_t*      __restrict__ signs2,
                                 unsigned char*     __restrict__ dst,
                                 unsigned int n_demote,
                                 unsigned int n_kv,
                                 unsigned int head_dim)
{
    const unsigned int groups_per_head = head_dim / ROT_GROUP;
    const unsigned int slot_head = blockIdx.x;
    const unsigned int grp       = blockIdx.y;
    if (slot_head >= n_demote * n_kv || grp >= groups_per_head) return;

    const unsigned int slot = slot_head / n_kv;
    const unsigned int head = slot_head % n_kv;

    const int lane = threadIdx.x;

    __shared__ float x[ROT_GROUP];
    __shared__ float red[ROT_GROUP];
    __shared__ uint8_t codes[ROT_GROUP];

    // Read one int8 + scale → fp32.
    const size_t src_off = ((size_t)slot * n_kv + head) * head_dim + grp * ROT_GROUP + lane;
    const float scale = src_s[(size_t)slot * n_kv + head];
    const float v_in  = (float)src_q[src_off] * scale;
    x[lane]   = v_in;
    red[lane] = v_in * v_in;
    __syncthreads();

    // L2 norm of the 128-element rotation group.
    for (int s = ROT_GROUP >> 1; s > 0; s >>= 1) {
        if (lane < s) red[lane] += red[lane + s];
        __syncthreads();
    }
    const float grp_norm = sqrtf(red[0]);
    const float inv_norm = (grp_norm > 1e-10f) ? (1.0f / grp_norm) : 0.0f;

    // Normalize + signs1.
    x[lane] = v_in * inv_norm * (float)signs1[lane];
    __syncthreads();

    // FWHT-128.
    for (int h = 1; h < ROT_GROUP; h *= 2) {
        if ((lane & h) == 0) {
            const float a = x[lane];
            const float b = x[lane + h];
            x[lane]     = a + b;
            x[lane + h] = a - b;
        }
        __syncthreads();
    }
    const float fwht_scale = 1.0f / sqrtf((float)ROT_GROUP);
    const float rotated = x[lane] * fwht_scale * (float)signs2[lane];
    __syncthreads();

    // Classify + reconstruction norm.
    const uint8_t code = classify_p3(rotated);
    codes[lane] = code;
    red[lane]   = CENTROIDS_P3[code] * CENTROIDS_P3[code];
    __syncthreads();
    for (int s = ROT_GROUP >> 1; s > 0; s >>= 1) {
        if (lane < s) red[lane] += red[lane + s];
        __syncthreads();
    }
    const float recon_norm = sqrtf(red[0]);
    const float corrected  = (recon_norm > 1e-10f) ? (grp_norm / recon_norm) : grp_norm;

    // Pack 4 turbo3 blocks. Same layout as kv_write_turbo3.cpp.
    const size_t dst_off = ((size_t)slot * n_kv + head) * groups_per_head * BYTES_PER_GROUP
                         + grp * BYTES_PER_GROUP;
    unsigned char* dst_grp = dst + dst_off;

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
