// Turbo3 encoder: f32 [N, head_dim] → packed turbo3 blocks (16 B / 32 vals).
//
// Pipeline per 128-element rotation group:
//   1. Compute L2 norm (stash for the L2-preserving correction)
//   2. Divide by norm (unit-norm vector)
//   3. RHT: multiply by signs1, FWHT-128, scale by 1/√128, multiply by signs2
//   4. Quantize each value to nearest of 8 Lloyd-Max centroids (3 bits)
//   5. Compute reconstruction norm; store grp_norm/recon_norm as the
//      block-replicated `norm` so dequant exactly preserves L2.
//
// Grid: (n_rows, groups_per_row) — one block per 128-group.
// Block: 128 threads — one per value, cooperative reductions.
//
// Sign tables are passed as device pointers, so K/V can share the kernel.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

constexpr int ROT_GROUP        = 128;
constexpr int BLOCK_SIZE       = 32;
constexpr int BYTES_PER_BLOCK  = 16;
constexpr int BLOCKS_PER_GROUP = ROT_GROUP / BLOCK_SIZE;

__device__ __constant__ float CENTROIDS_T3[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};

__device__ __forceinline__ uint8_t classify_turbo3(float v) {
    // 7 midpoints — branch-free chain: each (v >= midpoint) bumps idx.
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
void turbo3_encode_f32(const float*       __restrict__ src,    // [n_rows, head_dim]
                       const int8_t*      __restrict__ signs1, // [128]
                       const int8_t*      __restrict__ signs2, // [128]
                       unsigned char*     __restrict__ dst,    // [n_rows × groups × 4 × 16]
                       unsigned int n_rows,
                       unsigned int head_dim)
{
    const unsigned int groups_per_row = head_dim / ROT_GROUP;
    const unsigned int row = blockIdx.x;
    const unsigned int grp = blockIdx.y;
    if (row >= n_rows || grp >= groups_per_row) return;

    const int lane = threadIdx.x;

    __shared__ float x[ROT_GROUP];
    __shared__ float red[ROT_GROUP];
    __shared__ uint8_t codes[ROT_GROUP];

    // Load one value per thread.
    const float v_in = src[(size_t)row * head_dim + grp * ROT_GROUP + lane];
    x[lane]   = v_in;
    red[lane] = v_in * v_in;
    __syncthreads();

    // Reduce L2 norm² across the 128-wide group.
    for (int s = ROT_GROUP >> 1; s > 0; s >>= 1) {
        if (lane < s) red[lane] += red[lane + s];
        __syncthreads();
    }
    const float grp_norm = sqrtf(red[0]);
    const float inv_norm = (grp_norm > 1e-10f) ? (1.0f / grp_norm) : 0.0f;

    // Normalize + first sign flip.
    x[lane] = v_in * inv_norm * (float)signs1[lane];
    __syncthreads();

    // FWHT-128. Butterfly stride doubles each iter; lane in lower half
    // of each pair owns the add/sub. Re-sync between stages — output of
    // one stage feeds cross-thread reads in the next.
    for (int h = 1; h < ROT_GROUP; h *= 2) {
        if ((lane & h) == 0) {
            const float a = x[lane];
            const float b = x[lane + h];
            x[lane]     = a + b;
            x[lane + h] = a - b;
        }
        __syncthreads();
    }

    // 1/√128 scale + second sign flip.
    const float scale = 1.0f / sqrtf((float)ROT_GROUP);
    const float rotated = x[lane] * scale * (float)signs2[lane];
    __syncthreads();

    // Classify + stash code; reduce centroid² for the recon-norm
    // correction in parallel.
    const uint8_t code = classify_turbo3(rotated);
    codes[lane] = code;
    red[lane]   = CENTROIDS_T3[code] * CENTROIDS_T3[code];
    __syncthreads();
    for (int s = ROT_GROUP >> 1; s > 0; s >>= 1) {
        if (lane < s) red[lane] += red[lane + s];
        __syncthreads();
    }
    const float recon_norm = sqrtf(red[0]);
    const float corrected  = (recon_norm > 1e-10f) ? (grp_norm / recon_norm) : grp_norm;

    // Pack the 3-bit codes block-by-block. 4 blocks of 32 values each.
    //   qs byte b (0..7) holds 4 lower-2-bit codes (lanes 4b..4b+3 within block)
    //   signs byte b (0..3) holds 8 upper-1-bit codes (lanes 8b..8b+7 within block)
    // 8 + 4 = 12 packed bytes per block × 4 blocks = 48 lanes do the packing;
    // each packer lane walks its 4 or 8 contributing lanes' codes from `codes`.
    const size_t dst_off = ((size_t)row * groups_per_row + grp)
                         * BLOCKS_PER_GROUP * BYTES_PER_BLOCK;
    unsigned char* dst_grp = dst + dst_off;

    // norm: lanes 0..3 each write 2 bytes (one block's norm slot)
    if (lane < BLOCKS_PER_GROUP * 2) {
        const int blk_idx  = lane >> 1;
        const int byte_off = lane & 1;
        const __half norm_h = __float2half(corrected);
        const uint16_t nb = *reinterpret_cast<const uint16_t*>(&norm_h);
        dst_grp[blk_idx * BYTES_PER_BLOCK + byte_off]
            = (byte_off == 0) ? (nb & 0xFF) : ((nb >> 8) & 0xFF);
    }
    // qs bytes: 4 blocks × 8 bytes = 32 packer lanes (lanes 0..31).
    if (lane < BLOCKS_PER_GROUP * 8) {
        const int blk_idx  = lane >> 3;          // 0..3
        const int byte_idx = lane & 7;           // 0..7
        const int base = blk_idx * BLOCK_SIZE + byte_idx * 4;
        unsigned char b = 0;
        b |= (codes[base + 0] & 0x3) << 0;
        b |= (codes[base + 1] & 0x3) << 2;
        b |= (codes[base + 2] & 0x3) << 4;
        b |= (codes[base + 3] & 0x3) << 6;
        dst_grp[blk_idx * BYTES_PER_BLOCK + 2 + byte_idx] = b;
    }
    // signs bytes: 4 blocks × 4 bytes = 16 packer lanes.
    if (lane < BLOCKS_PER_GROUP * 4) {
        const int blk_idx  = lane >> 2;          // 0..3
        const int byte_idx = lane & 3;           // 0..3
        const int base = blk_idx * BLOCK_SIZE + byte_idx * 8;
        unsigned char b = 0;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            const unsigned char hi = (codes[base + i] >> 2) & 0x1;
            b |= hi << i;
        }
        dst_grp[blk_idx * BYTES_PER_BLOCK + 10 + byte_idx] = b;
    }
    // pad bytes — leave whatever was there (the upstream writer should
    // have zeroed; we treat them as don't-care).
    if (lane < BLOCKS_PER_GROUP * 2) {
        const int blk_idx  = lane >> 1;
        const int byte_off = (lane & 1) + 14;
        dst_grp[blk_idx * BYTES_PER_BLOCK + byte_off] = 0;
    }
}
