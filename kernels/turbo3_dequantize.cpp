// Turbo3 decoder: packed turbo3 blocks → f32 [N, head_dim].
//
// Mirror of turbo3_encode_f32. Per 128-element rotation group:
//   1. Unpack 3-bit codes from 4 blocks (8 qs + 4 signs bytes each)
//   2. Look up centroid × norm → fp32 in RHT space
//   3. Inverse RHT: signs2 (post-RHT) → FWHT (self-inverse) → signs1
//      → multiply by 1/√128 once
//
// Grid: (n_rows, groups_per_row). Block: 128 threads.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

constexpr int ROT_GROUP        = 128;
constexpr int BLOCK_SIZE       = 32;
constexpr int BYTES_PER_BLOCK  = 16;
constexpr int BLOCKS_PER_GROUP = ROT_GROUP / BLOCK_SIZE;

__device__ __constant__ float CENTROIDS_T3_DQ[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};

extern "C" __global__ __launch_bounds__(128, 4)
void turbo3_decode_f32(const unsigned char* __restrict__ src,    // packed
                       const int8_t*        __restrict__ signs1,
                       const int8_t*        __restrict__ signs2,
                       float*               __restrict__ dst,    // [n_rows, head_dim]
                       unsigned int n_rows,
                       unsigned int head_dim)
{
    const unsigned int groups_per_row = head_dim / ROT_GROUP;
    const unsigned int row = blockIdx.x;
    const unsigned int grp = blockIdx.y;
    if (row >= n_rows || grp >= groups_per_row) return;

    const int lane = threadIdx.x;
    const size_t src_off = ((size_t)row * groups_per_row + grp)
                         * BLOCKS_PER_GROUP * BYTES_PER_BLOCK;
    const unsigned char* src_grp = src + src_off;

    // Each lane handles one of the 128 values: figure out which block + slot.
    const int blk_idx  = lane >> 5;       // 0..3
    const int k        = lane & 31;       // 0..31 within block
    const unsigned char* blk = src_grp + blk_idx * BYTES_PER_BLOCK;

    const uint16_t nb  = (uint16_t)blk[0] | ((uint16_t)blk[1] << 8);
    const float    norm = __half2float(*reinterpret_cast<const __half*>(&nb));
    const unsigned char qs_byte    = blk[2 + (k >> 2)];
    const unsigned char signs_byte = blk[10 + (k >> 3)];
    const unsigned char lo = (qs_byte    >> ((k & 3) * 2)) & 0x3;
    const unsigned char hi = (signs_byte >> (k & 7))       & 0x1;
    const unsigned char code = (hi << 2) | lo;
    const float rotated = CENTROIDS_T3_DQ[code] * norm;

    __shared__ float x[ROT_GROUP];
    // Inverse RHT step 1: signs2 first (mirror order).
    x[lane] = rotated * (float)signs2[lane];
    __syncthreads();

    // FWHT-128 (self-inverse modulo the 1/N scale we apply at the end).
    for (int h = 1; h < ROT_GROUP; h *= 2) {
        if ((lane & h) == 0) {
            const float a = x[lane];
            const float b = x[lane + h];
            x[lane]     = a + b;
            x[lane + h] = a - b;
        }
        __syncthreads();
    }

    // Inverse scale: orthonormal RHT used 1/√128 on encode; inverse is
    // another 1/√128 (so the two RHTs compose to identity = full 1/128).
    const float scale = 1.0f / sqrtf((float)ROT_GROUP);
    // Inverse step 2: signs1 last.
    const float v_out = x[lane] * scale * (float)signs1[lane];

    dst[(size_t)row * head_dim + grp * ROT_GROUP + lane] = v_out;
}
