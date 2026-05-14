// Fused Q4_K dequantize + matvec.
//
// Q4_K layout (144 bytes / 256 weights):
//   fp16  d                super-block scale
//   fp16  dmin             super-block min
//   uint8 scales[12]       8 sub-blocks × (6-bit sc, 6-bit m), bit-packed
//   uint8 qs[128]          256 4-bit nibbles. qs[chunk*32 + l] holds two
//                          nibbles: low → sub-block 2*chunk, high → 2*chunk+1
//
// Per weight: w = d * sc * q  -  dmin * m
//
// Strategy: one block per output row; threads stride over sub-blocks
// (32 weights each). Each iteration unpacks (sc, m) for its sub-block,
// dequants 32 nibbles, dots with the matching 32 x[] values, accumulates.
// Tree reduction over thread partials produces y[row].

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

struct __attribute__((packed)) BlockQ4_K {
    uint16_t d;            // fp16 raw bits
    uint16_t dmin;         // fp16 raw bits
    uint8_t  scales[12];   // packed 6-bit sub-block (sc, m) pairs
    uint8_t  qs[128];      // 256 nibbles
};
static_assert(sizeof(BlockQ4_K) == 144, "BlockQ4_K must be 144 bytes");

// ggml's get_scale_min_k4: return (sc, m) for sub-block j in [0,8).
//   j < 4:   sc = q[j]   & 63              m = q[j+4] & 63
//   j >= 4:  sc = (q[j+4] & 0x0F) | ((q[j-4] >> 6) << 4)
//            m  = (q[j+4] >>  4)  | ((q[j]   >> 6) << 4)
__device__ __forceinline__
void get_scale_min_k4(int j, const uint8_t* q, uint8_t& sc, uint8_t& m) {
    if (j < 4) {
        sc = q[j]     & 63;
        m  = q[j + 4] & 63;
    } else {
        sc = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
        m  = (q[j + 4] >> 4)   | ((q[j]     >> 6) << 4);
    }
}

extern "C" __global__
void matvec_q4_k_f32(const BlockQ4_K* __restrict__ w_blocks,  // [out_dim, in_dim/256]
                     const float*     __restrict__ x,          // [in_dim]
                     float*           __restrict__ y,          // [out_dim]
                     unsigned int in_dim,
                     unsigned int out_dim)
{
    extern __shared__ float smem[];
    const int row = blockIdx.x;
    if (row >= (int)out_dim) return;
    const int tid = threadIdx.x;
    const int bs  = blockDim.x;

    const unsigned int n_blocks   = in_dim >> 8;   // /256
    const unsigned int n_subblocks = in_dim >> 5;  // /32
    const BlockQ4_K* row_blocks = w_blocks + (size_t)row * (size_t)n_blocks;

    float acc = 0.0f;
    for (int sb = tid; sb < (int)n_subblocks; sb += bs) {
        const int blk_idx       = sb >> 3;        // /8 sub-blocks per Q4_K block
        const int sub_in_block  = sb & 7;         // 0..7
        const int chunk         = sub_in_block >> 1; // 0..3, two sub-blocks per chunk
        const int is_high       = sub_in_block & 1;

        const BlockQ4_K* blk = row_blocks + blk_idx;
        const __half d_h    = *reinterpret_cast<const __half*>(&blk->d);
        const __half dm_h   = *reinterpret_cast<const __half*>(&blk->dmin);
        const float  d      = __half2float(d_h);
        const float  dmin   = __half2float(dm_h);

        uint8_t sc, m;
        get_scale_min_k4(sub_in_block, blk->scales, sc, m);
        const float sub_d = d    * (float)sc;
        const float sub_m = dmin * (float)m;

        const uint8_t* qs_base = blk->qs + chunk * 32;
        const float*   x_base  = x + (size_t)sb * 32;

        float partial = 0.0f;
        if (is_high) {
            #pragma unroll
            for (int l = 0; l < 32; l++) {
                const float w = sub_d * (float)(qs_base[l] >> 4) - sub_m;
                partial += w * x_base[l];
            }
        } else {
            #pragma unroll
            for (int l = 0; l < 32; l++) {
                const float w = sub_d * (float)(qs_base[l] & 0x0F) - sub_m;
                partial += w * x_base[l];
            }
        }
        acc += partial;
    }

    smem[tid] = acc;
    __syncthreads();
    for (int s = bs / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) y[row] = smem[0];
}
