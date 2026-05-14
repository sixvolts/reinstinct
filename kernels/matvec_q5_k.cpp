// Fused Q5_K dequantize + matvec.
//
// Q5_K layout (176 bytes / 256 weights):
//   fp16  d            super-block scale         (offset 0)
//   fp16  dmin         super-block min           (offset 2)
//   uint8 scales[12]   packed 6-bit (sc, m) — same as Q4_K (offset 4)
//   uint8 qh[32]       one high-bit per weight, transposed         (offset 16)
//   uint8 qs[128]      low 4 bits per weight, same layout as Q4_K  (offset 48)
//
// Per weight: q5 = (qs_nibble) | ((qh_bit) << 4) ∈ [0, 31]
//             w  = d * sub_d * q5  -  dmin * sub_m
//
// qh transpose: bit (chunk*2 + is) of qh[l] is the high bit for the
// weight at output position chunk*64 + is*32 + l.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

struct __attribute__((packed)) BlockQ5_K {
    uint16_t d;            // fp16 raw bits
    uint16_t dmin;         // fp16 raw bits
    uint8_t  scales[12];
    uint8_t  qh[32];
    uint8_t  qs[128];
};
static_assert(sizeof(BlockQ5_K) == 176, "BlockQ5_K must be 176 bytes");

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
void matvec_q5_k_f32(const BlockQ5_K* __restrict__ w_blocks,  // [out_dim, in_dim/256]
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

    const unsigned int n_blocks    = in_dim >> 8;   // /256
    const unsigned int n_subblocks = in_dim >> 5;   // /32
    const BlockQ5_K* row_blocks = w_blocks + (size_t)row * (size_t)n_blocks;

    float acc = 0.0f;
    for (int sb = tid; sb < (int)n_subblocks; sb += bs) {
        const int blk_idx      = sb >> 3;        // /8 sub-blocks per block
        const int sub_in_block = sb & 7;         // 0..7
        const int chunk        = sub_in_block >> 1;  // 0..3
        const int is           = sub_in_block & 1;   // 0 or 1
        const int qh_mask      = 1 << (chunk * 2 + is);
        const int qs_off_blk   = chunk * 32;

        const BlockQ5_K* blk = row_blocks + blk_idx;
        const __half d_h    = *reinterpret_cast<const __half*>(&blk->d);
        const __half dm_h   = *reinterpret_cast<const __half*>(&blk->dmin);
        const float  d      = __half2float(d_h);
        const float  dmin   = __half2float(dm_h);

        uint8_t sc, m;
        get_scale_min_k4(sub_in_block, blk->scales, sc, m);
        const float sub_d = d    * (float)sc;
        const float sub_m = dmin * (float)m;

        const uint8_t* qs_base = blk->qs + qs_off_blk;
        const uint8_t* qh_base = blk->qh;
        const float*   x_base  = x + (size_t)sb * 32;

        float partial = 0.0f;
        if (is == 0) {
            #pragma unroll
            for (int l = 0; l < 32; l++) {
                const int q_lo = (int)(qs_base[l] & 0x0F)
                               + ((qh_base[l] & qh_mask) ? 16 : 0);
                const float w = sub_d * (float)q_lo - sub_m;
                partial += w * x_base[l];
            }
        } else {
            #pragma unroll
            for (int l = 0; l < 32; l++) {
                const int q_hi = (int)(qs_base[l] >> 4)
                               + ((qh_base[l] & qh_mask) ? 16 : 0);
                const float w = sub_d * (float)q_hi - sub_m;
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
