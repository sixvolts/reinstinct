// Embedding lookup from a Q8_0 / Q4_0 table in the repacked matvec
// layout (quant::q8_0 / q4_0 ::repack_for_matvec), so a tied embedding
// can live once on the device in the layout the LM-head matvec reads at
// full bandwidth. Both layouts keep each row's sub-blocks contiguous
// and row-major (sub-block i = row * nsp + blk, nsp = padded sub-blocks
// per row); the planes are:
//
//   Q8_0: lo quants [vocab*nsp][16] | hi quants [vocab*nsp][16] | fp16 d [vocab*nsp]
//         weight k of a block = d * (k < 16 ? lo[k] : hi[k - 16])
//   Q4_0: nibbles  [vocab*nsp][16]  | fp16 d [vocab*nsp]
//         byte k holds weight k (low nibble) and k + 16 (high nibble);
//         weight = d * (q - 8)
//
// One thread per output element. grid = (ceil(hidden / 256), n_tokens);
// block = 256. The single-token entries read their id from row_idx[0].

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

__device__ __forceinline__ float rp_q8_0(const uint8_t* t, size_t plane, unsigned int sb, unsigned int k) {
    const int8_t q = (int8_t)(k < 16 ? t[(size_t)sb * 16 + k] : t[plane + (size_t)sb * 16 + k - 16]);
    const float d = __half2float(*reinterpret_cast<const __half*>(t + 2 * plane + (size_t)sb * 2));
    return d * (float)q;
}

__device__ __forceinline__ float rp_q4_0(const uint8_t* t, size_t plane, unsigned int sb, unsigned int k) {
    const uint8_t b = t[(size_t)sb * 16 + (k & 15)];
    const int q = (k < 16 ? (b & 0x0F) : (b >> 4)) - 8;
    const float d = __half2float(*reinterpret_cast<const __half*>(t + plane + (size_t)sb * 2));
    return d * (float)q;
}

#define EMBED_RP(name, fn)                                                            \
extern "C" __global__                                                                 \
void name##_f32(const uint8_t* __restrict__ t, float* __restrict__ out,               \
                const unsigned int* __restrict__ row_idx, unsigned int hidden,        \
                unsigned int nsp, unsigned int vocab)                                 \
{                                                                                     \
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;                     \
    if (i >= hidden) return;                                                          \
    const size_t plane = (size_t)vocab * nsp * 16;                                    \
    const unsigned int sb = row_idx[0] * nsp + (i >> 5);                             \
    out[i] = fn(t, plane, sb, i & 31);                                                \
}                                                                                     \
extern "C" __global__                                                                 \
void name##_batched_f32(const uint8_t* __restrict__ t, float* __restrict__ out,       \
                        const unsigned int* __restrict__ row_idx, unsigned int hidden,\
                        unsigned int nsp, unsigned int vocab)                         \
{                                                                                     \
    const unsigned int r = blockIdx.y;                                                \
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;                     \
    if (i >= hidden) return;                                                          \
    const size_t plane = (size_t)vocab * nsp * 16;                                    \
    const unsigned int sb = row_idx[r] * nsp + (i >> 5);                             \
    out[(size_t)r * hidden + i] = fn(t, plane, sb, i & 31);                           \
}

EMBED_RP(embed_lookup_q8_0_repacked, rp_q8_0)
EMBED_RP(embed_lookup_q4_0_repacked, rp_q4_0)
