// Q8_0 matvec via v_dot4_i32_i8, repacked two-plane weight layout.
//
// vs `matvec_q8_0_dp4a` (on-disk 34-byte blocks):
//   * `qs` plane: 32 bytes per sub-block, naturally aligned — the inner
//     loop reads one int32 per sdot4 instead of two uint16s OR-shifted
//     together to dodge the on-disk 2-byte alignment.
//   * `d` plane:  separate fp16 stream — no struct-stride interleaving
//     between the scale and the 32 quants.
//
// Layout (matches src/quant/q8_0.rs::repack_for_matvec):
//   slab + 0                      : out_dim × nsp × 32  qs bytes
//   slab + out_dim × nsp × 32     : out_dim × nsp × 2   fp16 d
//   nsp = (n_blocks is power of two) ? n_blocks + 1 : n_blocks  // anti-alias
//
// Two entry points, one per row tiling:
//   * `_f32`    — ROWS=2, grid = ceil(out_dim/2). Best for mid-size
//     out_dim (≈2048): one workgroup streams two adjacent weight rows.
//   * `_r1_f32` — ROWS=1, grid = out_dim. Doubles the wavefront count;
//     wins on large out_dim (≥4096) where ROWS=2 leaves too few
//     wavefront generations in flight to sustain HBM bandwidth
//     (measured: ROWS=2 out_dim=4096 stalls ~184 GB/s; ROWS=1 ~2× it).
// The caller picks per matvec — see qwen35 / gemma4 launch_matvec.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>
#include "gfx906_dpp.h"

struct __attribute__((packed)) BlockQ8 {
    float  d;
    float  xsum;
    int8_t qs[32];
};
static_assert(sizeof(BlockQ8) == 40, "BlockQ8 must be 40 bytes");

template<int ROWS>
__device__ __forceinline__
void mv_q8_0_repacked(const unsigned char* __restrict__ slab,
                      const BlockQ8*   __restrict__ xq,
                      float*           __restrict__ y,
                      unsigned int in_dim,
                      unsigned int out_dim)
{
    const unsigned int n_blocks = in_dim >> 5;
    const unsigned int nsp = ((n_blocks & (n_blocks - 1u)) == 0u)
                             ? (n_blocks + 1u) : n_blocks;

    const int*      qs_int = reinterpret_cast<const int*>(slab);
    const uint16_t* d_plane = reinterpret_cast<const uint16_t*>(
        slab + (size_t)out_dim * nsp * 32);

    const int row0 = blockIdx.x * ROWS;
    const int lane = threadIdx.x;

    float acc[ROWS];
    #pragma unroll
    for (int r = 0; r < ROWS; r++) acc[r] = 0.0f;

    for (int sb = lane; sb < (int)n_blocks; sb += 64) {
        const BlockQ8* xb = xq + sb;
        const float dx   = xb->d;
        const int*  xq32 = reinterpret_cast<const int*>(xb->qs);

        #pragma unroll
        for (int r = 0; r < ROWS; r++) {
            const int row = row0 + r;
            if (row >= (int)out_dim) continue;
            // qs[row][sb] — 32 bytes = 8 int32, naturally aligned
            const int*     w_int = qs_int + ((size_t)row * nsp + sb) * 8;
            const uint16_t db    = d_plane[(size_t)row * nsp + sb];
            const float    dw    = __half2float(*reinterpret_cast<const __half*>(&db));

            int idot = 0;
            #pragma unroll
            for (int g = 0; g < 8; g++) {
                idot = __builtin_amdgcn_sdot4(w_int[g], xq32[g], idot, false);
            }
            acc[r] += dw * dx * (float)idot;
        }
    }

    #pragma unroll
    for (int r = 0; r < ROWS; r++) {
        float a = acc[r];
        a = wave64_reduce_add_f32(a);
        if (lane == 0 && (row0 + r) < (int)out_dim) y[row0 + r] = a;
    }
}

extern "C" __global__
void matvec_q8_0_repacked_f32(const unsigned char* __restrict__ slab,
                              const BlockQ8*   __restrict__ xq,
                              float*           __restrict__ y,
                              unsigned int in_dim,
                              unsigned int out_dim)
{
    mv_q8_0_repacked<2>(slab, xq, y, in_dim, out_dim);
}

extern "C" __global__
void matvec_q8_0_repacked_r1_f32(const unsigned char* __restrict__ slab,
                                 const BlockQ8*   __restrict__ xq,
                                 float*           __restrict__ y,
                                 unsigned int in_dim,
                                 unsigned int out_dim)
{
    mv_q8_0_repacked<1>(slab, xq, y, in_dim, out_dim);
}
