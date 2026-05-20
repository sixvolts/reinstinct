// Q6_K matvec for K=2..8 activation rows. Same batching idea as
// matvec_q{4,5}k_repacked_batched but with Q6_K's two-scale-per-sub-block
// layout and the symmetric `-32` fold (real value = q-32). The
// `xis0/xis1` activation half-sums become per-batch-row arrays.
//
// 256-thread workgroup, 4 waves × ROWS=2 per WG. grid = ceil(out_dim/8).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#define ROWS         2
#define N_ROWS_MAX   4   // K upper bound; see matvec_q4k_repacked_batched.cpp

struct __attribute__((packed)) BlockQ8 {
    float  d;
    float  xsum;
    int8_t qs[32];
};
static_assert(sizeof(BlockQ8) == 40, "BlockQ8 must be 40 bytes");

__device__ __forceinline__ uint32_t spread2(uint32_t h) {
    return ((h & 0x03u) << 4) | ((h & 0x0Cu) << 10)
         | ((h & 0x30u) << 16) | ((h & 0xC0u) << 22);
}

extern "C" __global__
void matvec_q6k_repacked_batched_f32(const uint8_t* __restrict__ wbase,
                                     const BlockQ8* __restrict__ xq,
                                     float*         __restrict__ y,
                                     unsigned int in_dim,
                                     unsigned int out_dim,
                                     unsigned int n_rows)
{
    const int wave = threadIdx.x >> 6;
    const int lane = threadIdx.x & 63;
    const int row0 = blockIdx.x * (ROWS * 4) + wave * ROWS;
    const unsigned int n_sub = in_dim >> 5;
    const unsigned int nsp = ((n_sub & (n_sub - 1u)) == 0u) ? (n_sub + 1u) : n_sub;
    const unsigned int n_super = n_sub >> 3;

    const uint4*    nib = reinterpret_cast<const uint4*>(wbase);
    const uint32_t* h2p = reinterpret_cast<const uint32_t*>(
        wbase + (size_t)out_dim * nsp * 16);
    const uint16_t* smp = reinterpret_cast<const uint16_t*>(
        wbase + (size_t)out_dim * nsp * 16 + (size_t)out_dim * nsp * 8);
    const uint16_t* ddp = reinterpret_cast<const uint16_t*>(
        wbase + (size_t)out_dim * nsp * 16 + (size_t)out_dim * nsp * 8
              + (size_t)out_dim * nsp * 2);

    float acc[ROWS][N_ROWS_MAX];
    #pragma unroll
    for (int r = 0; r < ROWS; r++)
        #pragma unroll
        for (int b = 0; b < N_ROWS_MAX; b++) acc[r][b] = 0.0f;

    for (unsigned int sb = lane; sb < n_sub; sb += 64) {
        // Per-batch-row activation halves + half-sums.
        int xis0[N_ROWS_MAX], xis1[N_ROWS_MAX];
        float dx_arr[N_ROWS_MAX];
        const int* xq32_arr[N_ROWS_MAX];
        for (unsigned int b = 0; b < n_rows; b++) {
            const BlockQ8* xb = xq + (size_t)b * n_sub + sb;
            dx_arr[b]    = xb->d;
            xq32_arr[b]  = reinterpret_cast<const int*>(xb->qs);
            int s0 = 0, s1 = 0;
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                s0 = __builtin_amdgcn_sdot4(xq32_arr[b][j],     0x01010101, s0, false);
                s1 = __builtin_amdgcn_sdot4(xq32_arr[b][j + 4], 0x01010101, s1, false);
            }
            xis0[b] = s0;
            xis1[b] = s1;
        }

        #pragma unroll
        for (int r = 0; r < ROWS; r++) {
            const int row = row0 + r;
            if (row >= (int)out_dim) continue;

            const size_t idx = (size_t)row * nsp + sb;
            const uint4    q    = nib[idx];
            const uint32_t h2lo = h2p[idx * 2];
            const uint32_t h2hi = h2p[idx * 2 + 1];
            const uint16_t sm     = smp[idx];
            const uint16_t d_bits = ddp[(size_t)row * n_super + (sb >> 3)];
            const float d = __half2float(*reinterpret_cast<const __half*>(&d_bits));
            const float dsc_lo = d * (float)(int)(int8_t)(sm & 0xFFu);
            const float dsc_hi = d * (float)(int)(int8_t)(sm >> 8);

            // Form the 8 q6 chunks (4 lo + 4 hi) once per (sb, row).
            const uint32_t qa[4] = { q.x, q.y, q.z, q.w };
            uint32_t q6lo[4], q6hi[4];
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                const uint32_t ge = 2 * j;
                const uint32_t go = 2 * j + 1;
                const uint32_t he = ((ge < 4 ? h2lo : h2hi) >> (8 * (ge & 3))) & 0xFFu;
                const uint32_t ho = ((go < 4 ? h2lo : h2hi) >> (8 * (go & 3))) & 0xFFu;
                q6lo[j] = ( qa[j]       & 0x0F0F0F0Fu) | spread2(he);
                q6hi[j] = ((qa[j] >> 4) & 0x0F0F0F0Fu) | spread2(ho);
            }

            for (unsigned int b = 0; b < n_rows; b++) {
                const int* xq32 = xq32_arr[b];
                const float dx  = dx_arr[b];
                int idot0 = 0, idot1 = 0;
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    idot0 = __builtin_amdgcn_sdot4((int)q6lo[j], xq32[j],     idot0, false);
                    idot1 = __builtin_amdgcn_sdot4((int)q6hi[j], xq32[j + 4], idot1, false);
                }
                acc[r][b] += dsc_lo * dx * (float)(idot0 - 32 * xis0[b])
                           + dsc_hi * dx * (float)(idot1 - 32 * xis1[b]);
            }
        }
    }

    #pragma unroll
    for (int r = 0; r < ROWS; r++) {
        for (unsigned int b = 0; b < n_rows; b++) {
            float a = acc[r][b];
            a += __shfl_xor(a, 32);
            a += __shfl_xor(a, 16);
            a += __shfl_xor(a,  8);
            a += __shfl_xor(a,  4);
            a += __shfl_xor(a,  2);
            a += __shfl_xor(a,  1);
            if (lane == 0 && (row0 + r) < (int)out_dim) {
                y[(size_t)b * out_dim + (row0 + r)] = a;
            }
        }
    }
}
