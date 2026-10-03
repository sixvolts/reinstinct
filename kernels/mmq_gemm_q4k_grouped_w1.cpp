// Grouped-expert MMQ GEMM — repacked Q4_K, one wave per workgroup.
//
// Port of the llama fork's mmq_gemm_q4k_repacked_id_w1 (gfx906-perf,
// repack-gcn.cu; crossport ledger L8a) onto reinstinct's grouped-GEMM
// interface: same arguments, tile mapping (binary search over
// tile_off) and expert-sorted activation / output as
// mmq_gemm_q4k_grouped.cpp.
//
// A workgroup is one wave: 64 weight rows x 16 tokens of one expert.
// Lane l owns rows (l & 15) + 16i and tokens (l >> 4) + 4j — a 4x4
// register tile, so each unpacked weight sub-block feeds 4 tokens and
// each token read feeds 4 rows. Nibbles are split to int8 once at LDS
// staging (not per dp4a), and the next BK sub-blocks are prefetched
// into registers while the current ones compute.
//
// grid = (ceil(out_dim / 64), tile upper bound), block = 64.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#ifndef BK
#define BK 2
#endif
static_assert(BK == 1 || BK == 2 || BK == 4, "BK must be a power of two <= 4");

struct __attribute__((packed)) BlockQ8 {
    float  d;
    float  xsum;
    int8_t qs[32];
};
static_assert(sizeof(BlockQ8) == 40, "BlockQ8 must be 40 bytes");

extern "C" __global__ __launch_bounds__(64) __attribute__((amdgpu_waves_per_eu(2, 2)))
void mmq_gemm_q4k_grouped_w1_f32(const unsigned char* __restrict__ slab,
                                 unsigned int bytes_per_expert,
                                 const int*  __restrict__ expert_off,
                                 const int*  __restrict__ tile_off,
                                 unsigned int n_expert,
                                 const BlockQ8* __restrict__ xq,
                                 float*        __restrict__ y,
                                 unsigned int in_dim,
                                 unsigned int out_dim)
{
    // --- map workgroup → (expert, 16-token tile) ---
    const unsigned int by = blockIdx.y;
    if (by >= (unsigned int)tile_off[n_expert]) return;   // over-launched
    int lo = 0, hi = (int)n_expert;
    while (lo + 1 < hi) {
        int mid = (lo + hi) >> 1;
        if ((unsigned int)tile_off[mid] <= by) lo = mid; else hi = mid;
    }
    const unsigned int e      = (unsigned int)lo;
    const unsigned int a_base = (unsigned int)expert_off[e] + (by - (unsigned int)tile_off[e]) * 16;
    const unsigned int a_end  = (unsigned int)expert_off[e + 1];
    const unsigned char* wbase = slab + (size_t)e * bytes_per_expert;

    const int l  = threadIdx.x;
    const int rg = l & 15;
    const int tg = l >> 4;
    const unsigned int row0 = blockIdx.x * 64;

    const unsigned int n_sub   = in_dim >> 5;
    const unsigned int nsp     = ((n_sub & (n_sub - 1u)) == 0u) ? (n_sub + 1u) : n_sub;
    const unsigned int n_super = n_sub >> 3;
    const uint4*    nib = reinterpret_cast<const uint4*>(wbase);
    const uint16_t* smp = reinterpret_cast<const uint16_t*>(wbase + (size_t)out_dim * nsp * 16);
    const uint32_t* ddp = reinterpret_cast<const uint32_t*>(
        wbase + (size_t)out_dim * nsp * 16 + (size_t)out_dim * nsp * 2);

    __shared__ uint4  sW [64][2 * BK + 1];   // unpacked nibbles (lo, hi) per kk, padded rows
    __shared__ float2 sWs[64][BK];           // (d*sc, dmin*m)
    __shared__ uint4  sXq[16][2 * BK + 1];   // int8 activations, padded rows
    __shared__ float2 sXd[16][BK];           // (d, xsum)

    // Token staging: lane l < 16*BK owns slot (l / BK, l % BK); upper
    // lanes and out-of-range tokens alias a valid row (never written).
    const int xl = (l / BK) & 15;
    const int xk = l % BK;
    const unsigned int xa = min(a_base + xl, a_end - 1);
    const BlockQ8* xrow = xq + (size_t)xa * n_sub;

    float acc[4][4] = {};

    // Prefetch loads are unconditional (indices clamped) so they stay in
    // flight across the compute; out-of-range weight slots get zero
    // scales at staging.
    uint4 pw[BK]; uint16_t psm[BK]; uint32_t pdd[BK]; int px[10];
    auto gload = [&](unsigned int sb0) {
        #pragma unroll
        for (int i = 0; i < BK; i++) {
            const int it = l + 64 * i;
            const unsigned int wrow = min(row0 + it / BK, out_dim - 1);
            const unsigned int sb   = min(sb0 + it % BK, n_sub - 1);
            pw[i]  = nib[(size_t)wrow * nsp + sb];
            psm[i] = smp[(size_t)wrow * nsp + sb];
            pdd[i] = ddp[(size_t)wrow * n_super + (sb >> 3)];
        }
        const int* src = reinterpret_cast<const int*>(xrow + min(sb0 + xk, n_sub - 1));
        #pragma unroll
        for (int j = 0; j < 10; j++) px[j] = src[j];
    };
    auto lstore = [&](unsigned int sb0) {
        #pragma unroll
        for (int i = 0; i < BK; i++) {
            const int it = l + 64 * i;
            const int lr = it / BK, lk = it % BK;
            sW[lr][2 * lk]     = make_uint4( pw[i].x       & 0x0F0F0F0Fu,  pw[i].y       & 0x0F0F0F0Fu,
                                             pw[i].z       & 0x0F0F0F0Fu,  pw[i].w       & 0x0F0F0F0Fu);
            sW[lr][2 * lk + 1] = make_uint4((pw[i].x >> 4) & 0x0F0F0F0Fu, (pw[i].y >> 4) & 0x0F0F0F0Fu,
                                            (pw[i].z >> 4) & 0x0F0F0F0Fu, (pw[i].w >> 4) & 0x0F0F0F0Fu);
            const uint16_t d_bits = (uint16_t)(pdd[i] & 0xFFFF), m_bits = (uint16_t)(pdd[i] >> 16);
            const bool ok = row0 + lr < out_dim && sb0 + lk < n_sub;
            sWs[lr][lk] = ok ? make_float2(
                __half2float(*reinterpret_cast<const __half*>(&d_bits)) * (float)(psm[i] & 0xFFu),
                __half2float(*reinterpret_cast<const __half*>(&m_bits)) * (float)(psm[i] >> 8))
                : make_float2(0.0f, 0.0f);
        }
        if (l < 16 * BK) {
            sXq[xl][2 * xk]     = make_uint4(px[2], px[3], px[4], px[5]);
            sXq[xl][2 * xk + 1] = make_uint4(px[6], px[7], px[8], px[9]);
            sXd[xl][xk] = make_float2(__int_as_float(px[0]), __int_as_float(px[1]));
        }
    };

    gload(0);
    for (unsigned int sb0 = 0; sb0 < n_sub; sb0 += BK) {
        __syncthreads();
        lstore(sb0);
        __syncthreads();
        gload(sb0 + BK);   // clamped on the last step
        #pragma unroll 1
        for (int kk = 0; kk < BK; kk++) {
            int   xv[4][8];
            float dx[4], sx[4];
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                const uint4 a = sXq[tg + 4 * j][2 * kk], b = sXq[tg + 4 * j][2 * kk + 1];
                xv[j][0] = a.x; xv[j][1] = a.y; xv[j][2] = a.z; xv[j][3] = a.w;
                xv[j][4] = b.x; xv[j][5] = b.y; xv[j][6] = b.z; xv[j][7] = b.w;
                const float2 d = sXd[tg + 4 * j][kk];
                dx[j] = d.x; sx[j] = d.y;
            }
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                const int lr = rg + 16 * i;
                const uint4  qa = sW[lr][2 * kk], qb = sW[lr][2 * kk + 1];
                const float2 s  = sWs[lr][kk];
                const int w8[8] = { (int)qa.x, (int)qa.y, (int)qa.z, (int)qa.w,
                                    (int)qb.x, (int)qb.y, (int)qb.z, (int)qb.w };
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    int idot = 0;
                    #pragma unroll
                    for (int k = 0; k < 4; k++) {
                        idot = __builtin_amdgcn_sdot4(w8[k],     xv[j][k],     idot, false);
                        idot = __builtin_amdgcn_sdot4(w8[k + 4], xv[j][k + 4], idot, false);
                    }
                    acc[i][j] += s.x * dx[j] * (float)idot - s.y * sx[j];
                }
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const unsigned int row = row0 + rg + 16 * i;
        if (row >= out_dim) continue;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            const unsigned int a = a_base + tg + 4 * j;
            if (a < a_end) y[(size_t)a * out_dim + row] = acc[i][j];
        }
    }
}
