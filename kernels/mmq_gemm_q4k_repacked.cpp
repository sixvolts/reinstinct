// int8 MMQ GEMM — repacked Q4_K weights, dp4a, 2D-tiled with LDS
// staging. Y[P, out_dim] = Xq8[P, in_dim] · Wᵀ, consuming the quantised
// repacked weight directly (no dequant to fp16).
//
// A workgroup computes a BM×BN output tile (weight rows × tokens). The
// contraction is walked in chunks of BK sub-blocks; each chunk's weight
// tile and activation tile are cooperatively loaded into LDS (coalesced
// global reads), then every thread computes its TM×TN register
// micro-tile straight from LDS — each operand read from HBM once per
// output tile.
//
// 256-thread workgroup as a 16×16 thread grid; TM=TN=4 → BM=BN=64.
// The micro-tile is *strided* — thread (tx,ty) owns rows ty,ty+16,… and
// tokens tx,tx+16,… — so a wavefront's 16 distinct token reads land on
// 16 distinct LDS banks (a blocked mapping collides on the 40-byte
// BlockQ8 stride). The weight is held per chunk as raw uint4 and the
// nibbles unpacked inline in the dp4a; storing the unpacked lo/hi planes
// instead spilled 122 VGPRs to scratch.
//
// grid = (ceil(out_dim/64), ceil(P/64)).

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#define BM 64        // weight rows per workgroup tile
#define BN 64        // tokens per workgroup tile
#define BK 4         // sub-blocks per contraction chunk
#define TM 4         // rows  per thread micro-tile
#define TN 4         // tokens per thread micro-tile

struct __attribute__((packed)) BlockQ8 {
    float  d;
    float  xsum;
    int8_t qs[32];
};
static_assert(sizeof(BlockQ8) == 40, "BlockQ8 must be 40 bytes");

extern "C" __global__ __launch_bounds__(256, 2)
void mmq_gemm_q4k_repacked_f32(const unsigned char* __restrict__ wbase,
                               const BlockQ8*       __restrict__ xq,
                               float*               __restrict__ y,
                               unsigned int in_dim,
                               unsigned int out_dim,
                               unsigned int p_rows)
{
    const int t  = threadIdx.x;          // 0..255
    const int tx = t & 15;               // token group  0..15
    const int ty = t >> 4;               // row group    0..15
    const unsigned int row0 = blockIdx.x * BM;
    const unsigned int tok0 = blockIdx.y * BN;

    const unsigned int n_sub = in_dim >> 5;
    const unsigned int nsp = ((n_sub & (n_sub - 1u)) == 0u) ? (n_sub + 1u) : n_sub;
    const uint4*    nib = reinterpret_cast<const uint4*>(wbase);
    const uint32_t* scl = reinterpret_cast<const uint32_t*>(
        wbase + (size_t)out_dim * nsp * 16);

    // Tile staging — BK sub-blocks of BM weight rows / BN token rows.
    __shared__ uint4    sW[BM][BK];      // packed nibbles  — 4096 B
    __shared__ uint32_t sWs[BM][BK];     // dsc|deff fp16   — 1024 B
    __shared__ BlockQ8  sX[BN][BK];      // int8 acts       — 10240 B

    float acc[TM][TN];
    #pragma unroll
    for (int r = 0; r < TM; r++)
        #pragma unroll
        for (int n = 0; n < TN; n++) acc[r][n] = 0.0f;

    // Cooperative-load index: 256 threads ↔ BM·BK (= BN·BK = 256) elems.
    const int lr = t >> 2;               // row / token  0..63
    const int lk = t & 3;                // sub-block off 0..3

    for (unsigned int sb0 = 0; sb0 < n_sub; sb0 += BK) {
        const unsigned int sb = sb0 + lk;
        // Weight: OOB rows get a zero scale → zero contribution.
        const unsigned int wrow = row0 + lr;
        if (wrow < out_dim) {
            sW[lr][lk]  = nib[(size_t)wrow * nsp + sb];
            sWs[lr][lk] = scl[(size_t)wrow * nsp + sb];
        } else {
            sWs[lr][lk] = 0;
        }
        // Activation: OOB tokens get d = xsum = 0 → zero contribution.
        const unsigned int xtok = tok0 + lr;
        if (xtok < p_rows) {
            sX[lr][lk] = xq[(size_t)xtok * n_sub + sb];
        } else {
            sX[lr][lk].d = 0.0f;
            sX[lr][lk].xsum = 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < BK; kk++) {
            // Hold the TM weight rows as raw uint4 (16 VGPR) — not the
            // unpacked lo/hi planes (32 VGPR), which spilled. Nibbles are
            // unpacked inline in the dp4a; the AND ops are far cheaper
            // than the scratch traffic a spill costs.
            uint4 wq[TM];
            float dsc[TM], deff[TM];
            #pragma unroll
            for (int r = 0; r < TM; r++) {
                wq[r] = sW[ty + r * 16][kk];
                const uint32_t s = sWs[ty + r * 16][kk];
                const uint16_t dsc_bits  = (uint16_t)(s & 0xFFFF);
                const uint16_t deff_bits = (uint16_t)(s >> 16);
                dsc[r]  = __half2float(*reinterpret_cast<const __half*>(&dsc_bits));
                deff[r] = __half2float(*reinterpret_cast<const __half*>(&deff_bits));
            }
            #pragma unroll
            for (int n = 0; n < TN; n++) {
                const BlockQ8* xb   = &sX[tx + n * 16][kk];
                const int*     xq32 = reinterpret_cast<const int*>(xb->qs);
                const float    dx   = xb->d;
                const float    xsum = xb->xsum;
                #pragma unroll
                for (int r = 0; r < TM; r++) {
                    const uint32_t qa[4] = { wq[r].x, wq[r].y, wq[r].z, wq[r].w };
                    int idot = 0;
                    #pragma unroll
                    for (int j = 0; j < 4; j++) {
                        idot = __builtin_amdgcn_sdot4(
                            (int)( qa[j]       & 0x0F0F0F0Fu), xq32[j],     idot, false);
                        idot = __builtin_amdgcn_sdot4(
                            (int)((qa[j] >> 4) & 0x0F0F0F0Fu), xq32[j + 4], idot, false);
                    }
                    acc[r][n] += dsc[r] * dx * (float)idot - deff[r] * xsum;
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int r = 0; r < TM; r++) {
        const unsigned int row = row0 + ty + r * 16;
        if (row >= out_dim) continue;
        #pragma unroll
        for (int n = 0; n < TN; n++) {
            const unsigned int tok = tok0 + tx + n * 16;
            if (tok < p_rows) y[(size_t)tok * out_dim + row] = acc[r][n];
        }
    }
}
