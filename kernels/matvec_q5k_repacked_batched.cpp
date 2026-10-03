// Q5_K small-batch matvec (spec-decode verify, 1..8 activation rows):
// the shared matvec_batched_nr.h body over the repacked Q5_K layout of
// matvec_q5k_repacked (q5_k::repack_for_matvec): 16-byte nibble plane,
// 4-byte qh (5th bit) plane, 2-byte scale plane per sub-block, 4-byte
// (fp16 d, dmin) superblock plane.

#include "matvec_batched_nr.h"

// Spread 4 bits to bit 4 of each byte.
__device__ __forceinline__ uint32_t spread4(uint32_t h) {
    return ((h & 1u) << 4) | ((h & 2u) << 11) | ((h & 4u) << 18) | ((h & 8u) << 25);
}

struct WQ5K {
    const uint4* nib; const uint32_t* qhp; const uint16_t* smp; const uint32_t* ddp;
    unsigned int nsp, n_super;
    __device__ WQ5K(const uint8_t* w, unsigned int out_dim, unsigned int nsp_, unsigned int n_super_)
        : nib(reinterpret_cast<const uint4*>(w)),
          qhp(reinterpret_cast<const uint32_t*>(w + (size_t)out_dim * nsp_ * 16)),
          smp(reinterpret_cast<const uint16_t*>(w + (size_t)out_dim * nsp_ * 20)),
          ddp(reinterpret_cast<const uint32_t*>(w + (size_t)out_dim * nsp_ * 22)),
          nsp(nsp_), n_super(n_super_) {}
    struct Raw { uint4 q; uint32_t qh; uint16_t sm; uint32_t dd; };
    struct Dec { uint32_t lo[4], hi[4]; float dsc, deff; };
    static constexpr bool HALF_SUMS = false;
    __device__ Raw load(int row, unsigned int sb) const {
        const size_t i = (size_t)row * nsp + sb;
        return { nib[i], qhp[i], smp[i], ddp[(size_t)row * n_super + (sb >> 3)] };
    }
    __device__ static Dec decode(const Raw& r) {
        Dec d;
        const uint32_t qa[4] = { r.q.x, r.q.y, r.q.z, r.q.w };
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            d.lo[j] = ( qa[j]       & 0x0F0F0F0Fu) | spread4((r.qh >> (8 * j))     & 0xFu);
            d.hi[j] = ((qa[j] >> 4) & 0x0F0F0F0Fu) | spread4((r.qh >> (8 * j + 4)) & 0xFu);
        }
        d.dsc  = bnr_f16((uint16_t)(r.dd & 0xFFFF)) * (float)(r.sm & 0xFFu);
        d.deff = bnr_f16((uint16_t)(r.dd >> 16)) * (float)(r.sm >> 8);
        return d;
    }
    __device__ static float dot(const Dec& d, const XAct& x) {
        int idot = 0;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            idot = __builtin_amdgcn_sdot4((int)d.lo[j], x.w[j],     idot, false);
            idot = __builtin_amdgcn_sdot4((int)d.hi[j], x.w[j + 4], idot, false);
        }
        return d.dsc * x.d * (float)idot - d.deff * x.s;
    }
};

// The decoded Q5_K sub-block is 8 words: 4 rows of it at 4 activation
// rows needs ~140 VGPRs (1 wave/SIMD), so n4 takes 2 rows per wave.
#define BNR_R4 2
#include "matvec_batched_nr_entries.h"
BNR_ENTRIES(matvec_q5k_repacked_batched, WQ5K)
