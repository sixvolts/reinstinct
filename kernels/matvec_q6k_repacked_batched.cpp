// Q6_K small-batch matvec (spec-decode verify, 1..8 activation rows):
// the shared matvec_batched_nr.h body over the repacked Q6_K layout of
// matvec_q6k_repacked (q6_k::repack_for_matvec): 16-byte low-nibble
// plane, 8-byte high-2-bit plane, 2-byte (two int8 scales) plane per
// sub-block, fp16 d per superblock. Real value = q - 32, folded as
// -32 x (int sum of each 16-element activation half).

#include "matvec_batched_nr.h"

// Spread 4 x 2 bits to bits 4-5 of each byte.
__device__ __forceinline__ uint32_t spread2(uint32_t h) {
    return ((h & 0x03u) << 4) | ((h & 0x0Cu) << 10)
         | ((h & 0x30u) << 16) | ((h & 0xC0u) << 22);
}

struct WQ6K {
    const uint4* nib; const uint2* h2p; const uint16_t* smp; const uint16_t* ddp;
    unsigned int nsp, n_super;
    __device__ WQ6K(const uint8_t* w, unsigned int out_dim, unsigned int nsp_, unsigned int n_super_)
        : nib(reinterpret_cast<const uint4*>(w)),
          h2p(reinterpret_cast<const uint2*>(w + (size_t)out_dim * nsp_ * 16)),
          smp(reinterpret_cast<const uint16_t*>(w + (size_t)out_dim * nsp_ * 24)),
          ddp(reinterpret_cast<const uint16_t*>(w + (size_t)out_dim * nsp_ * 26)),
          nsp(nsp_), n_super(n_super_) {}
    struct Raw { uint4 q; uint2 h2; uint16_t sm; uint16_t d; };
    struct Dec { uint32_t lo[4], hi[4]; float dlo, dhi; };
    static constexpr bool HALF_SUMS = true;
    __device__ Raw load(int row, unsigned int sb) const {
        const size_t i = (size_t)row * nsp + sb;
        return { nib[i], h2p[i], smp[i], ddp[(size_t)row * n_super + (sb >> 3)] };
    }
    __device__ static Dec decode(const Raw& r) {
        Dec d;
        const uint32_t qa[4] = { r.q.x, r.q.y, r.q.z, r.q.w };
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            const uint32_t h = j < 2 ? r.h2.x : r.h2.y;
            d.lo[j] = ( qa[j]       & 0x0F0F0F0Fu) | spread2((h >> (16 * (j & 1)))     & 0xFFu);
            d.hi[j] = ((qa[j] >> 4) & 0x0F0F0F0Fu) | spread2((h >> (16 * (j & 1) + 8)) & 0xFFu);
        }
        const float dd = bnr_f16(r.d);
        d.dlo = dd * (float)(int)(int8_t)(r.sm & 0xFFu);
        d.dhi = dd * (float)(int)(int8_t)(r.sm >> 8);
        return d;
    }
    __device__ static float dot(const Dec& d, const XAct& x) {
        int i0 = 0, i1 = 0;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            i0 = __builtin_amdgcn_sdot4((int)d.lo[j], x.w[j],     i0, false);
            i1 = __builtin_amdgcn_sdot4((int)d.hi[j], x.w[j + 4], i1, false);
        }
        return x.d * (d.dlo * (float)(i0 - 32 * x.h0) + d.dhi * (float)(i1 - 32 * x.h1));
    }
};

// As Q5_K: 8 decoded words per sub-block, so n3 / n4 take 2 rows per
// wave to stay above 1 wave/SIMD.
#define BNR_R3 2
#define BNR_R4 2
#include "matvec_batched_nr_entries.h"
BNR_ENTRIES(matvec_q6k_repacked_batched, WQ6K)
