// Q4_0 small-batch matvec (spec-decode verify, 1..8 activation rows):
// the shared matvec_batched_nr.h body over the repacked Q4_0 layout of
// matvec_q4_0_repacked (q4_0::repack_for_matvec): 16-byte nibble plane
// and fp16 d plane per block. The constant -8 offset is folded through
// the activation's quantized sum (xsum = d * sum(qs)), the same domain
// as the dot — see matvec_q4_0_repacked.cpp for why that matters.

#include "matvec_batched_nr.h"

struct WQ4_0 {
    const uint4* nib; const uint16_t* dp; unsigned int nsp;
    __device__ WQ4_0(const uint8_t* w, unsigned int out_dim, unsigned int nsp_, unsigned int)
        : nib(reinterpret_cast<const uint4*>(w)),
          dp(reinterpret_cast<const uint16_t*>(w + (size_t)out_dim * nsp_ * 16)), nsp(nsp_) {}
    struct Raw { uint4 q; uint16_t d; };
    struct Dec { uint32_t q[4]; float dw; };
    static constexpr bool USES_XSUM = true;
    static constexpr bool HALF_SUMS = false;
    __device__ Raw load(int row, unsigned int sb) const {
        const size_t i = (size_t)row * nsp + sb;
        return { nib[i], dp[i] };
    }
    __device__ static Dec decode(const Raw& r) {
        return { { r.q.x, r.q.y, r.q.z, r.q.w }, bnr_f16(r.d) };
    }
    __device__ static float dot(const Dec& d, const XAct& x) {
        int idot = 0;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            idot = __builtin_amdgcn_sdot4((int)( d.q[j]       & 0x0F0F0F0Fu), x.w[j],     idot, false);
            idot = __builtin_amdgcn_sdot4((int)((d.q[j] >> 4) & 0x0F0F0F0Fu), x.w[j + 4], idot, false);
        }
        return d.dw * (x.d * (float)idot - 8.0f * x.s);
    }
};

#include "matvec_batched_nr_entries.h"
BNR_ENTRIES(matvec_q4_0_repacked_batched, WQ4_0)
