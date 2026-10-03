// Q8_0 small-batch matvec (spec-decode verify, 1..8 activation rows):
// the shared matvec_batched_nr.h body over the two-plane repacked Q8_0
// layout of matvec_q8_0_repacked (q8_0::repack_for_matvec): quants 0-15
// of every block, then quants 16-31, then the fp16 scales.

#include "matvec_batched_nr.h"

struct WQ8_0 {
    const uint4* lo; const uint4* hi; const uint16_t* dp; unsigned int nsp;
    __device__ WQ8_0(const uint8_t* w, unsigned int out_dim, unsigned int nsp_, unsigned int)
        : lo(reinterpret_cast<const uint4*>(w)),
          hi(reinterpret_cast<const uint4*>(w + (size_t)out_dim * nsp_ * 16)),
          dp(reinterpret_cast<const uint16_t*>(w + (size_t)out_dim * nsp_ * 32)), nsp(nsp_) {}
    struct Raw { uint4 a, b; uint16_t d; };
    struct Dec { uint32_t q[8]; float dw; };
    static constexpr bool HALF_SUMS = false;
    __device__ Raw load(int row, unsigned int sb) const {
        const size_t i = (size_t)row * nsp + sb;
        return { lo[i], hi[i], dp[i] };
    }
    __device__ static Dec decode(const Raw& r) {
        return { { r.a.x, r.a.y, r.a.z, r.a.w, r.b.x, r.b.y, r.b.z, r.b.w }, bnr_f16(r.d) };
    }
    __device__ static float dot(const Dec& d, const XAct& x) {
        int idot = 0;
        #pragma unroll
        for (int j = 0; j < 8; j++) idot = __builtin_amdgcn_sdot4((int)d.q[j], x.w[j], idot, false);
        return d.dw * x.d * (float)idot;
    }
};

#include "matvec_batched_nr_entries.h"
BNR_ENTRIES(matvec_q8_0_repacked_batched, WQ8_0)
