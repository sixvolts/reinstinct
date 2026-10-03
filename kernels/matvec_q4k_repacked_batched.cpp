// Q4_K small-batch matvec (spec-decode verify, 1..8 activation rows):
// the shared matvec_batched_nr.h body over the repacked Q4_K layout of
// matvec_q4k_repacked (q4_k::repack_for_matvec):
//   * nibble plane — 16 bytes per sub-block, sub-block-major.
//   * scale plane — 2 bytes per sub-block (6-bit sc | m).
//   * superblock plane — 4 bytes per 256-weight superblock (fp16 d, dmin).

#include "matvec_batched_nr.h"

struct WQ4K {
    const uint4* nib; const uint16_t* smp; const uint32_t* ddp;
    unsigned int nsp, n_super;
    __device__ WQ4K(const uint8_t* w, unsigned int out_dim, unsigned int nsp_, unsigned int n_super_)
        : nib(reinterpret_cast<const uint4*>(w)),
          smp(reinterpret_cast<const uint16_t*>(w + (size_t)out_dim * nsp_ * 16)),
          ddp(reinterpret_cast<const uint32_t*>(w + (size_t)out_dim * nsp_ * 18)),
          nsp(nsp_), n_super(n_super_) {}
    struct Raw { uint4 q; uint16_t sm; uint32_t dd; };
    struct Dec { uint32_t q[4]; float dsc, deff; };
    static constexpr bool USES_XSUM = true;
    static constexpr bool HALF_SUMS = false;
    __device__ Raw load(int row, unsigned int sb) const {
        return { nib[(size_t)row * nsp + sb], smp[(size_t)row * nsp + sb],
                 ddp[(size_t)row * n_super + (sb >> 3)] };
    }
    __device__ static Dec decode(const Raw& r) {
        return { { r.q.x, r.q.y, r.q.z, r.q.w },
                 bnr_f16((uint16_t)(r.dd & 0xFFFF)) * (float)(r.sm & 0xFFu),
                 bnr_f16((uint16_t)(r.dd >> 16)) * (float)(r.sm >> 8) };
    }
    __device__ static float dot(const Dec& d, const XAct& x) {
        int idot = 0;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            idot = __builtin_amdgcn_sdot4((int)( d.q[j]       & 0x0F0F0F0Fu), x.w[j],     idot, false);
            idot = __builtin_amdgcn_sdot4((int)((d.q[j] >> 4) & 0x0F0F0F0Fu), x.w[j + 4], idot, false);
        }
        return d.dsc * x.d * (float)idot - d.deff * x.s;
    }
};

#include "matvec_batched_nr_entries.h"
BNR_ENTRIES(matvec_q4k_repacked_batched, WQ4K)
