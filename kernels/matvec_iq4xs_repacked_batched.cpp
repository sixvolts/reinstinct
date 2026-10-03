// IQ4_XS small-batch matvec (spec-decode verify, 1..8 activation rows)
// — derived from the Q4_0 kernel of the same name.
//
// The repacked layout is byte-identical to Q4_0's (quant::iq4_xs::
// repack_for_matvec): a 16-byte nibble plane per 32-weight sub-block and
// an fp16 scale plane, here with the super-block d and 6-bit sub-scale
// pre-folded into `dl = d * (ls - 32)`. Two differences from Q4_0:
//   * each nibble maps through the 16-entry IQ4_NL codebook before the
//     dot, instead of the linear `n - 8`;
//   * the codebook values are already signed, so there is no constant
//     offset and no `8 * xqsum` term.
// Everything else — the shared matvec_batched_nr.h body, entry points — is Q4_0's.
//
// Layout:
//   slab + 0                    : out_dim × nsp × 16  nibble bytes
//   slab + out_dim × nsp × 16   : out_dim × nsp × 2   fp16 dl
//   nsp = (n_sub power of two) ? n_sub + 1 : n_sub   // anti-alias

#include "matvec_batched_nr.h"


// IQ4_NL / IQ4_XS codebook — the 16 int8 values a nibble indexes, packed
// four per 32-bit word (little-endian): entries 0-3, 4-7, 8-11, 12-15.
//   {-127,-104,-83,-65,-49,-35,-22,-10, 1,13,25,38,53,69,89,113}
// Overridable: IQ3_S repacks into this same layout with the codebook
// {-15,-13,...,15} and compiles this source with the four words
// predefined (quant::iq3_s::kernel_source).
#ifndef IQ4NL_KV_0_3
#define IQ4NL_KV_0_3   0xbfad9881u
#define IQ4NL_KV_4_7   0xf6eaddcfu
#define IQ4NL_KV_8_11  0x26190d01u
#define IQ4NL_KV_12_15 0x71594535u
#endif

// Map 4 nibbles (one per byte of `n4`, masked to 0x0F0F0F0F) to 4 int8
// codebook values packed for sdot4 — in registers, via v_perm_b32.
//
// A scalar table lookup here was catastrophic: 8 constant-memory byte
// loads per sdot4 inside the compute-bound MMQ loop made prefill 5.8x
// slower than the Q8_0 transcode it was meant to beat. v_perm selects
// bytes from an 8-byte pair per selector byte, so the low 3 bits of each
// nibble pick within a half-table, and bit 3 chooses which half.
__device__ __forceinline__ int iq4xs_lut4(uint32_t n4) {
    const uint32_t sel = n4 & 0x07070707u;
    // perm(a, b, sel): selector 0-3 -> bytes of b, 4-7 -> bytes of a.
    const uint32_t lo = __builtin_amdgcn_perm(IQ4NL_KV_4_7,   IQ4NL_KV_0_3,  sel);
    const uint32_t hi = __builtin_amdgcn_perm(IQ4NL_KV_12_15, IQ4NL_KV_8_11, sel);
    const uint32_t m  = ((n4 >> 3) & 0x01010101u) * 0xFFu;   // 0xFF where nibble >= 8
    return (int)((hi & m) | (lo & ~m));
}

struct WIQ4XS {
    const uint4* nib; const uint16_t* dp; unsigned int nsp;
    __device__ WIQ4XS(const uint8_t* w, unsigned int out_dim, unsigned int nsp_, unsigned int)
        : nib(reinterpret_cast<const uint4*>(w)),
          dp(reinterpret_cast<const uint16_t*>(w + (size_t)out_dim * nsp_ * 16)), nsp(nsp_) {}
    struct Raw { uint4 q; uint16_t d; };
    struct Dec { uint32_t q[4]; float dw; };
    static constexpr bool USES_XSUM = false;
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
            idot = __builtin_amdgcn_sdot4(iq4xs_lut4( d.q[j]       & 0x0F0F0F0Fu), x.w[j],     idot, false);
            idot = __builtin_amdgcn_sdot4(iq4xs_lut4((d.q[j] >> 4) & 0x0F0F0F0Fu), x.w[j + 4], idot, false);
        }
        return d.dw * x.d * (float)idot;
    }
};

#include "matvec_batched_nr_entries.h"
BNR_ENTRIES(matvec_iq4xs_repacked_batched, WIQ4XS)
