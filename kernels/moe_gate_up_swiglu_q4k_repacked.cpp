// Fused MoE expert gate+up matvec + SwiGLU — repacked Q4_K, dp4a.
//
// Replaces three kernels (gate matvec, up matvec, swiglu) with one: per
// (expert, output row) it streams the gate and up weight rows from the
// two expert slabs, computes both dot products against the shared
// quantised activation, and writes silu(gate)·up directly. Halving the
// expert-FFN matvec count and doubling each launch's weight stream
// sustains HBM bandwidth far better than the small per-expert matvecs.
//
// Body mirrors moe_matvec_q4k_repacked (same repacked two-plane layout);
// see that kernel for the layout derivation.
//
// grid = (ceil(out_dim/8), n_used, n_tok); 256-thread workgroup, ROWS=2.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#define ROWS 2

struct __attribute__((packed)) BlockQ8 {
    float  d;
    float  xsum;
    int8_t qs[32];
};
static_assert(sizeof(BlockQ8) == 40, "BlockQ8 must be 40 bytes");

extern "C" __global__
void moe_gate_up_swiglu_q4k_repacked_f32(const unsigned char* __restrict__ gate_slab,
                                         const unsigned char* __restrict__ up_slab,
                                         const int*       __restrict__ ids,
                                         const BlockQ8*   __restrict__ xq,
                                         float*           __restrict__ y,  // [n_tok,n_used,out_dim]
                                         unsigned int in_dim,
                                         unsigned int out_dim,
                                         unsigned int bytes_per_expert,
                                         unsigned int xq_tok_stride,
                                         unsigned int xq_slot_stride,
                                         unsigned int n_used)
{
    const int tok  = blockIdx.z;
    const int slot = blockIdx.y;
    const int eid  = ids[(size_t)tok * n_used + slot];
    const uint8_t* gbase = gate_slab + (size_t)eid * bytes_per_expert;
    const uint8_t* ubase = up_slab   + (size_t)eid * bytes_per_expert;
    const BlockQ8* xqs = xq + (size_t)tok * xq_tok_stride
                            + (size_t)slot * xq_slot_stride;
    float* yo = y + ((size_t)tok * n_used + slot) * out_dim;

    const int wave = threadIdx.x >> 6;
    const int lane = threadIdx.x & 63;
    const int row0 = blockIdx.x * (ROWS * 4) + wave * ROWS;
    const unsigned int n_sub = in_dim >> 5;
    const unsigned int nsp = ((n_sub & (n_sub - 1u)) == 0u) ? (n_sub + 1u) : n_sub;
    const unsigned int n_super = n_sub >> 3;

    const uint4*    g_nib = reinterpret_cast<const uint4*>(gbase);
    const uint16_t* g_smp = reinterpret_cast<const uint16_t*>(gbase + (size_t)out_dim*nsp*16);
    const uint32_t* g_ddp = reinterpret_cast<const uint32_t*>(
        gbase + (size_t)out_dim*nsp*16 + (size_t)out_dim*nsp*2);
    const uint4*    u_nib = reinterpret_cast<const uint4*>(ubase);
    const uint16_t* u_smp = reinterpret_cast<const uint16_t*>(ubase + (size_t)out_dim*nsp*16);
    const uint32_t* u_ddp = reinterpret_cast<const uint32_t*>(
        ubase + (size_t)out_dim*nsp*16 + (size_t)out_dim*nsp*2);

    float accg[ROWS], accu[ROWS];
    #pragma unroll
    for (int r = 0; r < ROWS; r++) { accg[r] = 0.0f; accu[r] = 0.0f; }

    for (unsigned int sb = lane; sb < n_sub; sb += 64) {
        const BlockQ8* xb   = xqs + sb;
        const float    dx   = xb->d;
        const float    xsum = xb->xsum;
        const int*     xq32 = reinterpret_cast<const int*>(xb->qs);

        #pragma unroll
        for (int r = 0; r < ROWS; r++) {
            const int row = row0 + r;
            if (row >= (int)out_dim) continue;
            const uint32_t dd_g = g_ddp[(size_t)row * n_super + (sb >> 3)];
            const uint32_t dd_u = u_ddp[(size_t)row * n_super + (sb >> 3)];
            const uint16_t sm_g = g_smp[(size_t)row * nsp + sb];
            const uint16_t sm_u = u_smp[(size_t)row * nsp + sb];
            const uint4    q_g  = g_nib[(size_t)row * nsp + sb];
            const uint4    q_u  = u_nib[(size_t)row * nsp + sb];

            const uint16_t gdb = (uint16_t)(dd_g & 0xFFFF), gmb = (uint16_t)(dd_g >> 16);
            const uint16_t udb = (uint16_t)(dd_u & 0xFFFF), umb = (uint16_t)(dd_u >> 16);
            const float gdsc  = __half2float(*reinterpret_cast<const __half*>(&gdb))
                                * (float)(sm_g & 0xFFu);
            const float gdeff = __half2float(*reinterpret_cast<const __half*>(&gmb))
                                * (float)(sm_g >> 8);
            const float udsc  = __half2float(*reinterpret_cast<const __half*>(&udb))
                                * (float)(sm_u & 0xFFu);
            const float udeff = __half2float(*reinterpret_cast<const __half*>(&umb))
                                * (float)(sm_u >> 8);

            const uint32_t ga[4] = { q_g.x, q_g.y, q_g.z, q_g.w };
            const uint32_t ua[4] = { q_u.x, q_u.y, q_u.z, q_u.w };
            int gdot = 0, udot = 0;
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                gdot = __builtin_amdgcn_sdot4((int)( ga[j]       & 0x0F0F0F0Fu),
                                              xq32[j],     gdot, false);
                gdot = __builtin_amdgcn_sdot4((int)((ga[j] >> 4) & 0x0F0F0F0Fu),
                                              xq32[j + 4], gdot, false);
                udot = __builtin_amdgcn_sdot4((int)( ua[j]       & 0x0F0F0F0Fu),
                                              xq32[j],     udot, false);
                udot = __builtin_amdgcn_sdot4((int)((ua[j] >> 4) & 0x0F0F0F0Fu),
                                              xq32[j + 4], udot, false);
            }
            accg[r] += gdsc * dx * (float)gdot - gdeff * xsum;
            accu[r] += udsc * dx * (float)udot - udeff * xsum;
        }
    }

    #pragma unroll
    for (int r = 0; r < ROWS; r++) {
        float g = accg[r], u = accu[r];
        g += __shfl_xor(g, 32); g += __shfl_xor(g, 16); g += __shfl_xor(g, 8);
        g += __shfl_xor(g,  4); g += __shfl_xor(g,  2); g += __shfl_xor(g, 1);
        u += __shfl_xor(u, 32); u += __shfl_xor(u, 16); u += __shfl_xor(u, 8);
        u += __shfl_xor(u,  4); u += __shfl_xor(u,  2); u += __shfl_xor(u, 1);
        if (lane == 0 && (row0 + r) < (int)out_dim) {
            const float silu = g / (1.0f + __expf(-g));   // SwiGLU: silu(gate)·up
            yo[row0 + r] = silu * u;
        }
    }
}
