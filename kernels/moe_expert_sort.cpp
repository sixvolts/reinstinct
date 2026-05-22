// MoE expert-routing counting sort.
//
// Groups the P*n_used (token, slot) routing entries by expert id so the
// grouped-expert GEMM can run one tiled GEMM per expert over a
// contiguous token block (weight read once, reused across the block).
//
// Pipeline (all stream-ordered, no host sync):
//   moe_sort_zero      count[]  <- 0
//   moe_sort_histogram count[e] <- # entries routed to expert e
//   moe_sort_scan      expert_off[], tile_off[]   (exclusive prefix sums)
//   moe_sort_zero      cursor[] <- 0
//   moe_sort_scatter   perm[]   <- entry indices grouped by expert
//
// `ids` is the topk output [n_tok, n_used] row-major; entry index
// i = tok*n_used + slot. `perm[p]` is the entry index at sorted
// position p; expert e owns perm[expert_off[e] .. expert_off[e+1]].
// `tile_off[e]` is the prefix sum of ceil(count/BN) — the grouped GEMM
// maps a workgroup tile index back to its expert through it.
// `expert_off` / `tile_off` have n_expert+1 entries (the last is the
// total). A token may route the same expert twice across slots; each
// (token, slot) entry is sorted independently, which is correct — the
// combine step sums per-slot contributions.

#include <hip/hip_runtime.h>
#include <stdint.h>

extern "C" __global__
void moe_sort_zero(int* __restrict__ buf, unsigned int n) {
    unsigned int i = blockIdx.x * 256u + threadIdx.x;
    if (i < n) buf[i] = 0;
}

extern "C" __global__
void moe_sort_histogram(const int* __restrict__ ids,
                        int* __restrict__ count,
                        unsigned int n_entries, unsigned int n_expert) {
    unsigned int i = blockIdx.x * 256u + threadIdx.x;
    if (i >= n_entries) return;
    int e = ids[i];
    if (e >= 0 && (unsigned int)e < n_expert) atomicAdd(&count[e], 1);
}

// One workgroup, serial on thread 0: n_expert (≈128) iterations is
// microseconds and keeps the prefix sum trivially correct.
extern "C" __global__
void moe_sort_scan(const int* __restrict__ count,
                   int* __restrict__ expert_off,
                   int* __restrict__ tile_off,
                   unsigned int n_expert, unsigned int bn) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    int eacc = 0, tacc = 0;
    for (unsigned int e = 0; e < n_expert; e++) {
        expert_off[e] = eacc;
        tile_off[e]   = tacc;
        unsigned int c = (unsigned int)count[e];
        eacc += (int)c;
        tacc += (int)((c + bn - 1u) / bn);
    }
    expert_off[n_expert] = eacc;
    tile_off[n_expert]   = tacc;
}

extern "C" __global__
void moe_sort_scatter(const int* __restrict__ ids,
                      const int* __restrict__ expert_off,
                      int* __restrict__ cursor,
                      int* __restrict__ perm,
                      unsigned int n_entries, unsigned int n_expert) {
    unsigned int i = blockIdx.x * 256u + threadIdx.x;
    if (i >= n_entries) return;
    int e = ids[i];
    if (e < 0 || (unsigned int)e >= n_expert) return;
    int rank = atomicAdd(&cursor[e], 1);
    perm[expert_off[e] + rank] = (int)i;
}

// Gather quantised activation rows into expert-contiguous order:
// xq_dst[p] = xq_src[ perm[p] / n_used ].  One BlockQ8 (40 B = 10 u32)
// per thread. grid (ceil(nsub/256), n_entries). Used for the grouped
// GEMM's gate/up input — every slot of a token shares one activation.
extern "C" __global__
void moe_gather_xq(const unsigned char* __restrict__ xq_src,
                   const int* __restrict__ perm,
                   unsigned char* __restrict__ xq_dst,
                   unsigned int nsub, unsigned int n_used,
                   unsigned int n_entries) {
    unsigned int p  = blockIdx.y;
    unsigned int sb = blockIdx.x * 256u + threadIdx.x;
    if (p >= n_entries || sb >= nsub) return;
    unsigned int tok = (unsigned int)perm[p] / n_used;
    const uint32_t* s = reinterpret_cast<const uint32_t*>(
        xq_src + ((size_t)tok * nsub + sb) * 40);
    uint32_t* d = reinterpret_cast<uint32_t*>(
        xq_dst + ((size_t)p * nsub + sb) * 40);
    #pragma unroll
    for (int i = 0; i < 10; i++) d[i] = s[i];
}

// Scatter sorted-order rows back to entry order: dst[perm[p]] = src[p].
// grid (ceil(dim/256), n_entries). `dim` floats per row.
extern "C" __global__
void moe_scatter_rows(const float* __restrict__ src,
                      const int* __restrict__ perm,
                      float* __restrict__ dst,
                      unsigned int dim, unsigned int n_entries) {
    unsigned int p = blockIdx.y;
    unsigned int c = blockIdx.x * 256u + threadIdx.x;
    if (p >= n_entries || c >= dim) return;
    dst[(size_t)perm[p] * dim + c] = src[(size_t)p * dim + c];
}
