// P1b: fork mul_mat_vec_q4k_repacked<false> vs P1 variant vs reinstinct matvec_q4k_repacked_f32,
// same repacked weights, graph-replayed, 27B Q4_K decode shapes.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include "matvec_q4k_repacked.cpp"

struct block_q8_1 { __half2 ds; int8_t qs[32]; };
static_assert(sizeof(block_q8_1) == 36, "");
__device__ __forceinline__ float wsum64(float v) {
    for (int o = 32; o > 0; o >>= 1) v += __shfl_xor(v, o, 64);
    return v;
}
#define DP4A(a, b, c) __builtin_amdgcn_sdot4((a), (b), (c), false)

// fork kernel, non-IDS path, verbatim math (GUARD=1) or clamped + loads hoisted (GUARD=0)
template <int GUARD>
__global__ void fork_q4k(const uint8_t* __restrict__ wbase, const block_q8_1* __restrict__ xq,
                         float* __restrict__ y, const uint32_t ne0, const uint32_t ne1) {
    constexpr int ROWS = 2;
    const int wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
    const int row0 = blockIdx.x * (ROWS * 4) + wave * ROWS;
    const uint32_t n_sub = ne0 >> 5;
    const uint32_t nsp = ((n_sub & (n_sub - 1u)) == 0u) ? (n_sub + 1u) : n_sub;
    const uint4* nib = reinterpret_cast<const uint4*>(wbase);
    const uint16_t* smp = reinterpret_cast<const uint16_t*>(wbase + (size_t)ne1 * nsp * 16);
    const uint32_t* ddp = reinterpret_cast<const uint32_t*>(wbase + (size_t)ne1 * nsp * 16 + (size_t)ne1 * nsp * 2);
    const uint32_t n_super = n_sub >> 3;
    float acc[ROWS] = {0.0f, 0.0f};
    for (uint32_t sb = lane; sb < n_sub; sb += 64) {
        uint4 q[ROWS]; uint16_t sm[ROWS]; uint32_t dd[ROWS];
        if (GUARD != 1) {
#pragma unroll
            for (int r = 0; r < ROWS; r++) {
                const int row = min(row0 + r, (int)ne1 - 1);
                q[r] = nib[(size_t)row * nsp + sb]; sm[r] = smp[(size_t)row * nsp + sb];
                dd[r] = ddp[(size_t)row * n_super + (sb >> 3)];
            }
        }
        const block_q8_1* xb = xq + sb;
        const float dx = __low2float(xb->ds), sx = __high2float(xb->ds);
        const int* xq32 = reinterpret_cast<const int*>(xb->qs);
        if (GUARD == 2) __builtin_amdgcn_sched_barrier(0);
#pragma unroll
        for (int r = 0; r < ROWS; r++) {
            const int row = row0 + r;
            if (GUARD == 1) {
                if (row >= (int)ne1) continue;
                q[r] = nib[(size_t)row * nsp + sb]; sm[r] = smp[(size_t)row * nsp + sb];
                dd[r] = ddp[(size_t)row * n_super + (sb >> 3)];
            }
            const uint16_t d_bits = (uint16_t)(dd[r] & 0xFFFF), dmin_bits = (uint16_t)(dd[r] >> 16);
            const float dsc = __half2float(*reinterpret_cast<const __half*>(&d_bits)) * (float)(sm[r] & 0xFFu);
            const float deff = __half2float(*reinterpret_cast<const __half*>(&dmin_bits)) * (float)(sm[r] >> 8);
            const uint32_t qa[4] = {q[r].x, q[r].y, q[r].z, q[r].w};
            int idot = 0;
#pragma unroll
            for (int j = 0; j < 4; j++) {
                idot = DP4A((int)(qa[j] & 0x0F0F0F0Fu), xq32[j], idot);
                idot = DP4A((int)((qa[j] >> 4) & 0x0F0F0F0Fu), xq32[j + 4], idot);
            }
            acc[r] += dsc * dx * (float)idot - deff * sx;
        }
    }
#pragma unroll
    for (int r = 0; r < ROWS; r++) {
        const float a = wsum64(acc[r]);
        if (lane == 0 && (row0 + r) < (int)ne1) y[row0 + r] = a;
    }
}

template <class F>
float time_graph(F launch, int iters) {
    hipStream_t s; hipStreamCreate(&s);
    hipGraph_t g; hipGraphExec_t ge;
    hipStreamBeginCapture(s, hipStreamCaptureModeGlobal);
    for (int i = 0; i < iters; i++) launch(s, i);
    hipStreamEndCapture(s, &g); hipGraphInstantiate(&ge, g, nullptr, nullptr, 0);
    hipGraphLaunch(ge, s); hipStreamSynchronize(s);
    hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
    float best = 1e9;
    for (int rep = 0; rep < 5; rep++) {
        hipEventRecord(e0, s); hipGraphLaunch(ge, s); hipEventRecord(e1, s); hipEventSynchronize(e1);
        float ms; hipEventElapsedTime(&ms, e0, e1); best = fminf(best, ms * 1000.f / iters);
    }
    hipGraphExecDestroy(ge); hipGraphDestroy(g); hipStreamDestroy(s);
    return best;
}

int main() {
    // (out_dim, in_dim) at the 27B's Q4_K grids
    const int shapes[][2] = {{17408, 5120}, {12288, 5120}, {10240, 5120}, {6144, 5120}, {5120, 17408}, {1024, 5120}};
    const size_t pool = 512ull << 20;
    uint8_t* w; hipMalloc(&w, pool);
    std::vector<uint8_t> h(pool);
    uint32_t st = 1;
    for (size_t i = 0; i < pool; i++) { st = st * 1664525u + 1013904223u; h[i] = st >> 24; }
    // keep fp16 scale planes finite: clear exponent top bits on every 2nd byte (layout-agnostic, timing only)
    for (size_t i = 1; i < pool; i += 2) h[i] &= 0x3B;
    hipMemcpy(w, h.data(), pool, hipMemcpyHostToDevice);
    void *xa, *xb; float* y;
    hipMalloc(&xa, 17408 / 32 * 40); hipMalloc(&xb, 17408 / 32 * 40); hipMalloc(&y, 17408 * 4);
    hipMemset(xa, 0, 17408 / 32 * 40); hipMemset(xb, 0, 17408 / 32 * 40);
    printf("%6s %6s %8s | %8s %8s %8s %8s | GB/s fork / P1 / P1+fence / ri\n", "out", "in", "MB", "fork", "P1", "P1+fnc", "ri");
    for (auto& sh : shapes) {
        const uint32_t out = sh[0], in = sh[1], n_sub = in / 32, nsp = (n_sub & (n_sub - 1)) ? n_sub : n_sub + 1;
        const size_t bytes = (size_t)out * nsp * 18 + (size_t)out * (n_sub / 8) * 4;
        const int copies = (int)(pool / bytes) < 8 ? (int)(pool / bytes) : 8;   // rotate to defeat L2
        const dim3 grid((out + 7) / 8);
        auto wp = [&](int i) { return w + (size_t)(i % copies) * bytes; };
        float tf = time_graph([&](hipStream_t s, int i) { fork_q4k<1><<<grid, 256, 0, s>>>(wp(i), (block_q8_1*)xa, y, in, out); }, 200);
        float tp = time_graph([&](hipStream_t s, int i) { fork_q4k<0><<<grid, 256, 0, s>>>(wp(i), (block_q8_1*)xa, y, in, out); }, 200);
        float tb = time_graph([&](hipStream_t s, int i) { fork_q4k<2><<<grid, 256, 0, s>>>(wp(i), (block_q8_1*)xa, y, in, out); }, 200);
        float tr = time_graph([&](hipStream_t s, int i) { matvec_q4k_repacked_f32<<<grid, 256, 0, s>>>(wp(i), (const BlockQ8*)xb, y, in, out); }, 200);
        printf("%6u %6u %8.1f | %8.1f %8.1f %8.1f %8.1f | %4.0f / %4.0f / %4.0f / %4.0f\n", out, in, bytes / 1e6, tf, tp, tb, tr,
               bytes / tf / 1e3, bytes / tp / 1e3, bytes / tb / 1e3, bytes / tr / 1e3);
    }
}
