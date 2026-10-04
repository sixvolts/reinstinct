// A/B of reinstinct's mmq_gemm_q4_0_repacked_f32 vs the fork's mmq_gemm_nib_repacked<Q4_0, TN=4>
// on identical repacked Q4_0 weights and identical int8 activations (one GPU, hipEvent timing).
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>
#include <cstdint>

#include "ri_q4_0.cpp"   // reinstinct kernel (extern "C" __global__ mmq_gemm_q4_0_repacked_f32), BlockQ8 40 B

// ---- fork side (verbatim structure of repack-gcn.cu's nib tile, Q4_0 mode) ----
struct __attribute__((packed)) block_q8_1 { half2 ds; int8_t qs[32]; };
static_assert(sizeof(block_q8_1) == 36, "block_q8_1");
#define MMQ_RP_BK 4
#define MMQ_RP_TM 4
#define MMQ_RP_BM (16 * MMQ_RP_TM)
static __device__ __forceinline__ int ggml_cuda_dp4a(const int a, const int b, int c) { return __builtin_amdgcn_sdot4(a, b, c, false); }
static __device__ __forceinline__ uint32_t repack_sub8(const uint32_t v) { return ((v | 0x80808080u) - 0x08080808u) ^ 0x80808080u; }
static __device__ __forceinline__ void repack_nib_decode(const uint32_t q, int & lo, int & hi) {
    lo = (int) repack_sub8( q       & 0x0F0F0F0Fu);
    hi = (int) repack_sub8((q >> 4) & 0x0F0F0F0Fu);
}
template <int TN_>
static __global__ void __launch_bounds__(256, 2) mmq_gemm_nib_repacked(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1,
        const uint32_t n_tok, const uint32_t x_stride, const uint32_t dst_s1) {
    const int t  = threadIdx.x;
    const int tx = t & 15;
    const int ty = t >> 4;
    const uint32_t row0 = blockIdx.x * MMQ_RP_BM;
    const uint32_t tok0 = blockIdx.y * (16 * TN_);
    const uint32_t n_sub = ne0 >> 5;
    const uint32_t nsp   = ((n_sub & (n_sub - 1u)) == 0u) ? (n_sub + 1u) : n_sub;
    const uint4    * nib = reinterpret_cast<const uint4 *>(wbase);
    const uint16_t * dp  = reinterpret_cast<const uint16_t *>(wbase + (size_t) ne1 * nsp * 16);
    __shared__ int        sW8[MMQ_RP_BM][MMQ_RP_BK][8];
    __shared__ float      sWs[MMQ_RP_BM][MMQ_RP_BK];
    __shared__ block_q8_1 sX [(16 * TN_)][MMQ_RP_BK + 1];
    float acc[MMQ_RP_TM][TN_] = {};
    const int lr = t >> 2;
    const int lk = t & 3;
    const bool xstage = lr < (16 * TN_);
    for (uint32_t sb0 = 0; sb0 < n_sub; sb0 += MMQ_RP_BK) {
        const uint32_t sb   = sb0 + lk;
        const uint32_t wrow = row0 + lr;
        if (wrow < ne1 && sb < n_sub) {
            const size_t   idx = (size_t) wrow * nsp + sb;
            const uint4    q   = nib[idx];
            const uint32_t qa[4] = { q.x, q.y, q.z, q.w };
#pragma unroll
            for (int j = 0; j < 4; j++) {
                repack_nib_decode(qa[j], sW8[lr][lk][j], sW8[lr][lk][4 + j]);
            }
            const uint16_t db = dp[idx];
            sWs[lr][lk] = __half2float(*reinterpret_cast<const __half *>(&db));
        } else {
#pragma unroll
            for (int j = 0; j < 8; j++) sW8[lr][lk][j] = 0;
            sWs[lr][lk] = 0.0f;
        }
        if (xstage) {
            if (tok0 + lr < n_tok && sb < n_sub) {
                sX[lr][lk] = xq[(size_t) (tok0 + lr) * x_stride + sb];
            } else {
                sX[lr][lk].ds = make_half2(0.0f, 0.0f);
            }
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < MMQ_RP_BK; kk++) {
#pragma unroll
            for (int n = 0; n < TN_; n++) {
                const block_q8_1 * xb = &sX[tx + n * 16][kk];
                const int * xq32 = reinterpret_cast<const int *>(xb->qs);
                const float dx = __low2float(xb->ds);
#pragma unroll
                for (int r = 0; r < MMQ_RP_TM; r++) {
                    const int * w8 = sW8[ty + r * 16][kk];
                    int idot = 0;
#pragma unroll
                    for (int j = 0; j < 8; j++) idot = ggml_cuda_dp4a(w8[j], xq32[j], idot);
                    acc[r][n] += sWs[ty + r * 16][kk] * dx * (float) idot;
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < MMQ_RP_TM; r++) {
        const uint32_t row = row0 + ty + r * 16;
        if (row >= ne1) continue;
#pragma unroll
        for (int n = 0; n < TN_; n++) {
            const uint32_t tok = tok0 + tx + n * 16;
            if (tok < n_tok) y[(size_t) tok * dst_s1 + row] = acc[r][n];
        }
    }
}
struct __align__(8) mmq_x8 { float d; float s; int8_t qs[32]; };
static __device__ __forceinline__ void mmq_x8_stage(mmq_x8 & dst, const block_q8_1 & src) {
    dst.d = __low2float(src.ds); dst.s = __high2float(src.ds);
    const int * q = reinterpret_cast<const int *>(src.qs); int * o = reinterpret_cast<int *>(dst.qs);
#pragma unroll
    for (int j = 0; j < 8; j++) o[j] = q[j];
}
template <int TN_>
static __global__ void __launch_bounds__(256, 2) mmq_gemm_nib_repacked_x8(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1,
        const uint32_t n_tok, const uint32_t x_stride, const uint32_t dst_s1) {
    const int t  = threadIdx.x;
    const int tx = t & 15;
    const int ty = t >> 4;
    const uint32_t row0 = blockIdx.x * MMQ_RP_BM;
    const uint32_t tok0 = blockIdx.y * (16 * TN_);
    const uint32_t n_sub = ne0 >> 5;
    const uint32_t nsp   = ((n_sub & (n_sub - 1u)) == 0u) ? (n_sub + 1u) : n_sub;
    const uint4    * nib = reinterpret_cast<const uint4 *>(wbase);
    const uint16_t * dp  = reinterpret_cast<const uint16_t *>(wbase + (size_t) ne1 * nsp * 16);
    __shared__ int        sW8[MMQ_RP_BM][MMQ_RP_BK][8];
    __shared__ float      sWs[MMQ_RP_BM][MMQ_RP_BK];
    __shared__ mmq_x8 sX [(16 * TN_)][MMQ_RP_BK + 1];
    float acc[MMQ_RP_TM][TN_] = {};
    const int lr = t >> 2;
    const int lk = t & 3;
    const bool xstage = lr < (16 * TN_);
    for (uint32_t sb0 = 0; sb0 < n_sub; sb0 += MMQ_RP_BK) {
        const uint32_t sb   = sb0 + lk;
        const uint32_t wrow = row0 + lr;
        if (wrow < ne1 && sb < n_sub) {
            const size_t   idx = (size_t) wrow * nsp + sb;
            const uint4    q   = nib[idx];
            const uint32_t qa[4] = { q.x, q.y, q.z, q.w };
#pragma unroll
            for (int j = 0; j < 4; j++) {
                repack_nib_decode(qa[j], sW8[lr][lk][j], sW8[lr][lk][4 + j]);
            }
            const uint16_t db = dp[idx];
            sWs[lr][lk] = __half2float(*reinterpret_cast<const __half *>(&db));
        } else {
#pragma unroll
            for (int j = 0; j < 8; j++) sW8[lr][lk][j] = 0;
            sWs[lr][lk] = 0.0f;
        }
        if (xstage) {
            if (tok0 + lr < n_tok && sb < n_sub) {
                mmq_x8_stage(sX[lr][lk], xq[(size_t) (tok0 + lr) * x_stride + sb]);
            } else {
                sX[lr][lk].d = 0.0f; sX[lr][lk].s = 0.0f;
            }
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < MMQ_RP_BK; kk++) {
#pragma unroll
            for (int n = 0; n < TN_; n++) {
                const mmq_x8 * xb = &sX[tx + n * 16][kk];
                const int * xq32 = reinterpret_cast<const int *>(xb->qs);
                const float dx = xb->d;
#pragma unroll
                for (int r = 0; r < MMQ_RP_TM; r++) {
                    const int * w8 = sW8[ty + r * 16][kk];
                    int idot = 0;
#pragma unroll
                    for (int j = 0; j < 8; j++) idot = ggml_cuda_dp4a(w8[j], xq32[j], idot);
                    acc[r][n] += sWs[ty + r * 16][kk] * dx * (float) idot;
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < MMQ_RP_TM; r++) {
        const uint32_t row = row0 + ty + r * 16;
        if (row >= ne1) continue;
#pragma unroll
        for (int n = 0; n < TN_; n++) {
            const uint32_t tok = tok0 + tx + n * 16;
            if (tok < n_tok) y[(size_t) tok * dst_s1 + row] = acc[r][n];
        }
    }
}
template <int TN_>
static __global__ void __launch_bounds__(256, 2) mmq_gemm_nib_repacked_x8p(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1,
        const uint32_t n_tok, const uint32_t x_stride, const uint32_t dst_s1) {
    const int t  = threadIdx.x;
    const int tx = t & 15;
    const int ty = t >> 4;
    const uint32_t row0 = blockIdx.x * MMQ_RP_BM;
    const uint32_t tok0 = blockIdx.y * (16 * TN_);
    const uint32_t n_sub = ne0 >> 5;
    const uint32_t nsp   = ((n_sub & (n_sub - 1u)) == 0u) ? (n_sub + 1u) : n_sub;
    const uint4    * nib = reinterpret_cast<const uint4 *>(wbase);
    const uint16_t * dp  = reinterpret_cast<const uint16_t *>(wbase + (size_t) ne1 * nsp * 16);
    __shared__ __align__(16) int sW8[MMQ_RP_BM][MMQ_RP_BK * 8 + 4]; // 144 B rows: 2-way instead of 16-way bank conflicts
    __shared__ float      sWs[MMQ_RP_BM][MMQ_RP_BK];
    __shared__ mmq_x8 sX [(16 * TN_)][MMQ_RP_BK + 1];
    float acc[MMQ_RP_TM][TN_] = {};
    const int lr = t >> 2;
    const int lk = t & 3;
    const bool xstage = lr < (16 * TN_);
    for (uint32_t sb0 = 0; sb0 < n_sub; sb0 += MMQ_RP_BK) {
        const uint32_t sb   = sb0 + lk;
        const uint32_t wrow = row0 + lr;
        if (wrow < ne1 && sb < n_sub) {
            const size_t   idx = (size_t) wrow * nsp + sb;
            const uint4    q   = nib[idx];
            const uint32_t qa[4] = { q.x, q.y, q.z, q.w };
#pragma unroll
            for (int j = 0; j < 4; j++) {
                repack_nib_decode(qa[j], sW8[lr][lk * 8 + j], sW8[lr][lk * 8 + 4 + j]);
            }
            const uint16_t db = dp[idx];
            sWs[lr][lk] = __half2float(*reinterpret_cast<const __half *>(&db));
        } else {
#pragma unroll
            for (int j = 0; j < 8; j++) sW8[lr][lk * 8 + j] = 0;
            sWs[lr][lk] = 0.0f;
        }
        if (xstage) {
            if (tok0 + lr < n_tok && sb < n_sub) {
                mmq_x8_stage(sX[lr][lk], xq[(size_t) (tok0 + lr) * x_stride + sb]);
            } else {
                sX[lr][lk].d = 0.0f; sX[lr][lk].s = 0.0f;
            }
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < MMQ_RP_BK; kk++) {
#pragma unroll
            for (int n = 0; n < TN_; n++) {
                const mmq_x8 * xb = &sX[tx + n * 16][kk];
                const int * xq32 = reinterpret_cast<const int *>(xb->qs);
                const float dx = xb->d;
#pragma unroll
                for (int r = 0; r < MMQ_RP_TM; r++) {
                    const int * w8 = &sW8[ty + r * 16][kk * 8];
                    int idot = 0;
#pragma unroll
                    for (int j = 0; j < 8; j++) idot = ggml_cuda_dp4a(w8[j], xq32[j], idot);
                    acc[r][n] += sWs[ty + r * 16][kk] * dx * (float) idot;
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int r = 0; r < MMQ_RP_TM; r++) {
        const uint32_t row = row0 + ty + r * 16;
        if (row >= ne1) continue;
#pragma unroll
        for (int n = 0; n < TN_; n++) {
            const uint32_t tok = tok0 + tx + n * 16;
            if (tok < n_tok) y[(size_t) tok * dst_s1 + row] = acc[r][n];
        }
    }
}

#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("HIP error %s at %d\n", hipGetErrorString(e), __LINE__); exit(1); } } while (0)

static uint16_t f2h(float f) { __half h = __float2half(f); uint16_t b; memcpy(&b, &h, 2); return b; }

static void run(int K, int N, int P, int reps) {
    const int n_sub = K / 32; const int nsp = (n_sub & (n_sub - 1)) == 0 ? n_sub + 1 : n_sub;
    const size_t wbytes = (size_t) N * nsp * 18;
    std::vector<uint8_t> w(wbytes, 0);
    uint64_t s = 12345;
    auto rnd = [&]() { s = s * 6364136223846793005ull + 1442695040888963407ull; return (uint32_t) (s >> 33); };
    for (int r = 0; r < N; r++) for (int b = 0; b < n_sub; b++) {
        for (int k = 0; k < 16; k++) w[((size_t) r * nsp + b) * 16 + k] = (uint8_t) rnd();
        uint16_t d = f2h(0.001f + (rnd() % 1000) * 1e-6f); memcpy(&w[(size_t) N * nsp * 16 + ((size_t) r * nsp + b) * 2], &d, 2);
    }
    // activations: int8 blocks, both layouts from the same data
    const int x_stride = ((K + 511) / 512 * 512) / 32; // fork pads rows to 512
    std::vector<BlockQ8>    xa((size_t) P * n_sub);
    std::vector<block_q8_1> xb((size_t) P * x_stride);
    memset(xb.data(), 0, xb.size() * sizeof(block_q8_1));
    for (int t = 0; t < P; t++) for (int b = 0; b < n_sub; b++) {
        BlockQ8 & A = xa[(size_t) t * n_sub + b]; int sum = 0;
        for (int k = 0; k < 32; k++) { A.qs[k] = (int8_t) ((int) (rnd() % 255) - 127); sum += A.qs[k]; }
        A.d = 0.01f + (rnd() % 100) * 1e-4f; A.xsum = A.d * sum;
        block_q8_1 & B = xb[(size_t) t * x_stride + b]; memcpy(B.qs, A.qs, 32);
        B.ds = __halves2half2(__float2half(A.d), __float2half(A.d * sum));
    }
    uint8_t * dw; BlockQ8 * dxa; block_q8_1 * dxb; float * ya; float * yb;
    CK(hipMalloc(&dw, wbytes)); CK(hipMalloc(&dxa, xa.size() * sizeof(BlockQ8))); CK(hipMalloc(&dxb, xb.size() * sizeof(block_q8_1)));
    CK(hipMalloc(&ya, (size_t) P * N * 4)); CK(hipMalloc(&yb, (size_t) P * N * 4));
    CK(hipMemcpy(dw, w.data(), wbytes, hipMemcpyHostToDevice));
    CK(hipMemcpy(dxa, xa.data(), xa.size() * sizeof(BlockQ8), hipMemcpyHostToDevice));
    CK(hipMemcpy(dxb, xb.data(), xb.size() * sizeof(block_q8_1), hipMemcpyHostToDevice));
    dim3 grid((N + 63) / 64, (P + 63) / 64);
    auto ri = [&]() { hipLaunchKernelGGL(mmq_gemm_q4_0_repacked_f32, grid, dim3(256), 0, 0, dw, dxa, ya, (unsigned) K, (unsigned) N, (unsigned) P); };
    auto fk = [&]() { hipLaunchKernelGGL(mmq_gemm_nib_repacked<4>, grid, dim3(256), 0, 0, dw, dxb, yb, (uint32_t) K, (uint32_t) N, (uint32_t) P, (uint32_t) x_stride, (uint32_t) N); };
    auto fx = [&]() { hipLaunchKernelGGL(mmq_gemm_nib_repacked_x8<4>, grid, dim3(256), 0, 0, dw, dxb, yb, (uint32_t) K, (uint32_t) N, (uint32_t) P, (uint32_t) x_stride, (uint32_t) N); };
    auto fp = [&]() { hipLaunchKernelGGL(mmq_gemm_nib_repacked_x8p<4>, grid, dim3(256), 0, 0, dw, dxb, yb, (uint32_t) K, (uint32_t) N, (uint32_t) P, (uint32_t) x_stride, (uint32_t) N); };
    hipEvent_t e0, e1; CK(hipEventCreate(&e0)); CK(hipEventCreate(&e1));
    float tri = 0, tfk = 0, tfx = 0, tfp = 0;
    for (int pass = 0; pass < 2; pass++) {   // interleaved: ri, fk, fx, ri, fk, fx
        for (int i = 0; i < 3; i++) ri(); CK(hipDeviceSynchronize());
        CK(hipEventRecord(e0)); for (int i = 0; i < reps; i++) ri(); CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1)); float ms; CK(hipEventElapsedTime(&ms, e0, e1)); tri += ms;
        for (int i = 0; i < 3; i++) fk(); CK(hipDeviceSynchronize());
        CK(hipEventRecord(e0)); for (int i = 0; i < reps; i++) fk(); CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1)); CK(hipEventElapsedTime(&ms, e0, e1)); tfk += ms;
        for (int i = 0; i < 3; i++) fx(); CK(hipDeviceSynchronize());
        CK(hipEventRecord(e0)); for (int i = 0; i < reps; i++) fx(); CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1)); CK(hipEventElapsedTime(&ms, e0, e1)); tfx += ms;
        for (int i = 0; i < 3; i++) fp(); CK(hipDeviceSynchronize());
        CK(hipEventRecord(e0)); for (int i = 0; i < reps; i++) fp(); CK(hipEventRecord(e1)); CK(hipEventSynchronize(e1)); CK(hipEventElapsedTime(&ms, e0, e1)); tfp += ms;
    }
    tri /= 2 * reps; tfk /= 2 * reps; tfx /= 2 * reps; tfp /= 2 * reps;
    std::vector<float> ha((size_t) P * N), hb((size_t) P * N);
    CK(hipMemcpy(ha.data(), ya, ha.size() * 4, hipMemcpyDeviceToHost)); CK(hipMemcpy(hb.data(), yb, hb.size() * 4, hipMemcpyDeviceToHost));
    double num = 0, den = 0; for (size_t i = 0; i < ha.size(); i++) { double d = ha[i] - hb[i]; num += d * d; den += (double) ha[i] * ha[i]; }
    const double macs = (double) K * N * P;
    printf("K=%5d N=%5d P=%d: reinstinct %7.1f us (%5.2f TMAC/s)  fork 36B %7.1f (%5.2f)  40B %7.1f (%5.2f)  40B+padW %7.1f (%5.2f)  nmse %.2e\n",
        K, N, P, tri * 1e3, macs / (tri * 1e-3) * 1e-12, tfk * 1e3, macs / (tfk * 1e-3) * 1e-12, tfx * 1e3, macs / (tfx * 1e-3) * 1e-12, tfp * 1e3, macs / (tfp * 1e-3) * 1e-12, num / den);
    CK(hipFree(dw)); CK(hipFree(dxa)); CK(hipFree(dxb)); CK(hipFree(ya)); CK(hipFree(yb));
}

int main(int argc, char ** argv) {
    const int P = argc > 1 ? atoi(argv[1]) : 512;
    run(5376, 21504, P, 20);
    run(21504, 5376, P, 20);
    run(5376, 4096, P, 20);
    run(5120, 17408, P, 20);
    return 0;
}
