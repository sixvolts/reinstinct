// F32-weight GEMM, 64x64 output tile per 256-thread block, 4x4 outputs per
// thread, K staged through LDS in 16-wide k-major slices so the inner loop
// reads float4s. The llama fork's gcn_f32_gemm_tn_rb (gfx906-perf,
// ggml-cuda.cu; crossport ledger L8c), verbatim apart from the entry points.
//
// Contract: C[m + n*ldc] = sum_k A[m*lda + k] * B[n*ldb + k]. Reinstinct uses
// A = the [out, in] weight, B = the [rows, in] activations, C = [rows, out]
// (ldc = out). grid = (ceil(N/64), ceil(M/64)), block = 256. The _vec entry
// needs lda, ldb and the base pointers 16-byte aligned (K % 4 == 0).
#include <hip/hip_runtime.h>

#define GCN_F32_RB_BM 64
#define GCN_F32_RB_BN 64
#define GCN_F32_RB_BK 16

template <bool VEC>
__device__ __forceinline__ void gcn_f32_gemm_tn_rb(
        const float * __restrict__ A, const float * __restrict__ B, float * __restrict__ C,
        const int M, const int N, const int K, const int lda, const int ldb, const int ldc,
        const int k_begin) {
    __shared__ float As[GCN_F32_RB_BK][GCN_F32_RB_BM + 4];
    __shared__ float Bs[GCN_F32_RB_BK][GCN_F32_RB_BN + 4];

    const int tid = threadIdx.x;
    const int tn  = tid % 16;
    const int tm  = tid / 16;
    const int m0  = blockIdx.y * GCN_F32_RB_BM;
    const int n0  = blockIdx.x * GCN_F32_RB_BN;

    // each thread loads 4 consecutive k of one row of A and one row of B per slice
    const int lr = tid / 4;
    const int lk = (tid % 4) * 4;

    float acc[4][4] = {{0.0f}};

    for (int k0 = k_begin; k0 < K; k0 += GCN_F32_RB_BK) {
        const int k = k0 + lk;
        float4 a = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 b = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        const int m = m0 + lr;
        const int n = n0 + lr;
        if (VEC && k + 3 < K) {
            if (m < M) { a = *(const float4 *) (A + (size_t) m * lda + k); }
            if (n < N) { b = *(const float4 *) (B + (size_t) n * ldb + k); }
        } else {
            if (m < M) {
                a.x = k + 0 < K ? A[(size_t) m * lda + k + 0] : 0.0f;
                a.y = k + 1 < K ? A[(size_t) m * lda + k + 1] : 0.0f;
                a.z = k + 2 < K ? A[(size_t) m * lda + k + 2] : 0.0f;
                a.w = k + 3 < K ? A[(size_t) m * lda + k + 3] : 0.0f;
            }
            if (n < N) {
                b.x = k + 0 < K ? B[(size_t) n * ldb + k + 0] : 0.0f;
                b.y = k + 1 < K ? B[(size_t) n * ldb + k + 1] : 0.0f;
                b.z = k + 2 < K ? B[(size_t) n * ldb + k + 2] : 0.0f;
                b.w = k + 3 < K ? B[(size_t) n * ldb + k + 3] : 0.0f;
            }
        }
        As[lk + 0][lr] = a.x; As[lk + 1][lr] = a.y; As[lk + 2][lr] = a.z; As[lk + 3][lr] = a.w;
        Bs[lk + 0][lr] = b.x; Bs[lk + 1][lr] = b.y; Bs[lk + 2][lr] = b.z; Bs[lk + 3][lr] = b.w;
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < GCN_F32_RB_BK; ++kk) {
            const float4 av = *(const float4 *) &As[kk][tm * 4];
            const float4 bv = *(const float4 *) &Bs[kk][tn * 4];
            const float ar[4] = {av.x, av.y, av.z, av.w};
            const float br[4] = {bv.x, bv.y, bv.z, bv.w};
#pragma unroll
            for (int i = 0; i < 4; ++i) {
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc[i][j] += ar[i] * br[j];
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int n = n0 + tn * 4 + j;
        if (n >= N) {
            continue;
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int m = m0 + tm * 4 + i;
            if (m < M) {
                C[(size_t) m + (size_t) n * ldc] = acc[i][j];
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(256) void gemm_f32_tn_vec_f32(
        const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
        int M, int N, int K, int lda, int ldb, int ldc)
{ gcn_f32_gemm_tn_rb<true>(A, B, C, M, N, K, lda, ldb, ldc, 0); }

extern "C" __global__ __launch_bounds__(256) void gemm_f32_tn_f32(
        const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
        int M, int N, int K, int lda, int ldb, int ldc)
{ gcn_f32_gemm_tn_rb<false>(A, B, C, M, N, K, lda, ldb, ldc, 0); }

// Split-K: blockIdx.z takes k in [z * k_chunk, min(K, (z + 1) * k_chunk))
// and writes its partial C to C + z * split_stride; gemm_f32_splitk_reduce
// then sums the slices in z order (deterministic, unlike atomics). For
// skinny problems — the MoE router (256 tokens x 2048 -> 256) and the
// GDN alpha/beta projections (-> 32) give 8-16 tiles on a 60-CU card.
// k_chunk is a multiple of GCN_F32_RB_BK.
extern "C" __global__ __launch_bounds__(256) void gemm_f32_tn_splitk_vec_f32(
        const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
        int M, int N, int K, int lda, int ldb, int ldc, int k_chunk, long long split_stride)
{
    const int kb = blockIdx.z * k_chunk;
    gcn_f32_gemm_tn_rb<true>(A, B, C + (size_t)blockIdx.z * split_stride,
                             M, N, min(K, kb + k_chunk), lda, ldb, ldc, kb);
}

extern "C" __global__ __launch_bounds__(256) void gemm_f32_tn_splitk_f32(
        const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
        int M, int N, int K, int lda, int ldb, int ldc, int k_chunk, long long split_stride)
{
    const int kb = blockIdx.z * k_chunk;
    gcn_f32_gemm_tn_rb<false>(A, B, C + (size_t)blockIdx.z * split_stride,
                              M, N, min(K, kb + k_chunk), lda, ldb, ldc, kb);
}

// y[i] = sum_z part[z * n + i], z ascending.
extern "C" __global__ __launch_bounds__(256) void gemm_f32_splitk_reduce(
        const float* __restrict__ part, float* __restrict__ y, long long n, int splits)
{
    const long long i = (long long)blockIdx.x * 256 + threadIdx.x;
    if (i >= n) return;
    float acc = part[i];
    for (int z = 1; z < splits; z++) acc += part[(size_t)z * n + i];
    y[i] = acc;
}
