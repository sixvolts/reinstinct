// Apply the forward Randomized Hadamard Transform to Q.
//
// The cold-tier K cache stores values in rotated space (signs1 × FWHT
// × 1/√N × signs2 of the original). Because RHT is orthonormal, the
// dot product is rotation-invariant:
//
//     <Q, K_orig> == <R·Q, R·K_orig> == <R·Q, K_stored>
//
// So if we pre-rotate Q ONCE per attention call, the per-position cold
// score becomes just a dot product against the un-rotated stored
// centroids — eliminating the FWHT-128 per cached position that
// dominates the naive attn_partial_superquant.cpp.
//
// Grid: (n_heads, groups_per_head). Block: 128 threads (one per rotation
// group element, since ROT_GROUP=128).
//
// The same kernel is reused for K-side and V-side rotations by passing
// the matching `signs1` / `signs2` arrays at launch (CacheKind::K /
// CacheKind::V have independent rotations).

#include <hip/hip_runtime.h>
#include <stdint.h>

constexpr int ROT_GROUP = 128;

extern "C" __global__ __launch_bounds__(128, 4)
void rotate_q_rht_f32(const float*  __restrict__ q,        // [n_heads, head_dim]
                      const int8_t* __restrict__ signs1,    // [128]
                      const int8_t* __restrict__ signs2,    // [128]
                      float*        __restrict__ q_rot,     // [n_heads, head_dim]
                      unsigned int n_heads,
                      unsigned int head_dim)
{
    const unsigned int h   = blockIdx.x;
    const unsigned int grp = blockIdx.y;
    const unsigned int groups_per_head = head_dim / ROT_GROUP;
    if (h >= n_heads || grp >= groups_per_head) return;

    const int lane = threadIdx.x;
    __shared__ float x[ROT_GROUP];

    // Load + first sign flip.
    const float v_in = q[(size_t)h * head_dim + grp * ROT_GROUP + lane];
    x[lane] = v_in * (float)signs1[lane];
    __syncthreads();

    // FWHT-128.
    for (int hstride = 1; hstride < ROT_GROUP; hstride *= 2) {
        if ((lane & hstride) == 0) {
            const float a = x[lane];
            const float b = x[lane + hstride];
            x[lane]            = a + b;
            x[lane + hstride]  = a - b;
        }
        __syncthreads();
    }

    // 1/√128 scale + second sign flip → final Q_rot.
    const float scale = 1.0f / sqrtf((float)ROT_GROUP);
    const float out = x[lane] * scale * (float)signs2[lane];
    q_rot[(size_t)h * head_dim + grp * ROT_GROUP + lane] = out;
}
