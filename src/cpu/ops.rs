//! Primitive ops for the Qwen 3.5 forward pass.
//!
//! Conventions:
//! - All math in f32. Buffers are passed as `&[f32]` / `&mut [f32]`.
//! - Weight tensors follow GGUF layout: shape `[in, out]` in ggml order
//!   means flat layout `w[out_idx * in + in_idx]` (the leftmost dim is
//!   the fastest-varying in ggml). Each output reads a contiguous row of
//!   length `in_dim`. See `matvec` for the canonical access pattern.
//! - RMSNorm uses Qwen 3.5 / Gemma "scale-shifted-from-zero" semantics:
//!   `out = normalize(x) * (1.0 + weight)`.

/// SiLU (a.k.a. Swish): `x * sigmoid(x)`.
#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Sigmoid.
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Softplus: `ln(1 + exp(x))`. Numerically stable for both signs.
#[inline]
pub fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

/// In-place RMSNorm:
///   out = x * rsqrt(mean(x^2) + eps) * weight
///
/// Important: Qwen 3.5's HF modeling code applies `(1.0 + weight)` because the
/// PyTorch weight is initialized to 0. **In the GGUF, that `+1` has already been
/// baked in** by `convert_hf_to_gguf.py` (`Qwen3NextModel.modify_tensors` does
/// `data_torch = data_torch + 1` for all `*norm.weight` except
/// `linear_attn.norm.weight`). So this kernel applies plain `weight` —
/// applying `(1+weight)` here would double-shift.
pub fn rmsnorm(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    let n = x.len();
    assert_eq!(weight.len(), n);
    assert_eq!(out.len(), n);
    let mut sum_sq = 0.0_f32;
    for &v in x { sum_sq += v * v; }
    let rrms = (sum_sq / n as f32 + eps).sqrt().recip();
    for i in 0..n {
        out[i] = x[i] * rrms * weight[i];
    }
}

/// Gated RMSNorm from `Qwen3_5RMSNormGated`:
///   out = (rmsnorm_no_shift(x) * weight) * silu(gate)
/// Note: `weight` is multiplied directly (no `1.0 +`) — this is the
/// non-shifted variant the GDN block uses internally.
pub fn rmsnorm_gated(
    x: &[f32], gate: &[f32], weight: &[f32], eps: f32, out: &mut [f32],
) {
    let n = x.len();
    assert_eq!(gate.len(), n);
    assert_eq!(weight.len(), n);
    assert_eq!(out.len(), n);
    let mut sum_sq = 0.0_f32;
    for &v in x { sum_sq += v * v; }
    let rrms = (sum_sq / n as f32 + eps).sqrt().recip();
    for i in 0..n {
        out[i] = x[i] * rrms * weight[i] * silu(gate[i]);
    }
}

/// L2 normalization along the vector: `x / sqrt(sum(x^2) + eps)`.
/// Used inside the GDN recurrence on Q and K (`use_qk_l2norm_in_kernel=True`).
pub fn l2norm(x: &[f32], eps: f32, out: &mut [f32]) {
    assert_eq!(out.len(), x.len());
    let mut sum_sq = 0.0_f32;
    for &v in x { sum_sq += v * v; }
    let inv = (sum_sq + eps).sqrt().recip();
    for i in 0..x.len() {
        out[i] = x[i] * inv;
    }
}

/// SwiGLU: down(silu(gate(x)) * up(x)).
/// `gate_out` and `up_out` must already hold gate(x) and up(x); this
/// function fuses the elementwise silu(gate) * up into `gate_out`.
pub fn swiglu_mul(gate_out: &mut [f32], up_out: &[f32]) {
    assert_eq!(gate_out.len(), up_out.len());
    for i in 0..gate_out.len() {
        gate_out[i] = silu(gate_out[i]) * up_out[i];
    }
}

/// Matrix-vector product following the GGUF storage layout.
/// `w` has ggml shape `[in_dim, out_dim]` (flat layout `w[j * in_dim + i]`),
/// so each output `y[j]` reads a contiguous row of length `in_dim`.
///   y[j] = sum_i x[i] * w[j * in_dim + i]
pub fn matvec(x: &[f32], w: &[f32], in_dim: usize, out_dim: usize, y: &mut [f32]) {
    assert_eq!(x.len(), in_dim);
    assert_eq!(w.len(), in_dim * out_dim);
    assert_eq!(y.len(), out_dim);
    for j in 0..out_dim {
        let row = &w[j * in_dim..(j + 1) * in_dim];
        let mut acc = 0.0_f32;
        for i in 0..in_dim {
            acc += x[i] * row[i];
        }
        y[j] = acc;
    }
}

/// Numerically stable softmax in-place along the entire slice.
pub fn softmax(x: &mut [f32]) {
    let mut max = f32::NEG_INFINITY;
    for &v in x.iter() {
        if v > max { max = v; }
    }
    let mut sum = 0.0_f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv = sum.recip();
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// Elementwise residual add, in place into `acc`.
pub fn add_(acc: &mut [f32], src: &[f32]) {
    assert_eq!(acc.len(), src.len());
    for i in 0..acc.len() {
        acc[i] += src[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol * (1.0 + b.abs())
    }

    #[test]
    fn silu_known_values() {
        // SiLU(0) = 0, SiLU(1) = 1 * sigmoid(1) = 1/(1+e^-1) ≈ 0.731059
        assert_eq!(silu(0.0), 0.0);
        assert!(approx_eq(silu(1.0), 0.7310585786, 1e-6));
        assert!(approx_eq(silu(-1.0), -0.2689414213, 1e-6));
    }

    #[test]
    fn sigmoid_basic() {
        assert!(approx_eq(sigmoid(0.0), 0.5, 1e-7));
        assert!(approx_eq(sigmoid(100.0), 1.0, 1e-6));
        assert!(approx_eq(sigmoid(-100.0), 0.0, 1e-6));
    }

    #[test]
    fn softplus_avoids_overflow() {
        assert!(approx_eq(softplus(0.0), (2.0_f32).ln(), 1e-7));
        // softplus(50) ≈ 50 (within 1e-22), and ln(1+exp(50)) would overflow
        // f32 → infinity; the > 20 fast path keeps it sane.
        assert!(softplus(50.0).is_finite());
        assert!(approx_eq(softplus(50.0), 50.0, 1e-6));
    }

    #[test]
    fn rmsnorm_with_unit_weight_returns_normalized_input() {
        // weight = 1 → output = x / rms(x).
        // x = [3, 4]: mean(x^2) = (9+16)/2 = 12.5, rms = sqrt(12.5) ≈ 3.5355
        let x = vec![3.0, 4.0];
        let w = vec![1.0, 1.0];
        let mut out = vec![0.0; 2];
        rmsnorm(&x, &w, 0.0, &mut out);
        let rms = 12.5_f32.sqrt();
        assert!(approx_eq(out[0], 3.0 / rms, 1e-6));
        assert!(approx_eq(out[1], 4.0 / rms, 1e-6));
    }

    #[test]
    fn rmsnorm_applies_per_dim_weight_directly() {
        // weight = [2, 0.5], x = [1, 1] → mean(x^2)=1, rms=1, x/rms*w = [2, 0.5]
        // (no `1 + w` shift — the convert step already baked +1 into the weight)
        let x = vec![1.0, 1.0];
        let w = vec![2.0, 0.5];
        let mut out = vec![0.0; 2];
        rmsnorm(&x, &w, 0.0, &mut out);
        assert!(approx_eq(out[0], 2.0, 1e-6));
        assert!(approx_eq(out[1], 0.5, 1e-6));
    }

    #[test]
    fn rmsnorm_gated_multiplies_silu_gate() {
        // weight = [1, 1], gate = [0, 1], x = [1, 1]
        // rmsnorm_no_shift = x/rms(x) * w = [1, 1] (since rms=1, w=1)
        // silu(gate) = [0, silu(1)] ≈ [0, 0.731]
        let x = vec![1.0, 1.0];
        let g = vec![0.0, 1.0];
        let w = vec![1.0, 1.0];
        let mut out = vec![0.0; 2];
        rmsnorm_gated(&x, &g, &w, 0.0, &mut out);
        assert!(approx_eq(out[0], 0.0, 1e-6));
        assert!(approx_eq(out[1], silu(1.0), 1e-6));
    }

    #[test]
    fn l2norm_unit_vector() {
        // x = [3, 4] → ||x|| = 5 → output [0.6, 0.8]
        let x = vec![3.0, 4.0];
        let mut out = vec![0.0; 2];
        l2norm(&x, 0.0, &mut out);
        assert!(approx_eq(out[0], 0.6, 1e-6));
        assert!(approx_eq(out[1], 0.8, 1e-6));
    }

    #[test]
    fn matvec_against_handcalculated() {
        // x = [1, 2, 3], W (ggml [in=3, out=2]) = [[a, b, c], [d, e, f]]
        //   flat layout (out fastest? no — in fastest, out slow)
        //   w[j * in + i] for j in 0..out, i in 0..in
        //   so w_flat = [a b c d e f] meaning row 0 = [a b c], row 1 = [d e f]
        // y[0] = a*1 + b*2 + c*3
        // y[1] = d*1 + e*2 + f*3
        let x = vec![1.0, 2.0, 3.0];
        let w = vec![1.0, 0.5, -1.0,   2.0, -0.5, 1.0];
        let mut y = vec![0.0; 2];
        matvec(&x, &w, 3, 2, &mut y);
        // y[0] = 1*1 + 0.5*2 + -1*3 = 1 + 1 - 3 = -1
        // y[1] = 2*1 + -0.5*2 + 1*3 = 2 - 1 + 3 = 4
        assert!(approx_eq(y[0], -1.0, 1e-6));
        assert!(approx_eq(y[1], 4.0, 1e-6));
    }

    #[test]
    fn matvec_identity() {
        // W = identity (in=out=3) → y = x.
        let x = vec![1.5, -2.0, 7.0];
        let w = vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ];
        let mut y = vec![0.0; 3];
        matvec(&x, &w, 3, 3, &mut y);
        assert_eq!(y, x);
    }

    #[test]
    fn softmax_normalizes_to_one() {
        let mut x = vec![1.0, 2.0, 3.0];
        softmax(&mut x);
        let sum: f32 = x.iter().sum();
        assert!(approx_eq(sum, 1.0, 1e-6));
        // monotonic: x[2] > x[1] > x[0]
        assert!(x[2] > x[1] && x[1] > x[0]);
    }

    #[test]
    fn softmax_handles_large_values() {
        // Without the max-subtract trick, exp(1000) would overflow.
        let mut x = vec![1000.0, 1000.0, 1000.0];
        softmax(&mut x);
        for v in &x {
            assert!(approx_eq(*v, 1.0 / 3.0, 1e-6));
        }
    }

    #[test]
    fn swiglu_mul_in_place() {
        // gate = [0, 1], up = [2, 3] → silu(gate)*up = [0, silu(1)*3]
        let mut gate = vec![0.0_f32, 1.0];
        let up = vec![2.0_f32, 3.0];
        swiglu_mul(&mut gate, &up);
        assert!(approx_eq(gate[0], 0.0, 1e-6));
        assert!(approx_eq(gate[1], silu(1.0) * 3.0, 1e-6));
    }
}
