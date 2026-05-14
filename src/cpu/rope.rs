//! Rotary Position Embedding cache (partial, half-split rotation).
//!
//! Qwen 3.5 0.8B uses `rope.dimension_count = 64` of `head_dim = 256` —
//! only the first 64 elements of each Q/K head are rotated, the remaining
//! 192 pass through unchanged. The rotation is the standard llama/HF
//! "half-split" form (`rotate_half` flips the two halves with sign change).
//!
//! M-RoPE (mrope_section [11, 11, 10]) is the multimodal extension for
//! vision tokens. For pure-text tokens all three position axes share the
//! same position id, so the M-RoPE freq table collapses to standard RoPE.
//! This module implements the text path only.

/// Precomputed cos/sin tables for positions 0..max_seq_len.
///
/// Layout: `cos[pos * rotary_dim + i]` for `i in 0..rotary_dim`. The table
/// duplicates each frequency value into both halves (`cos[i] == cos[i + half]`)
/// to keep the rotation kernel branch-free.
pub struct RopeCache {
    pub rotary_dim: usize,
    pub max_seq_len: usize,
    cos: Vec<f32>,
    sin: Vec<f32>,
}

impl RopeCache {
    pub fn new(rotary_dim: usize, max_seq_len: usize, freq_base: f32) -> Self {
        assert!(rotary_dim % 2 == 0, "rotary_dim must be even");
        let half = rotary_dim / 2;
        let mut inv_freq = vec![0.0_f32; half];
        for i in 0..half {
            inv_freq[i] = freq_base.powf(-2.0 * i as f32 / rotary_dim as f32);
        }
        let mut cos = vec![0.0_f32; max_seq_len * rotary_dim];
        let mut sin = vec![0.0_f32; max_seq_len * rotary_dim];
        for pos in 0..max_seq_len {
            for i in 0..half {
                let theta = pos as f32 * inv_freq[i];
                let c = theta.cos();
                let s = theta.sin();
                cos[pos * rotary_dim + i]        = c;
                cos[pos * rotary_dim + i + half] = c;
                sin[pos * rotary_dim + i]        = s;
                sin[pos * rotary_dim + i + half] = s;
            }
        }
        Self { rotary_dim, max_seq_len, cos, sin }
    }

    /// Slice the (cos, sin) row for a given position.
    pub fn get(&self, position: usize) -> (&[f32], &[f32]) {
        let off = position * self.rotary_dim;
        (
            &self.cos[off..off + self.rotary_dim],
            &self.sin[off..off + self.rotary_dim],
        )
    }
}

/// Apply RoPE in place to a single head.
///
/// `head` has length `head_dim`; the first `rope_cache.rotary_dim` elements
/// are rotated using the half-split convention:
///   half = rotary_dim / 2
///   y[i]        = x[i]        * cos[i]        - x[i + half] * sin[i]
///   y[i + half] = x[i + half] * cos[i + half] + x[i]        * sin[i + half]
/// Elements beyond `rotary_dim` are passed through unchanged.
pub fn apply_rope(head: &mut [f32], rope_cache: &RopeCache, position: usize) {
    let rd = rope_cache.rotary_dim;
    assert!(head.len() >= rd, "head_dim {} < rotary_dim {}", head.len(), rd);
    let half = rd / 2;
    let (cos, sin) = rope_cache.get(position);

    // Snapshot the rotated portion first (rotation reads both halves).
    let mut tmp = [0.0_f32; 256];
    let buf = &mut tmp[..rd];
    buf.copy_from_slice(&head[..rd]);

    for i in 0..half {
        head[i]        = buf[i]        * cos[i]        - buf[i + half] * sin[i];
        head[i + half] = buf[i + half] * cos[i + half] + buf[i]        * sin[i + half];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol * (1.0 + b.abs())
    }

    #[test]
    fn position_zero_is_identity() {
        // At position 0, cos = 1 and sin = 0, so rotation is a no-op.
        let cache = RopeCache::new(64, 16, 10000.0);
        let mut head = vec![0.0_f32; 256];
        for i in 0..256 { head[i] = (i as f32) * 0.01; }
        let original = head.clone();
        apply_rope(&mut head, &cache, 0);
        for i in 0..256 {
            assert!(approx_eq(head[i], original[i], 1e-6),
                "i={i} got {} expected {}", head[i], original[i]);
        }
    }

    #[test]
    fn pass_through_dims_unchanged() {
        // For partial RoPE (rotary_dim=64 of head_dim=256), elements >= 64
        // must be byte-equal to input regardless of position.
        let cache = RopeCache::new(64, 16, 10000.0);
        let mut head = vec![0.0_f32; 256];
        for i in 0..256 { head[i] = (i as f32 + 1.0).sqrt(); }
        let original = head.clone();
        apply_rope(&mut head, &cache, 5);
        for i in 64..256 {
            assert_eq!(head[i].to_bits(), original[i].to_bits(), "i={i}");
        }
    }

    #[test]
    fn rotation_at_position_one_matches_handcomputed() {
        // For freq_base=10000, rotary_dim=4:
        //   inv_freq[0] = 10000^0 = 1
        //   inv_freq[1] = 10000^(-2/4) = 1 / sqrt(10000) = 0.01
        // At position 1:
        //   theta_0 = 1 * 1 = 1 → cos=cos(1), sin=sin(1)
        //   theta_1 = 1 * 0.01 = 0.01 → cos≈cos(0.01), sin≈sin(0.01)
        // For input x = [a, b, c, d]:
        //   y[0] = a*cos(1) - c*sin(1)
        //   y[1] = b*cos(0.01) - d*sin(0.01)
        //   y[2] = c*cos(1) + a*sin(1)
        //   y[3] = d*cos(0.01) + b*sin(0.01)
        let cache = RopeCache::new(4, 4, 10000.0);
        let mut head = vec![1.0_f32, 2.0, 3.0, 4.0];
        apply_rope(&mut head, &cache, 1);

        let c0 = (1.0_f32).cos();
        let s0 = (1.0_f32).sin();
        let c1 = (0.01_f32).cos();
        let s1 = (0.01_f32).sin();
        assert!(approx_eq(head[0], 1.0 * c0 - 3.0 * s0, 1e-6));
        assert!(approx_eq(head[1], 2.0 * c1 - 4.0 * s1, 1e-6));
        assert!(approx_eq(head[2], 3.0 * c0 + 1.0 * s0, 1e-6));
        assert!(approx_eq(head[3], 4.0 * c1 + 2.0 * s1, 1e-6));
    }

    #[test]
    fn rotation_preserves_norm() {
        // Rotation is a unitary transformation → ||x||² preserved on the
        // rotated portion. Use head_dim = rotary_dim so we measure the whole vector.
        let cache = RopeCache::new(8, 32, 10000.0);
        let head_init: Vec<f32> = (0..8).map(|i| (i as f32 + 1.0) * 0.5).collect();
        let norm_in: f32 = head_init.iter().map(|v| v * v).sum::<f32>().sqrt();
        for pos in 0..32 {
            let mut head = head_init.clone();
            apply_rope(&mut head, &cache, pos);
            let norm_out: f32 = head.iter().map(|v| v * v).sum::<f32>().sqrt();
            assert!(approx_eq(norm_in, norm_out, 1e-6),
                "pos={pos}: norm changed {} → {}", norm_in, norm_out);
        }
    }
}
