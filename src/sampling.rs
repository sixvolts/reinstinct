//! Token samplers — turn a vocab-length logit vector into a token id.
//!
//! Three samplers cover most use cases:
//! - `argmax`              : pick the highest-logit token (greedy)
//! - `sample_temp_topk`    : top-k filter then softmax-with-temperature, then PRNG
//! - `Rng`                 : tiny xorshift state shared across calls
//!
//! Sampling lives entirely on the host: logits are already D2H'd by
//! `forward_token`, and the work is O(vocab) — fast enough to ignore.

/// Tiny deterministic PRNG so generations are reproducible from a seed.
pub struct Rng { state: u64 }

impl Rng {
    pub fn new(seed: u64) -> Self {
        // Use a non-zero state. xorshift on zero stays zero.
        Self { state: if seed == 0 { 0x9E3779B97F4A7C15 } else { seed } }
    }

    /// Advance and return a u64.
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Uniform f32 in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        // 24-bit mantissa is enough for this purpose.
        let bits = (self.next_u64() >> 40) as u32;
        bits as f32 / (1u32 << 24) as f32
    }
}

/// Greedy: index of the maximum logit. Ties broken by lowest index.
/// NaN entries are ignored (so a degenerate model output that produced
/// NaNs in a subset of vocab positions still returns the best of the
/// finite entries instead of poisoning the comparison).
pub fn argmax(logits: &[f32]) -> u32 {
    assert!(!logits.is_empty(), "argmax called with empty logits");
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v.is_finite() && v > best_v { best_v = v; best = i; }
    }
    best as u32
}

/// Top-k + temperature sampling.
///
///   1. find the k largest logits
///   2. divide by `temperature` (clamped at 1e-3 to avoid divide-by-zero)
///   3. softmax over those k
///   4. PRNG draw against the resulting distribution
///
/// `k = 0` means "no filter" — softmax + sample over the full vocab.
/// `temperature = 0.0` reduces to greedy regardless of k.
pub fn sample_temp_topk(logits: &[f32], temperature: f32, k: usize, rng: &mut Rng) -> u32 {
    assert!(!logits.is_empty(), "sample_temp_topk called with empty logits");
    if temperature <= 0.0 {
        return argmax(logits);
    }

    let n = logits.len();
    let k_eff = if k == 0 || k > n { n } else { k };

    // Partial selection of top-k. Non-finite (NaN/±Inf) logits are
    // skipped — they'd otherwise corrupt the softmax sum into NaN and
    // produce a garbage token. If the model ever emits all-NaN logits
    // (a real failure mode on numerical blowup), `top` ends up empty
    // and we fall back to argmax (which has its own NaN guard).
    let mut top: Vec<(usize, f32)> = Vec::with_capacity(k_eff);
    for (i, &v) in logits.iter().enumerate() {
        if !v.is_finite() { continue; }
        if top.len() < k_eff {
            top.push((i, v));
            top.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        } else if v > top[0].1 {
            top[0] = (i, v);
            top.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        }
    }
    if top.is_empty() { return argmax(logits); }

    // Softmax with temperature over the surviving k.
    let inv_t = 1.0 / temperature.max(1e-3);
    let max_logit = top.iter().map(|(_, v)| *v).fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for (_, v) in top.iter_mut() {
        *v = ((*v - max_logit) * inv_t).exp();
        sum += *v;
    }
    // sum can collapse to 0 if every weight underflowed (extreme temperature
    // or pathological logits) — fall back to the most-probable.
    if !(sum > 0.0) {
        return top.last().expect("top non-empty above").0 as u32;
    }
    let r = rng.next_f32() * sum;
    let mut cum = 0.0_f32;
    for (idx, w) in &top {
        cum += *w;
        if r <= cum { return *idx as u32; }
    }
    // Fall through (numerical edge): return the most-probable.
    top.last().expect("top non-empty above").0 as u32
}

/// Standard temperature-softmax over the full vocab. Subtract max for
/// stability, exp, normalize. `temperature` must be > 0 (caller checks).
/// Used by the MTP spec-decode acceptance loop, which needs full
/// distributions for the rejection-sampling test.
pub fn softmax_with_temp(logits: &[f32], temperature: f32) -> Vec<f32> {
    let inv_t = 1.0 / temperature;
    let mut max_v = f32::NEG_INFINITY;
    for &x in logits { if x > max_v { max_v = x; } }
    let mut out: Vec<f32> = logits.iter()
        .map(|&x| ((x - max_v) * inv_t).exp())
        .collect();
    let s: f32 = out.iter().sum();
    if s > 0.0 { for x in &mut out { *x /= s; } }
    out
}

/// Sample a token id from a logits vector via temperature-softmax over
/// the full vocab. Companion to `softmax_with_temp` for callers that
/// need the distribution.
pub fn sample_from_logits(logits: &[f32], temperature: f32, rng: &mut Rng) -> u32 {
    let p = softmax_with_temp(logits, temperature);
    sample_from_probs(&p, rng)
}

/// Sample from a vector of probabilities (assumed to sum ≈ 1.0). Linear
/// scan; fine for the small distributions used in spec-decode acceptance.
pub fn sample_from_probs(probs: &[f32], rng: &mut Rng) -> u32 {
    let r = rng.next_f32();
    let mut acc = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        acc += p;
        if r < acc { return i as u32; }
    }
    (probs.len() - 1) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_picks_largest() {
        let logits = vec![1.0, 3.0, 2.0, 5.0, 4.0];
        assert_eq!(argmax(&logits), 3);
    }

    #[test]
    fn temperature_zero_is_greedy() {
        let logits = vec![1.0, 3.0, 2.0, 5.0, 4.0];
        let mut rng = Rng::new(42);
        for _ in 0..10 {
            assert_eq!(sample_temp_topk(&logits, 0.0, 5, &mut rng), 3);
        }
    }

    #[test]
    fn topk_one_is_argmax() {
        let logits = vec![1.0, 3.0, 2.0, 5.0, 4.0];
        let mut rng = Rng::new(42);
        for _ in 0..10 {
            assert_eq!(sample_temp_topk(&logits, 1.0, 1, &mut rng), 3);
        }
    }

    #[test]
    fn topk_full_distribution_is_proportional() {
        // logits = [log(1), log(2), log(3), log(4)] → softmax weights
        // proportional to [1, 2, 3, 4]. With many samples the empirical
        // distribution should look right.
        let logits: Vec<f32> = (1..=4).map(|x| (x as f32).ln()).collect();
        let mut rng = Rng::new(0xC0FFEE);
        let mut counts = [0usize; 4];
        let n = 100_000;
        for _ in 0..n {
            let t = sample_temp_topk(&logits, 1.0, 0, &mut rng);
            counts[t as usize] += 1;
        }
        // Expected proportions [0.1, 0.2, 0.3, 0.4]; allow ±0.02.
        for (i, &c) in counts.iter().enumerate() {
            let p = c as f32 / n as f32;
            let expected = (i + 1) as f32 / 10.0;
            let err = (p - expected).abs();
            assert!(err < 0.02, "token {i}: empirical {p:.4} vs expected {expected:.4}");
        }
    }

    #[test]
    fn rng_is_deterministic_per_seed() {
        let mut a = Rng::new(7);
        let mut b = Rng::new(7);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }
}
