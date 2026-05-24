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
        // Strict inequality so an underflowed (w == 0) entry can't
        // pre-empt the real winner when r itself is at the boundary.
        // This matters for collapsed-softmax cases where 9 of 10 top-k
        // weights underflow to 0 and the one survivor at the END is
        // the actual winner: with `r <= cum`, r=0 would return the
        // first zero-weight entry instead of waiting for the survivor.
        if r < cum { return *idx as u32; }
    }
    // Fall through (numerical edge): return the most-probable.
    top.last().expect("top non-empty above").0 as u32
}

// --------------- Sampler chain (OpenAI-style request parameters) -------------
//
// Modern serving APIs expose several samplers at once. The canonical chain
// order (matching llama.cpp + OpenAI):
//   1. Penalties (repetition / frequency / presence) — modify logits using
//      the recent token history.
//   2. Top-k filter — keep only the K largest logits.
//   3. Top-p filter (nucleus) — keep enough logits to cover P of the mass.
//   4. Min-p filter — drop logits below `min_p * max_prob`.
//   5. Temperature scale.
//   6. Sample via inverse-CDF.
//
// Mirostat v2 is a separate sampler that replaces top-k/top-p/temperature with
// adaptive entropy control; chain config disables the other filters when on.

/// Knobs for one sampling call. Defaults pass the logits through unchanged
/// except for temperature (>0 enables sampling; ==0 falls back to greedy).
#[derive(Clone, Debug)]
pub struct SamplerParams {
    pub temperature: f32,
    pub top_k: usize,         // 0 ⇒ no filter
    pub top_p: f32,           // 1.0 ⇒ no filter
    pub min_p: f32,           // 0.0 ⇒ no filter
    pub repetition_penalty: f32,        // 1.0 ⇒ no penalty
    pub repetition_window: usize,       // tokens of history to consider
    pub frequency_penalty: f32,         // 0.0 ⇒ no penalty (OpenAI-style)
    pub presence_penalty:  f32,         // 0.0 ⇒ no penalty (OpenAI-style)
    pub mirostat: Option<MirostatV2>,   // Some ⇒ replaces top_k/top_p/temp
    pub seed: u64,
}

impl Default for SamplerParams {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_k: 40,
            top_p: 1.0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            repetition_window: 64,
            frequency_penalty: 0.0,
            presence_penalty:  0.0,
            mirostat: None,
            seed: 0,
        }
    }
}

/// Mirostat v2 — adaptive sampling targeting a fixed surprise (`tau`,
/// natural-log). `eta` is the learning rate on `mu` (the current threshold).
#[derive(Clone, Debug)]
pub struct MirostatV2 {
    pub tau: f32,
    pub eta: f32,
    /// Persistent state across calls in a generation: the current
    /// dynamic threshold. Initialised to `2.0 * tau` per the original
    /// paper; updated on each accept.
    pub mu: f32,
}

impl MirostatV2 {
    pub fn new(tau: f32, eta: f32) -> Self {
        Self { tau, eta, mu: 2.0 * tau }
    }
}

/// Apply OpenAI-style frequency + presence penalties in-place:
///   logits[t] -= count(t) * freq_penalty + (count(t) > 0) * presence_penalty
/// `counts` is `t -> usage count in the prior decoded tokens`.
pub fn apply_freq_presence_penalty(logits: &mut [f32], counts: &[u16],
                                    frequency: f32, presence: f32)
{
    if frequency == 0.0 && presence == 0.0 { return; }
    assert_eq!(logits.len(), counts.len(),
        "apply_freq_presence_penalty: counts must be vocab-length");
    for (i, &c) in counts.iter().enumerate() {
        if c == 0 { continue; }
        logits[i] -= c as f32 * frequency;
        if presence != 0.0 { logits[i] -= presence; }
    }
}

/// Apply a multiplicative repetition penalty over the last `window` tokens
/// of `history`. For each token id seen, divide its logit by `penalty` (if
/// positive) or multiply by it (if negative — the conventional negative-logit
/// handling). `penalty == 1.0` is a no-op.
pub fn apply_repetition_penalty(logits: &mut [f32], history: &[u32],
                                 window: usize, penalty: f32)
{
    if penalty == 1.0 || window == 0 || history.is_empty() { return; }
    let start = history.len().saturating_sub(window);
    for &tok in &history[start..] {
        let i = tok as usize;
        if i >= logits.len() { continue; }
        let v = logits[i];
        logits[i] = if v > 0.0 { v / penalty } else { v * penalty };
    }
}

/// Replace logits NOT in the top-k with -inf so subsequent softmax drops them.
/// In-place. `k == 0` or `k >= vocab` is a no-op.
pub fn apply_top_k(logits: &mut [f32], k: usize) {
    if k == 0 || k >= logits.len() { return; }
    // Partial selection: find the k-th largest value, mask everything below.
    let mut copy: Vec<f32> = logits.iter().copied()
        .filter(|v| v.is_finite()).collect();
    if copy.len() <= k { return; }
    // nth-element ordering: kth from the top.
    let kth_idx = copy.len() - k;
    copy.select_nth_unstable_by(kth_idx, |a, b|
        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let threshold = copy[kth_idx];
    for v in logits.iter_mut() {
        if !v.is_finite() || *v < threshold { *v = f32::NEG_INFINITY; }
    }
}

/// Nucleus (top-p) filter: sort descending probabilities, keep the smallest
/// prefix that covers `p` of the mass; mask the rest. `p >= 1.0` is a no-op.
/// Operates on logits in-place; computes a temporary softmax internally.
pub fn apply_top_p(logits: &mut [f32], p: f32) {
    if !(p > 0.0 && p < 1.0) { return; }
    // Softmax probabilities (stable).
    let max = logits.iter().copied()
        .filter(|v| v.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() { return; }
    let mut probs: Vec<(usize, f32)> = logits.iter().enumerate()
        .map(|(i, &v)| (i, if v.is_finite() { (v - max).exp() } else { 0.0 }))
        .collect();
    let sum: f32 = probs.iter().map(|(_, w)| w).sum();
    if !(sum > 0.0) { return; }
    for (_, w) in probs.iter_mut() { *w /= sum; }
    // Sort descending by prob.
    probs.sort_unstable_by(|a, b|
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    // Walk the cumulative; mask anything past the cutoff (but always keep
    // at least the top-1, even if its prob alone covers p).
    let mut cum = 0.0_f32;
    let mut cutoff_rank = probs.len();
    for (rank, (_, w)) in probs.iter().enumerate() {
        cum += *w;
        if cum >= p { cutoff_rank = rank + 1; break; }
    }
    for (_, (idx, _)) in probs.iter().enumerate().skip(cutoff_rank) {
        logits[*idx] = f32::NEG_INFINITY;
    }
}

/// Min-p filter: drop any logit whose post-softmax prob is below
/// `min_p * max_prob`. `min_p == 0.0` is a no-op.
pub fn apply_min_p(logits: &mut [f32], min_p: f32) {
    if !(min_p > 0.0) { return; }
    let max = logits.iter().copied()
        .filter(|v| v.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() { return; }
    // log(min_p * max_prob) = log(min_p) + max_logit (after the -max shift the
    // max becomes 0, and any logit `v` with `(v-max) < log(min_p)` falls below
    // the threshold). Avoids computing a full softmax just to filter.
    let log_min = min_p.ln();
    for v in logits.iter_mut() {
        if !v.is_finite() || (*v - max) < log_min { *v = f32::NEG_INFINITY; }
    }
}

/// End-of-chain sampler: temperature scale + softmax + draw via inverse-CDF.
/// Assumes the caller's filters have already masked anything not in
/// contention (those entries are `-inf` and contribute 0 to the softmax).
/// `temperature == 0.0` falls back to argmax (works correctly with `-inf`
/// masked entries via the existing NaN/Inf-skip in `argmax`).
pub fn sample_softmax_temp(logits: &[f32], temperature: f32, rng: &mut Rng) -> u32 {
    if temperature <= 0.0 { return argmax(logits); }
    let inv_t = 1.0 / temperature.max(1e-3);
    let max = logits.iter().copied()
        .filter(|v| v.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() { return argmax(logits); }
    let mut weights: Vec<(usize, f32)> = logits.iter().enumerate()
        .map(|(i, &v)| (i, if v.is_finite() { ((v - max) * inv_t).exp() } else { 0.0 }))
        .collect();
    let sum: f32 = weights.iter().map(|(_, w)| *w).sum();
    if !(sum > 0.0) { return argmax(logits); }
    let r = rng.next_f32() * sum;
    let mut cum = 0.0_f32;
    for (idx, w) in &weights {
        cum += *w;
        if r < cum { return *idx as u32; }
    }
    // Fall through: return the maximum-weight entry (sorted scan would have
    // returned this naturally).
    weights.sort_unstable_by(|a, b|
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    weights.first().map(|(i, _)| *i as u32).unwrap_or(0)
}

/// Mirostat v2 sampling. Sort logits, compute the smallest prefix whose
/// cumulative -log(p) exceeds `mu`, sample uniformly from it, then update
/// `mu -= eta * (observed_surprise - tau)`.
pub fn sample_mirostat_v2(logits: &[f32], state: &mut MirostatV2, rng: &mut Rng) -> u32 {
    // Stable softmax.
    let max = logits.iter().copied()
        .filter(|v| v.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() { return argmax(logits); }
    let mut probs: Vec<(usize, f32)> = logits.iter().enumerate()
        .map(|(i, &v)| (i, if v.is_finite() { (v - max).exp() } else { 0.0 }))
        .collect();
    let s: f32 = probs.iter().map(|(_, w)| *w).sum();
    if !(s > 0.0) { return argmax(logits); }
    for (_, w) in probs.iter_mut() { *w /= s; }
    probs.sort_unstable_by(|a, b|
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Keep the smallest prefix whose total surprise (-log p) ≤ mu.
    // Equivalently, drop entries past the first one that pushes the
    // cumulative surprise above mu.
    let mut keep_until = probs.len();
    for (rank, (_, p)) in probs.iter().enumerate() {
        let s = -p.max(1e-30).ln();
        if s > state.mu { keep_until = rank.max(1); break; }
    }
    let truncated = &probs[..keep_until];
    // Renormalize and sample.
    let total: f32 = truncated.iter().map(|(_, w)| *w).sum();
    let r = rng.next_f32() * total;
    let mut cum = 0.0_f32;
    let mut picked: u32 = truncated.last().map(|(i, _)| *i as u32).unwrap_or(0);
    let mut picked_p = truncated.last().map(|(_, p)| *p).unwrap_or(1e-30);
    for (idx, p) in truncated {
        cum += *p;
        if r < cum { picked = *idx as u32; picked_p = *p; break; }
    }
    // Mirostat update: nudge mu by the surprise error.
    let observed = -picked_p.max(1e-30).ln();
    state.mu -= state.eta * (observed - state.tau);
    picked
}

/// Run the full sampling chain. `history` is the sequence of tokens already
/// emitted (used by penalty samplers). `counts` is the per-token usage count
/// across the same history — supply an empty slice to skip OpenAI penalties.
/// Returns the sampled token id.
pub fn sample_chain(logits: &mut [f32], params: &mut SamplerParams,
                    history: &[u32], counts: &[u16], rng: &mut Rng) -> u32
{
    sample_chain_lp(logits, params, history, counts, rng, 0).token
}

/// One token's worth of sampling output, with optional log-probability
/// diagnostics for the OpenAI `logprobs:true` response field.
#[derive(Clone, Debug)]
pub struct SampleResult {
    pub token: u32,
    /// log P(token) under the POST-filter, POST-temperature distribution.
    /// `None` when `top_logprobs_n == 0` (caller didn't ask for diagnostics).
    pub logprob: Option<f32>,
    /// Top-K alternatives by probability, including the chosen token,
    /// each tagged with their log-probability. Length ≤ `top_logprobs_n`,
    /// sorted descending by logprob. Empty when not requested.
    pub top_logprobs: Vec<(u32, f32)>,
}

/// Like [`sample_chain`] but also returns the chosen-token logprob and
/// the top-`top_logprobs_n` alternatives. `top_logprobs_n == 0` skips
/// the diagnostic work (no extra allocation, equivalent to plain
/// `sample_chain` cost).
///
/// All probabilities are reported on the POST-filter, POST-temperature
/// distribution — i.e. they match what was actually sampled from, not
/// the raw model logits. This matches what OpenAI returns: the user
/// can reproduce the sampling distribution from these numbers.
///
/// For Mirostat the truncation is dynamic per-step; `top_logprobs`
/// reports the top-K of the renormalized truncated distribution.
pub fn sample_chain_lp(logits: &mut [f32], params: &mut SamplerParams,
                       history: &[u32], counts: &[u16], rng: &mut Rng,
                       top_logprobs_n: usize) -> SampleResult
{
    assert!(!logits.is_empty(), "sample_chain called with empty logits");
    // Mirostat path is mutually exclusive with the top-k/top-p/temperature
    // chain — those filters would interfere with mu's update.
    if let Some(ms) = &mut params.mirostat {
        apply_freq_presence_penalty(logits, counts,
            params.frequency_penalty, params.presence_penalty);
        apply_repetition_penalty(logits, history,
            params.repetition_window, params.repetition_penalty);
        let token = sample_mirostat_v2(logits, ms, rng);
        return finalize_lp(logits, token, top_logprobs_n, 1.0);
    }
    apply_freq_presence_penalty(logits, counts,
        params.frequency_penalty, params.presence_penalty);
    apply_repetition_penalty(logits, history,
        params.repetition_window, params.repetition_penalty);
    apply_top_k(logits, params.top_k);
    apply_top_p(logits, params.top_p);
    apply_min_p(logits, params.min_p);
    let token = sample_softmax_temp(logits, params.temperature, rng);
    finalize_lp(logits, token, top_logprobs_n, params.temperature.max(1e-3))
}

/// Compute logprob of `picked` + top-K alternatives over the (already
/// filtered) logits. Skips the work entirely when `top_n == 0`.
fn finalize_lp(logits: &[f32], picked: u32, top_n: usize, temperature: f32)
    -> SampleResult
{
    if top_n == 0 {
        return SampleResult { token: picked, logprob: None, top_logprobs: vec![] };
    }
    let inv_t = 1.0 / temperature;
    let max = logits.iter().copied()
        .filter(|v| v.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return SampleResult { token: picked, logprob: Some(f32::NEG_INFINITY),
                              top_logprobs: vec![] };
    }
    // Stable softmax. Track (idx, logp) pairs — log so picked token's
    // value comes back as log P, not P.
    let mut logps: Vec<(u32, f32)> = Vec::with_capacity(logits.len());
    let mut sumexp = 0.0_f32;
    let mut scaled: Vec<f32> = Vec::with_capacity(logits.len());
    for &v in logits {
        let s = if v.is_finite() { (v - max) * inv_t } else { f32::NEG_INFINITY };
        scaled.push(s);
        if s.is_finite() { sumexp += s.exp(); }
    }
    let lse = sumexp.max(1e-30).ln();
    for (i, &s) in scaled.iter().enumerate() {
        logps.push((i as u32, s - lse));
    }
    let picked_lp = logps[picked as usize].1;
    // Partial sort: top-N alternatives by logp.
    let cap = top_n.min(logps.len());
    let mut top = logps;
    top.select_nth_unstable_by(cap - 1, |a, b|
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    top.truncate(cap);
    top.sort_unstable_by(|a, b|
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    SampleResult { token: picked, logprob: Some(picked_lp), top_logprobs: top }
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
    fn sample_chain_lp_matches_softmax() {
        // Logits = [log 1, log 2, log 3, log 4], temp 1, no filters →
        // post-softmax probs = [0.1, 0.2, 0.3, 0.4], logprobs ≈
        // ln of those values.
        let mut logits: Vec<f32> = (1..=4).map(|x| (x as f32).ln()).collect();
        let mut params = SamplerParams {
            temperature: 1.0, top_k: 0, top_p: 1.0, min_p: 0.0,
            repetition_penalty: 1.0, repetition_window: 0,
            frequency_penalty: 0.0, presence_penalty: 0.0,
            mirostat: None, seed: 0,
        };
        let mut rng = Rng::new(123);
        let r = sample_chain_lp(&mut logits, &mut params, &[], &[], &mut rng, 4);
        let tl = r.top_logprobs;
        assert_eq!(tl.len(), 4);
        // Top entry should be token id 3 (largest logit).
        assert_eq!(tl[0].0, 3);
        // logprob of token 3 ≈ ln(0.4) ≈ -0.916
        assert!((tl[0].1 - 0.4f32.ln()).abs() < 1e-4, "{:?}", tl[0]);
        // Picked token's logprob should equal whatever it's reported as.
        let picked_lp = tl.iter().find(|(i, _)| *i == r.token).unwrap().1;
        assert!((r.logprob.unwrap() - picked_lp).abs() < 1e-5);
    }

    #[test]
    fn sample_chain_lp_zero_n_skips_diagnostics() {
        let mut logits = vec![1.0, 3.0, 2.0, 5.0, 4.0];
        let mut params = SamplerParams::default();
        params.temperature = 0.0; // greedy
        let mut rng = Rng::new(7);
        let r = sample_chain_lp(&mut logits, &mut params, &[], &[], &mut rng, 0);
        assert_eq!(r.token, 3);
        assert!(r.logprob.is_none());
        assert!(r.top_logprobs.is_empty());
    }

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

    #[test]
    fn argmax_skips_nan() {
        // NaN entries shouldn't poison the comparison — the max of the
        // finite entries wins.
        let logits = vec![1.0, f32::NAN, 3.0, f32::NAN, 2.0];
        assert_eq!(argmax(&logits), 2);
    }

    #[test]
    fn argmax_skips_inf() {
        // Positive infinity dominates everything (still finite-checked
        // via is_finite, which excludes +inf), so the finite max wins.
        let logits = vec![1.0, f32::INFINITY, 3.0, 2.0];
        assert_eq!(argmax(&logits), 2);
    }

    #[test]
    fn argmax_all_nan_returns_first() {
        // Degenerate input — every entry is NaN. argmax falls through
        // to the initial `best = 0` so we return a deterministic
        // sentinel instead of panicking.
        let logits = vec![f32::NAN; 5];
        assert_eq!(argmax(&logits), 0);
    }

    #[test]
    fn sample_topk_handles_nan_logits() {
        // A few NaNs in the vocab shouldn't crash sampling. The
        // surviving finite entries should drive the distribution.
        let mut logits = vec![f32::NAN; 100];
        logits[42] = 10.0;
        logits[77] = 2.0;
        let mut rng = Rng::new(0xDEADBEEF);
        for _ in 0..50 {
            let t = sample_temp_topk(&logits, 1.0, 5, &mut rng);
            assert!(t == 42 || t == 77,
                "sampled garbage index {t}; should have been 42 or 77");
        }
    }

    #[test]
    fn sample_topk_all_nan_returns_deterministic() {
        // All NaN → falls back to argmax → deterministic index. The
        // request will produce garbage output but at least the worker
        // stays alive (serve hardening rationale).
        let logits = vec![f32::NAN; 10];
        let mut rng = Rng::new(1);
        let t = sample_temp_topk(&logits, 1.0, 5, &mut rng);
        assert_eq!(t, 0);
    }

    #[test]
    fn top_p_keeps_smallest_prefix_to_p() {
        // Probabilities proportional to [4, 3, 2, 1] (sum=10).
        // Cumulative: 0.4, 0.7, 0.9, 1.0. With p=0.6, the first entry
        // (0.4) doesn't cover yet; the second pushes to 0.7 ≥ 0.6, so
        // we keep the first 2 entries. Indices: 3 (val 4.0), 2 (val 3.0).
        let mut logits: Vec<f32> = (1..=4).map(|v| (v as f32).ln()).collect();
        apply_top_p(&mut logits, 0.6);
        // Entries 0 (1.0 → ln=0) and 1 (2.0 → ln=0.69) should be -inf.
        assert!(logits[0].is_infinite() && logits[0].is_sign_negative());
        assert!(logits[1].is_infinite() && logits[1].is_sign_negative());
        // Entries 2 (val 3.0) and 3 (val 4.0) survive.
        assert!(logits[2].is_finite());
        assert!(logits[3].is_finite());
    }

    #[test]
    fn top_p_keeps_at_least_one() {
        // Even when the top entry's prob > p, top-1 must survive.
        let mut logits = vec![10.0_f32, 0.0, 0.0, 0.0];
        apply_top_p(&mut logits, 0.1);
        assert!(logits[0].is_finite(),
            "top-1 must survive even when its prob alone exceeds p");
    }

    #[test]
    fn min_p_drops_below_floor() {
        // logits [10, 0, -10] → after softmax-max-shift [-0, -10, -20].
        // log(0.5) ≈ -0.693. With min_p=0.5: any logit-shift < -0.693
        // gets masked. So 0 (the max) survives, -10 and -20 don't.
        let mut logits = vec![10.0_f32, 0.0, -10.0];
        apply_min_p(&mut logits, 0.5);
        assert!(logits[0].is_finite());
        assert!(logits[1].is_infinite());
        assert!(logits[2].is_infinite());
    }

    #[test]
    fn freq_presence_penalty_shifts_logits() {
        // counts = [0, 2, 0, 5], freq=0.1, presence=0.5
        // Expected shift on i=1: -(2*0.1 + 0.5) = -0.7
        // Expected shift on i=3: -(5*0.1 + 0.5) = -1.0
        let mut logits = vec![1.0_f32; 4];
        let counts = vec![0u16, 2, 0, 5];
        apply_freq_presence_penalty(&mut logits, &counts, 0.1, 0.5);
        assert!((logits[0] - 1.0).abs() < 1e-6);
        assert!((logits[1] - 0.3).abs() < 1e-6);
        assert!((logits[2] - 1.0).abs() < 1e-6);
        assert!((logits[3] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn repetition_penalty_divides_positive_logits() {
        // history = [0, 2], penalty=2.0, window large enough.
        // logits[0] = 4.0 (positive) → 4.0 / 2 = 2.0
        // logits[2] = -3.0 (negative) → -3.0 * 2 = -6.0
        let mut logits = vec![4.0_f32, 1.0, -3.0, 5.0];
        apply_repetition_penalty(&mut logits, &[0u32, 2u32], 64, 2.0);
        assert!((logits[0] - 2.0).abs() < 1e-6);
        assert!((logits[1] - 1.0).abs() < 1e-6);
        assert!((logits[2] - (-6.0)).abs() < 1e-6);
        assert!((logits[3] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn repetition_penalty_respects_window() {
        // window=1, history=[0, 1, 2]: only token 2 in scope.
        let mut logits = vec![4.0_f32; 3];
        apply_repetition_penalty(&mut logits, &[0u32, 1u32, 2u32], 1, 2.0);
        assert!((logits[0] - 4.0).abs() < 1e-6);
        assert!((logits[1] - 4.0).abs() < 1e-6);
        assert!((logits[2] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn chain_with_all_filters_returns_valid_token() {
        // Build a vocab, apply every filter, sample many times. The
        // outputs should never panic and always be a valid index.
        let vocab = 100;
        let mut params = SamplerParams::default();
        params.top_k = 10; params.top_p = 0.95; params.min_p = 0.05;
        params.repetition_penalty = 1.1; params.frequency_penalty = 0.1;
        params.presence_penalty = 0.05;
        let history = vec![5u32, 7u32, 5u32, 12u32];
        let mut counts = vec![0u16; vocab];
        for &t in &history { counts[t as usize] = counts[t as usize].saturating_add(1); }
        let mut rng = Rng::new(42);
        for _ in 0..200 {
            let mut logits: Vec<f32> = (0..vocab).map(|i| (i as f32) * 0.1).collect();
            let t = sample_chain(&mut logits, &mut params, &history, &counts, &mut rng);
            assert!((t as usize) < vocab);
        }
    }

    #[test]
    fn mirostat_v2_stays_bounded() {
        // mu should remain bounded (the controller is stable). Exact
        // convergence depends on the prob distribution + tau; we just
        // check that mu doesn't run away to a wild value either direction.
        // Distribution: probs ∝ [1..50] — surprise range ≈ [3.24, 7.15].
        let logits: Vec<f32> = (1..=50).map(|v| (v as f32).ln()).collect();
        let mut state = MirostatV2::new(/*tau=*/ 5.0, /*eta=*/ 0.1);
        let mut rng = Rng::new(123);
        for _ in 0..200 {
            let mut lg = logits.clone();
            let _ = sample_mirostat_v2(&mut lg, &mut state, &mut rng);
            // Should never NaN / Inf and should stay in a reasonable band.
            assert!(state.mu.is_finite(), "mirostat mu diverged to {}", state.mu);
            assert!(state.mu.abs() < 100.0, "mirostat mu unbounded: {}", state.mu);
        }
    }

    #[test]
    fn mirostat_v2_samples_valid_indices() {
        let logits: Vec<f32> = (1..=100).map(|v| (v as f32).ln()).collect();
        let mut state = MirostatV2::new(5.0, 0.1);
        let mut rng = Rng::new(1);
        for _ in 0..50 {
            let mut lg = logits.clone();
            let t = sample_mirostat_v2(&mut lg, &mut state, &mut rng);
            assert!((t as usize) < logits.len(),
                "mirostat returned out-of-range index {t}");
        }
    }

    #[test]
    fn sample_topk_handles_collapsed_softmax() {
        // Logit spread so wide that all but one weight underflows to 0
        // at the chosen temperature. Should return the dominant token
        // instead of the fallthrough panic that the pre-hardened code
        // would hit via `top.last().unwrap()` on an empty cum-sum.
        let mut logits = vec![-1000.0_f32; 50];
        logits[7] = 100.0;
        let mut rng = Rng::new(1);
        for _ in 0..10 {
            assert_eq!(sample_temp_topk(&logits, 0.01, 10, &mut rng), 7);
        }
    }
}
