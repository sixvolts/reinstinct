//! Gemma 4 MTP spec-decode loop, shared between the `mtp-gen` CLI and
//! the serve worker. Drafter+verifier with greedy or rejection-sampling
//! acceptance, K drafts per round, optional bonus token on a full
//! accept. See [[gemma4-mtp]] in memory for the full architecture.
//!
//! Entry point: [`spec_decode_generate`]. Callers provide a loaded
//! target (`GpuGemma4`), an MTP drafter (`GpuGemma4Assistant`), a fresh
//! decode state, the encoded prompt, and per-request sampling params.
//! The function does the initial prefill + verify-graph capture, runs
//! the spec-decode loop until `max_tokens` accepted or EOS, and returns
//! the generated token ids + per-call stats.
//!
//! NOT thread-safe: callers must serialise concurrent calls against
//! the same target/drafter pair (they share kernel modules + scratch).

use crate::hip::GraphExec;
use crate::runtime::gemma4::{GpuGemma4, Gemma4GpuState};
use crate::runtime::gemma4_assistant::GpuGemma4Assistant;
use crate::sampling::{
    argmax, sample_from_logits, sample_from_probs, softmax_with_temp, Rng,
};

/// Per-call stats returned alongside the generated tokens. Used by the
/// CLI for the "X / Y = Z%" accept-rate report and by the serve worker
/// for the response-stats field.
#[derive(Debug, Default, Clone, Copy)]
pub struct SpecStats {
    pub n_drafted:  usize,
    pub n_accepted: usize,
    pub hit_eos:    bool,
}

impl SpecStats {
    pub fn accept_rate(&self) -> f64 {
        if self.n_drafted == 0 { 0.0 }
        else { self.n_accepted as f64 / self.n_drafted as f64 }
    }
}

/// One spec-decode round: K drafted tokens, one batched verify, greedy
/// or rejection-sampling acceptance, optional bonus on full accept.
///
/// `verify_graph` is optional — when present and `K == captured_k` the
/// verify runs as a single `hipGraphLaunch`; otherwise the inline
/// `verify_forward` path is used. Both produce identical logits.
///
/// `prompt` must already be prefilled into `state`; `verify_logits`
/// should hold target's prediction for the position immediately after
/// the prompt (typically from `prefill_forward` + `forward_token`).
/// `last_tok` is the most recently committed token (== prompt's last
/// on the first round, == previously-accepted on later rounds).
#[allow(clippy::too_many_arguments)]
/// `p_min`: drafter early-stop threshold. After each AR draft step the
/// drafted token's probability under the drafter's softmax is computed;
/// if it falls below `p_min`, the K-round terminates early and the
/// verify batch only contains the high-confidence prefix. `0.0`
/// disables (default — always draft K). Ported from llama.cpp
/// common/speculative.cpp's `params.p_min`. Lets K=4 default safely:
/// creative prompts terminate the draft early instead of paying for
/// 4 verify positions at ~30% accept.
pub fn spec_decode_generate(
    target: &GpuGemma4,
    drafter: &GpuGemma4Assistant,
    state: &mut Gemma4GpuState,
    verify_graph: Option<&GraphExec>,
    captured_k: usize,
    mut verify_logits: Vec<f32>,
    mut last_tok: u32,
    eos: u32,
    max_tokens: usize,
    k: usize,
    temperature: f32,
    seed: u64,
    p_min: f32,
) -> Result<(Vec<u32>, SpecStats), String> {
    let mut generated: Vec<u32> = Vec::new();
    let mut stats = SpecStats::default();
    let sampling = temperature > 0.0;
    let mut rng = Rng::new(seed);
    let no_bonus = std::env::var("REINSTINCT_DRAFTER_NO_BONUS").is_ok();

    while generated.len() < max_tokens {
        // --- DRAFT phase: K autoregressive drafter steps pinned to
        //     pos_const = state.pos - 1 (the last-validated position).
        let pos_const = state.pos - 1;
        drafter.set_h_prev_from_target(target)?;
        let mut drafted: Vec<u32> = Vec::with_capacity(k);
        let mut drafter_logits_arr: Vec<Vec<f32>> = Vec::with_capacity(k);
        let mut prev = last_tok;
        for _ in 0..k {
            let logits_d = drafter.forward_step(target, state, prev, pos_const)?;
            let d = if sampling { sample_from_logits(&logits_d, temperature, &mut rng) }
                    else        { argmax(&logits_d) };
            // `p_min` early stop: if the drafter isn't confident at the
            // chosen token, bail before the rest of the K-round — the
            // verify cost of the trailing low-confidence positions
            // usually exceeds the few they'd add via lucky acceptance.
            if p_min > 0.0 {
                let max_logit = logits_d.iter().copied()
                    .filter(|v| v.is_finite())
                    .fold(f32::NEG_INFINITY, f32::max);
                if max_logit.is_finite() {
                    let denom: f32 = logits_d.iter()
                        .map(|&v| if v.is_finite() { (v - max_logit).exp() } else { 0.0 })
                        .sum();
                    let pick_logit = logits_d.get(d as usize).copied().unwrap_or(f32::NEG_INFINITY);
                    let pick_p = if denom > 0.0 && pick_logit.is_finite() {
                        (pick_logit - max_logit).exp() / denom
                    } else { 0.0 };
                    if pick_p < p_min {
                        // Don't push this draft — its low confidence
                        // means the verify pass would almost certainly
                        // reject it. Cap the round at what we have.
                        drafter_logits_arr.push(logits_d);
                        drafted.push(d);
                        break;
                    }
                }
            }
            drafted.push(d);
            drafter_logits_arr.push(logits_d);
            prev = d;
        }
        if drafted.is_empty() {
            // p_min triggered on the FIRST draft step — fall back to a
            // single forward to avoid an empty-verify round.
            let next = target.forward_token(last_tok, state)?;
            let pick = if sampling { sample_from_logits(&next, temperature, &mut rng) }
                       else        { argmax(&next) };
            generated.push(pick);
            last_tok = pick;
            verify_logits = target.forward_token(pick, state)?;
            stats.hit_eos = pick == eos;
            if stats.hit_eos { break; }
            continue;
        }

        // --- BATCHED VERIFY: target processes K candidates in one
        //     forward, returns K logit vectors.
        let pre_verify_pos = state.pos;
        let verify_batch = match verify_graph {
            Some(g) if drafted.len() == captured_k =>
                target.forward_verify_via_graph(g, captured_k, &drafted, state)?,
            _ => target.verify_forward(&drafted, state)?,
        };

        // --- ACCEPTANCE: greedy or rejection-sampling. drafted[i] is
        //     verified against the target's prediction for that slot;
        //     on first reject we keep drafted[..i], commit target's
        //     replacement, truncate the rejected suffix.
        let mut accepted_this_round = 0usize;
        let mut rejected = false;
        for i in 0..drafted.len() {
            let predicting = if i == 0 { &verify_logits } else { &verify_batch[i - 1] };
            let d = drafted[i];
            let (accept, replacement) = if sampling {
                let p_t = softmax_with_temp(predicting, temperature);
                let p_d = softmax_with_temp(&drafter_logits_arr[i], temperature);
                let r = rng.next_f32();
                let ratio = if p_d[d as usize] > 0.0 {
                    (p_t[d as usize] / p_d[d as usize]).min(1.0)
                } else { 0.0 };
                if r < ratio {
                    (true, d)
                } else {
                    let mut residual: Vec<f32> = p_t.iter().zip(p_d.iter())
                        .map(|(t, d)| (t - d).max(0.0)).collect();
                    let s: f32 = residual.iter().sum();
                    if s > 0.0 { for x in &mut residual { *x /= s; } }
                    else       { residual.copy_from_slice(&p_t); }
                    (false, sample_from_probs(&residual, &mut rng))
                }
            } else {
                let target_pred = argmax(predicting);
                if d == target_pred { (true, d) } else { (false, target_pred) }
            };

            if accept {
                generated.push(d);
                accepted_this_round += 1;
                last_tok = d;
                if d == eos { stats.hit_eos = true; rejected = true; break; }
                if generated.len() >= max_tokens { rejected = true; break; }
            } else {
                state.truncate(pre_verify_pos + i);
                let new_verify = target.forward_token(replacement, state)?;
                generated.push(replacement);
                last_tok = replacement;
                stats.hit_eos = replacement == eos;
                verify_logits = new_verify;
                rejected = true;
                break;
            }
        }
        if !rejected {
            // All K drafts accepted ⇒ HF-style bonus: consume target's
            // K+1-th logit as a free committed token. `REINSTINCT_DRAFTER_
            // NO_BONUS=1` reverts to the literature's plain "advance by K"
            // behaviour, for measurement only.
            if no_bonus {
                verify_logits = verify_batch.last().cloned().unwrap();
            } else {
                let bonus = argmax(verify_batch.last().unwrap());
                verify_logits = target.forward_token(bonus, state)?;
                generated.push(bonus);
                last_tok = bonus;
                stats.hit_eos = bonus == eos;
            }
        }
        stats.n_drafted  += drafted.len();
        stats.n_accepted += accepted_this_round;
        if stats.hit_eos { break; }
    }
    Ok((generated, stats))
}
