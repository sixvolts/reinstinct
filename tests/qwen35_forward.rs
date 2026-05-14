//! End-to-end forward pass smoke test on real Qwen 3.5 0.8B weights.
//!
//! No reference comparison yet — that's the validation gate against HF
//! transformers. This test verifies the pipeline runs through all 24 blocks
//! producing finite logits with sensible distribution properties.
//!
//! Skipped when the GGUF fixture is absent.

use std::path::PathBuf;
use std::time::Instant;

use reinstinct_engine::cpu::qwen3_5::Qwen35F32Model;
use reinstinct_engine::gguf::GgufFile;

fn fixture_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("REINSTINCT_GGUF_FIXTURE") {
        return Some(PathBuf::from(p));
    }
    let home = std::env::var_os("HOME")?;
    let p = PathBuf::from(home).join("models/qwen-3.5-0.8B/Qwen3.5-0.8B-UD-Q4_K_XL.gguf");
    p.exists().then_some(p)
}

#[test]
fn forward_token_produces_plausible_logits() {
    let Some(path) = fixture_path() else {
        eprintln!("skipping: GGUF fixture not found");
        return;
    };

    let g = GgufFile::open(&path).expect("open Qwen 0.8B");
    let m = Qwen35F32Model::load(&g).expect("Qwen35F32Model::load");
    let cfg = &m.model.config;
    let v = cfg.vocab_size as usize;

    // Use the EOS token id from the GGUF metadata as a known-valid token.
    let token = cfg.eos_token_id;
    eprintln!("forward token = {token} (eos), vocab = {v}");

    let mut state = m.new_state(16);

    let t0 = Instant::now();
    let logits = m.forward_token(token, &mut state);
    let elapsed = t0.elapsed();
    eprintln!("forward_token took {:.2} s", elapsed.as_secs_f32());

    assert_eq!(logits.len(), v);
    assert_eq!(state.pos, 1);

    // Distribution sanity.
    let nan = logits.iter().filter(|x| !x.is_finite()).count();
    assert_eq!(nan, 0, "{nan} non-finite logits");

    let (mut min, mut max) = (f32::INFINITY, f32::NEG_INFINITY);
    let mut sum = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    let mut argmax = 0usize;
    for (i, &x) in logits.iter().enumerate() {
        if x < min { min = x; }
        if x > max { max = x; argmax = i; }
        sum += x as f64;
        sum_sq += (x as f64) * (x as f64);
    }
    let mean = (sum / v as f64) as f32;
    let std = ((sum_sq / v as f64) - (mean as f64).powi(2)).sqrt() as f32;
    eprintln!("logits: min={min:.4} max={max:.4} mean={mean:.4} std={std:.4} argmax={argmax}");

    // Sanity bounds. Pre-softmax logits for a real LLM are typically in
    // [-30, 40] after a single decode step; we give wide slack.
    assert!(min > -200.0 && max < 200.0,
        "logits magnitude looks broken: [{min}, {max}]");
    // Some variation across the vocab (not all-the-same).
    assert!(std > 0.1, "logit std={std} suspiciously small — pipeline may be degenerate");
}
