//! Self-consistency properties of the CPU forward.
//!
//! Catches bug classes the golden-logits test doesn't: state leakage across
//! prompts, position-blind forward (RoPE/KV not actually applied), accidental
//! determinism breaks. None of these need an external reference.

use std::path::PathBuf;

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

fn argmax(logits: &[f32]) -> usize {
    let mut best = 0;
    for (i, &v) in logits.iter().enumerate() {
        if v > logits[best] { best = i; }
    }
    best
}

fn diff_rms(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut sum = 0.0_f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        sum += d * d;
    }
    (sum / a.len() as f32).sqrt()
}

#[test]
fn forward_is_deterministic() {
    let Some(p) = fixture_path() else { eprintln!("skipping"); return; };
    let g = GgufFile::open(&p).unwrap();
    let m = Qwen35F32Model::load(&g).unwrap();
    let token = m.model.config.eos_token_id;

    let mut s1 = m.new_state(8);
    let mut s2 = m.new_state(8);
    let l1 = m.forward_token(token, &mut s1);
    let l2 = m.forward_token(token, &mut s2);
    assert_eq!(l1.len(), l2.len());
    for i in 0..l1.len() {
        assert_eq!(l1[i].to_bits(), l2[i].to_bits(),
            "non-deterministic at index {i}: {} vs {}", l1[i], l2[i]);
    }
}

#[test]
fn reset_restores_initial_behavior() {
    let Some(p) = fixture_path() else { eprintln!("skipping"); return; };
    let g = GgufFile::open(&p).unwrap();
    let m = Qwen35F32Model::load(&g).unwrap();
    let token = m.model.config.eos_token_id;

    let mut s = m.new_state(8);
    let initial = m.forward_token(token, &mut s);
    // Run a couple more tokens to populate KV/GDN state.
    let _ = m.forward_token(0, &mut s);
    let _ = m.forward_token(1, &mut s);
    assert_eq!(s.pos, 3);

    s.reset();
    assert_eq!(s.pos, 0);
    let after_reset = m.forward_token(token, &mut s);

    for i in 0..initial.len() {
        assert_eq!(initial[i].to_bits(), after_reset[i].to_bits(),
            "reset didn't fully clear state at index {i}: initial {} vs after_reset {}",
            initial[i], after_reset[i]);
    }
}

#[test]
fn distinct_tokens_produce_distinct_logits() {
    let Some(p) = fixture_path() else { eprintln!("skipping"); return; };
    let g = GgufFile::open(&p).unwrap();
    let m = Qwen35F32Model::load(&g).unwrap();

    let mut s_a = m.new_state(8);
    let mut s_b = m.new_state(8);
    // Two different tokens at fresh state.
    let la = m.forward_token(100, &mut s_a);
    let lb = m.forward_token(2000, &mut s_b);
    let rms = diff_rms(&la, &lb);
    eprintln!("forward(100) vs forward(2000) diff rms = {rms:.4}");
    assert!(rms > 0.5,
        "forward output appears insensitive to input token (rms diff {rms})");
    // And argmax usually differs too.
    let am_a = argmax(&la);
    let am_b = argmax(&lb);
    eprintln!("argmax(100) = {am_a}, argmax(2000) = {am_b}");
    // We don't strictly assert different argmax — could collide for some pairs —
    // but rms diff > 0.5 is an extremely strong signal that tokens propagate.
}

#[test]
fn second_token_differs_from_first_due_to_state() {
    let Some(p) = fixture_path() else { eprintln!("skipping"); return; };
    let g = GgufFile::open(&p).unwrap();
    let m = Qwen35F32Model::load(&g).unwrap();
    let token = m.model.config.eos_token_id;

    let mut s_a = m.new_state(8);
    let mut s_b = m.new_state(8);

    // s_a: forward(EOS) at position 0 only.
    let la = m.forward_token(token, &mut s_a);

    // s_b: feed an arbitrary token first, then forward(EOS) at position 1.
    let _ = m.forward_token(100, &mut s_b);
    let lb = m.forward_token(token, &mut s_b);

    let rms = diff_rms(&la, &lb);
    eprintln!("forward(EOS) at pos 0 vs at pos 1 (after token 100): diff rms = {rms:.4}");
    assert!(rms > 0.1,
        "position/state seems to have no effect on forward (diff rms {rms})");
}
