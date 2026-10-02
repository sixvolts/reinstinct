//! Gemma batched verify (`verify_forward`, dense targets) against the
//! decode oracle: from the same restored state, verify_forward over K
//! tokens and K sequential forward_token calls must give the same
//! per-row logits up to the decode/prefill rounding floor. Also checks
//! a captured verify graph replayed far past its capture position.
//!
//!   REINSTINCT_GEMMA_FIXTURE=... cargo test --release --test \
//!       gemma4_verify_check -- --ignored --nocapture

use reinstinct_engine::gguf::GgufFile;
use reinstinct_engine::hip;
use reinstinct_engine::model::gemma4::Gemma4Model;
use reinstinct_engine::runtime::gemma4::{Gemma4GpuState, GpuGemma4};
use reinstinct_engine::runtime::KernelCache;
use reinstinct_engine::test_support;

fn rel_l2(a: &[f32], b: &[f32]) -> f32 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
    let den: f64 = b.iter().map(|y| (*y as f64).powi(2)).sum();
    (num / den.max(1e-30)).sqrt() as f32
}
fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().fold((0, f32::MIN), |m, (i, &x)| if x > m.1 { (i, x) } else { m }).0
}

#[test]
#[ignore]
fn verify_forward_matches_decode() {
    let (Some(path), Some(())) = (test_support::gemma_fixture(), test_support::gpu()) else { return };
    let g = GgufFile::open(&path).expect("open");
    let _dev = hip::Device::set(0).unwrap();
    let cache = KernelCache::new().unwrap();
    let model = Gemma4Model::load(&g).unwrap();
    let tok = reinstinct_engine::tokenizer::GemmaTokenizer::from_gguf(&g).unwrap();
    let max_seq = 2048;
    let gm = GpuGemma4::new(&model, &g, &cache, max_seq).unwrap();
    let mut state = Gemma4GpuState::new(&model, max_seq).unwrap();
    let text = "The history of the lighthouse begins in the ancient world, where fires were \
                lit on hilltops to guide ships into harbour. The most famous of these was the \
                Pharos of Alexandria, built in the third century BC on a small island off the \
                Egyptian coast. Standing more than one hundred metres tall, it remained one of \
                the tallest man-made structures for many centuries.";
    let mut ids = vec![tok.bos_id];
    ids.extend(tok.encode(text));
    let k = 4usize;
    let (pre, rest) = ids.split_at(ids.len() - k);
    state.reset();
    gm.prefill_forward(pre, &mut state).unwrap();
    let snap = state.snapshot().unwrap();

    let mut dec = Vec::new();
    for &t in rest { dec.push(gm.forward_token(t, &mut state).unwrap()); }
    state.restore(&snap).unwrap();
    let ver = gm.verify_forward(rest, &mut state).unwrap();
    let mut worst = 0f32;
    for i in 0..k {
        let r = rel_l2(&ver[i], &dec[i]);
        eprintln!("row {i}: rel_l2 {r:.2e}  argmax verify {} decode {}", argmax(&ver[i]), argmax(&dec[i]));
        worst = worst.max(r);
    }
    // The 31B K-quant's decode and batched paths differ by ~0.05-0.15
    // rel_l2 on their own; this bounds gross breakage only.
    assert!(worst < 0.2, "verify_forward diverges from decode: {worst:.2e}");

    // A verify graph captured at the prompt position and replayed ~600
    // positions later must match the inline verify there bit-for-bit
    // (its LDS used to be sized for the capture position).
    state.restore(&snap).unwrap();
    let exec = gm.capture_verify_graph(&state, k).unwrap();
    let mut t = rest[0];
    for _ in 0..600 { let l = gm.forward_token(t, &mut state).unwrap(); t = argmax(&l) as u32; }
    let far = state.snapshot().unwrap();
    let probe = [t, 1000, 2000, 3000];
    let inline = gm.verify_forward(&probe, &mut state).unwrap();
    state.restore(&far).unwrap();
    let graph = gm.forward_verify_via_graph(&exec, k, &probe, &mut state).unwrap();
    for i in 0..k {
        let r = rel_l2(&graph[i], &inline[i]);
        eprintln!("far row {i} (pos {}): graph vs inline rel_l2 {r:.2e}", far.pos() + i);
        assert!(r < 1e-6, "verify graph replayed at a later position diverges: {r:.2e}");
    }
}
