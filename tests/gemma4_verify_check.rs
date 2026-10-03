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
    // REINSTINCT_VERIFY_K: rows per verify (default 4).
    let k: usize = std::env::var("REINSTINCT_VERIFY_K").ok().and_then(|v| v.parse().ok()).unwrap_or(4);
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
    let probe: Vec<u32> = std::iter::once(t).chain((1..k as u32).map(|i| 1000 * i)).collect();
    let inline = gm.verify_forward(&probe, &mut state).unwrap();
    state.restore(&far).unwrap();
    let graph = gm.forward_verify_via_graph(&exec, k, &probe, &mut state).unwrap();
    for i in 0..k {
        let r = rel_l2(&graph[i], &inline[i]);
        eprintln!("far row {i} (pos {}): graph vs inline rel_l2 {r:.2e}", far.pos() + i);
        assert!(r < 1e-6, "verify graph replayed at a later position diverges: {r:.2e}");
    }
}

/// Diagnostic: last-position top-5 logits for REINSTINCT_DIAG_TOKENS
/// (comma-separated ids) via batched prefill and via the decode loop,
/// for comparison against tests/golden/dump_logits.
#[test]
#[ignore]
fn diag_last_logits() {
    let Ok(csv) = std::env::var("REINSTINCT_DIAG_TOKENS") else { return };
    let ids: Vec<u32> = csv.split(',').map(|t| t.trim().parse().unwrap()).collect();
    let (Some(path), Some(())) = (test_support::gemma_fixture(), test_support::gpu()) else { return };
    let g = GgufFile::open(&path).expect("open");
    let _dev = hip::Device::set(0).unwrap();
    let cache = KernelCache::new().unwrap();
    let model = Gemma4Model::load(&g).unwrap();
    let gm = GpuGemma4::new(&model, &g, &cache, ids.len() + 8).unwrap();
    let mut state = Gemma4GpuState::new(&model, ids.len() + 8).unwrap();
    let top = |v: &[f32]| { let mut i: Vec<usize> = (0..v.len()).collect();
        i.sort_by(|a, b| v[*b].partial_cmp(&v[*a]).unwrap()); i.truncate(5);
        i.iter().map(|&j| (j, (v[j] * 100.0).round() / 100.0)).collect::<Vec<_>>() };
    state.reset();
    let lp = gm.prefill_forward(&ids, &mut state).unwrap();
    state.reset();
    let mut ld = Vec::new();
    for &t in &ids { ld = gm.forward_token(t, &mut state).unwrap(); }
    eprintln!("prefill: {:?}\ndecode:  {:?}\nrel_l2 prefill vs decode {:.3e}", top(&lp), top(&ld), rel_l2(&lp, &ld));
}

/// Diagnostic: mean next-token NLL (and perplexity) of the decode loop
/// over the texts in REINSTINCT_PPL_FILE (separated by "=====", each
/// truncated to REINSTINCT_PPL_TOKENS, default 384, BOS-prefixed).
#[test]
#[ignore]
fn diag_decode_perplexity() {
    let Ok(file) = std::env::var("REINSTINCT_PPL_FILE") else { return };
    let n_tok: usize = std::env::var("REINSTINCT_PPL_TOKENS").ok().and_then(|v| v.parse().ok()).unwrap_or(384);
    let (Some(path), Some(())) = (test_support::gemma_fixture(), test_support::gpu()) else { return };
    let g = GgufFile::open(&path).expect("open");
    let _dev = hip::Device::set(0).unwrap();
    let cache = KernelCache::new().unwrap();
    let model = Gemma4Model::load(&g).unwrap();
    let tok = reinstinct_engine::tokenizer::GemmaTokenizer::from_gguf(&g).unwrap();
    let gm = GpuGemma4::new(&model, &g, &cache, n_tok + 8).unwrap();
    let mut state = Gemma4GpuState::new(&model, n_tok + 8).unwrap();
    let text = std::fs::read_to_string(file).unwrap();
    let (mut nll, mut n) = (0f64, 0usize);
    for chunk in text.split("=====") {
        let mut ids = vec![tok.bos_id];
        ids.extend(tok.encode(chunk.trim()));
        ids.truncate(n_tok);
        state.reset();
        let verbose = std::env::var_os("REINSTINCT_PPL_VERBOSE").is_some();
        for w in ids.windows(2) {
            let l = gm.forward_token(w[0], &mut state).unwrap();
            let mx = l.iter().cloned().fold(f32::MIN, f32::max) as f64;
            let lse = mx + l.iter().map(|&x| (x as f64 - mx).exp()).sum::<f64>().ln();
            let t_nll = lse - l[w[1] as usize] as f64;
            if verbose {
                eprintln!("  next {:>7} {:<14?} nll {:6.2}  argmax {:>7} {:?}", w[1],
                          tok.decode(&[w[1]]), t_nll, argmax(&l), tok.decode(&[argmax(&l) as u32]));
            }
            nll += t_nll;
            n += 1;
        }
    }
    eprintln!("{}: {n} tokens, mean NLL {:.4}, ppl {:.3}", path.display(), nll / n as f64, (nll / n as f64).exp());
}
