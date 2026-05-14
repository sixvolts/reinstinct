//! Compare our CPU forward against the llama.cpp golden-logits fixture.
//!
//! Fixture: tests/golden/qwen35_0_8B_eos.json (top-32 logits + summary stats
//! for token 248046 on Qwen3.5-0.8B-UD-Q4_K_XL, captured with llama.cpp 9113).
//!
//! Skipped when the GGUF fixture is absent.

use std::fs;
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

#[derive(Debug)]
struct Golden {
    input_token: u32,
    vocab_size: usize,
    min: f32,
    max: f32,
    mean: f32,
    std: f32,
    top: Vec<(u32, f32)>,
}

/// Hand-rolled extractor for the fixed-shape JSON we produce in dump_logits.cpp.
/// Avoids pulling in serde_json for one consumer.
fn parse_golden(text: &str) -> Golden {
    fn find_num(text: &str, key: &str) -> f64 {
        let needle = format!("\"{key}\":");
        let i = text.find(&needle).unwrap_or_else(|| panic!("key not found: {key}")) + needle.len();
        let rest = &text[i..];
        let s = rest.trim_start();
        let end = s.find(|c: char| !(c.is_ascii_digit() || c == '-' || c == '.' || c == 'e' || c == '+'))
            .unwrap_or(s.len());
        s[..end].parse().unwrap_or_else(|_| panic!("bad number for {key}: {:?}", &s[..end]))
    }

    let input_token = find_num(text, "input_token") as u32;
    let vocab_size  = find_num(text, "vocab_size") as usize;
    let min  = find_num(text, "min") as f32;
    let max  = find_num(text, "max") as f32;
    let mean = find_num(text, "mean") as f32;
    let std  = find_num(text, "std") as f32;

    let mut top = Vec::new();
    let mut cursor = 0;
    while let Some(rel) = text[cursor..].find("\"idx\":") {
        let idx_start = cursor + rel + "\"idx\":".len();
        let idx_str_end = text[idx_start..].find(',').unwrap();
        let idx: u32 = text[idx_start..idx_start + idx_str_end].trim().parse().unwrap();

        let logit_key = "\"logit\":";
        let lo_rel = text[idx_start..].find(logit_key).unwrap();
        let lo_start = idx_start + lo_rel + logit_key.len();
        let lo_end = text[lo_start..].find('}').unwrap();
        let logit: f32 = text[lo_start..lo_start + lo_end].trim().parse().unwrap();

        top.push((idx, logit));
        cursor = lo_start + lo_end;
    }

    Golden { input_token, vocab_size, min, max, mean, std, top }
}

#[test]
fn forward_matches_llamacpp_golden_within_tolerance() {
    let Some(model_path) = fixture_path() else {
        eprintln!("skipping: GGUF fixture not found");
        return;
    };

    let golden_text = fs::read_to_string("tests/golden/qwen35_0_8B_eos.json")
        .expect("golden fixture missing — run tests/golden/build.sh + dump_logits");
    let g = parse_golden(&golden_text);
    eprintln!("golden parsed: top[0] = {:?}", g.top[0]);

    let gguf = GgufFile::open(&model_path).expect("open Qwen 0.8B");
    let m = Qwen35F32Model::load(&gguf).expect("Qwen35F32Model::load");
    assert_eq!(m.model.config.vocab_size as usize, g.vocab_size);

    let mut state = m.new_state(16);
    let logits = m.forward_token(g.input_token, &mut state);

    // Compute our summary stats.
    let (mut omin, mut omax) = (f32::INFINITY, f32::NEG_INFINITY);
    let mut sum = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    let mut argmax = 0usize;
    for (i, &v) in logits.iter().enumerate() {
        if v < omin { omin = v; }
        if v > omax { omax = v; argmax = i; }
        sum += v as f64;
        sum_sq += (v as f64) * (v as f64);
    }
    let n = logits.len() as f64;
    let omean = (sum / n) as f32;
    let ostd = ((sum_sq / n) - (omean as f64).powi(2)).sqrt() as f32;

    eprintln!("        ours        golden        delta");
    eprintln!("min   {omin:>9.4}   {:>9.4}   {:+.4}", g.min, omin - g.min);
    eprintln!("max   {omax:>9.4}   {:>9.4}   {:+.4}", g.max, omax - g.max);
    eprintln!("mean  {omean:>9.4}   {:>9.4}   {:+.4}", g.mean, omean - g.mean);
    eprintln!("std   {ostd:>9.4}   {:>9.4}   {:+.4}", g.std, ostd - g.std);
    eprintln!("argmax = {argmax}, golden argmax = {}", g.top[0].0);

    // Hard assertion: argmax must match.
    assert_eq!(argmax as u32, g.top[0].0,
        "argmax mismatch: we predict {argmax}, golden predicts {}", g.top[0].0);

    // Summary-stat tolerance: 5% relative + 0.2 absolute floor.
    let tol = |a: f32, b: f32| (a - b).abs() <= 0.05 * b.abs() + 0.2;
    assert!(tol(omin, g.min),  "min outside tolerance");
    assert!(tol(omax, g.max),  "max outside tolerance");
    assert!(tol(omean, g.mean), "mean outside tolerance");
    assert!(tol(ostd, g.std),   "std outside tolerance");

    // Top-5 token id overlap: at least 4 of 5 must match.
    let our_top5: Vec<u32> = {
        let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
        idx.sort_by(|a, b| logits[*b as usize].partial_cmp(&logits[*a as usize]).unwrap());
        idx.into_iter().take(5).collect()
    };
    let golden_top5: Vec<u32> = g.top.iter().take(5).map(|(t, _)| *t).collect();
    let overlap = our_top5.iter().filter(|t| golden_top5.contains(t)).count();
    eprintln!("top-5 ours    = {:?}", our_top5);
    eprintln!("top-5 golden  = {:?}", golden_top5);
    eprintln!("top-5 overlap = {overlap}/5");
    assert!(overlap >= 4, "top-5 overlap = {overlap}, expected >= 4");

    // For tokens in BOTH top-5s, check logit values are close.
    //
    // Tolerance: 1.5 absolute or 20% relative — whichever is larger.
    // We use fp32 throughout while llama.cpp's CPU path uses fp16/bf16 for
    // intermediates (Q/K/V cache, attention scores), so per-token logit
    // values diverge by 1–2 logit units even when the rank ordering and
    // distribution shape match. The bar that matters is argmax + top-K
    // overlap (already asserted above), not bit-equal logits.
    for &(tok, golden_logit) in g.top.iter().take(5) {
        let our_logit = logits[tok as usize];
        let abs_err = (our_logit - golden_logit).abs();
        let rel_err = abs_err / (golden_logit.abs() + 1e-3);
        eprintln!("  token {tok:>6}: ours {our_logit:.4}  golden {golden_logit:.4}  abs {abs_err:.3}  rel {rel_err:.3}");
        assert!(abs_err < 1.5 || rel_err < 0.20,
            "token {tok}: ours {our_logit} vs golden {golden_logit}, abs_err={abs_err}, rel_err={rel_err}");
    }
}
