//! Compare our CPU forward against llama.cpp golden-logits fixtures.
//!
//! Fixtures live at tests/golden/qwen35_0_8B_*.json, each captured by
//! tests/golden/dump_logits on the same Qwen3.5-0.8B-UD-Q4_K_XL GGUF
//! file via libllama. The test iterates every JSON it finds.
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
    input_tokens: Vec<u32>,
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

    fn find_tokens(text: &str) -> Vec<u32> {
        let needle = "\"input_tokens\":";
        let Some(i) = text.find(needle) else { return Vec::new(); };
        let after = &text[i + needle.len()..];
        let lb = after.find('[').expect("input_tokens missing [");
        let rb = after.find(']').expect("input_tokens missing ]");
        after[lb + 1..rb]
            .split(',')
            .filter_map(|s| s.trim().parse::<u32>().ok())
            .collect()
    }

    let mut input_tokens = find_tokens(text);
    if input_tokens.is_empty() {
        // Older fixtures had only `input_token` (singular); fall back.
        input_tokens.push(find_num(text, "input_token") as u32);
    }
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

    Golden { input_tokens, vocab_size, min, max, mean, std, top }
}

fn check_against_golden(name: &str, m: &Qwen35F32Model, g: &Golden) {
    eprintln!("\n=== fixture: {name} (input_tokens = {:?}) ===", g.input_tokens);
    assert_eq!(m.model.config.vocab_size as usize, g.vocab_size);

    let mut state = m.new_state((g.input_tokens.len() + 8).max(16));
    let mut logits = Vec::new();
    for &tok in &g.input_tokens {
        logits = m.forward_token(tok, &mut state);
    }
    assert_eq!(state.pos, g.input_tokens.len());

    // Summary stats.
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
    // Summary-stat tolerance: 10% relative + 0.5 absolute floor. Multi-token
    // sequences accumulate fp32-vs-fp16 rounding through more state updates,
    // so we can't be as tight as the single-token case.
    let tol = |a: f32, b: f32| (a - b).abs() <= 0.10 * b.abs() + 0.5;
    assert!(tol(omin, g.min),  "[{name}] min outside tolerance");
    assert!(tol(omax, g.max),  "[{name}] max outside tolerance");
    assert!(tol(omean, g.mean), "[{name}] mean outside tolerance");
    assert!(tol(ostd, g.std),   "[{name}] std outside tolerance");

    // Top-K sets — sort our logits once, reuse for both checks.
    let mut sorted_idx: Vec<u32> = (0..logits.len() as u32).collect();
    sorted_idx.sort_by(|a, b| logits[*b as usize].partial_cmp(&logits[*a as usize]).unwrap());
    let our_top3: Vec<u32> = sorted_idx.iter().take(3).copied().collect();
    let our_top5: Vec<u32> = sorted_idx.iter().take(5).copied().collect();
    let golden_top5: Vec<u32> = g.top.iter().take(5).map(|(t, _)| *t).collect();
    let overlap = our_top5.iter().filter(|t| golden_top5.contains(t)).count();
    eprintln!("argmax ours = {argmax}, golden = {}", g.top[0].0);
    eprintln!("top-5 ours    = {our_top5:?}");
    eprintln!("top-5 golden  = {golden_top5:?}");
    eprintln!("top-5 overlap = {overlap}/5");

    // Argmax: tolerate top-1/top-2 flips. Golden's argmax must be in our
    // top-3 — when the top tokens are within ~0.2 logits of each other,
    // fp precision noise can swap the rank.
    assert!(our_top3.contains(&g.top[0].0),
        "[{name}] golden argmax {} not in our top-3 {our_top3:?}", g.top[0].0);

    assert!(overlap >= 4, "[{name}] top-5 overlap = {overlap}, expected >= 4");

    // Per-token logit tolerance: 1.5 abs OR 20% rel.
    for &(tok, golden_logit) in g.top.iter().take(5) {
        let our_logit = logits[tok as usize];
        let abs_err = (our_logit - golden_logit).abs();
        let rel_err = abs_err / (golden_logit.abs() + 1e-3);
        eprintln!("  token {tok:>6}: ours {our_logit:.4}  golden {golden_logit:.4}  abs {abs_err:.3}  rel {rel_err:.3}");
        assert!(abs_err < 1.5 || rel_err < 0.20,
            "[{name}] token {tok}: ours {our_logit} vs golden {golden_logit}, abs={abs_err}, rel={rel_err}");
    }
}

#[test]
fn forward_matches_all_llamacpp_golden_fixtures() {
    let Some(model_path) = fixture_path() else {
        eprintln!("skipping: GGUF fixture not found");
        return;
    };
    let gguf = GgufFile::open(&model_path).expect("open Qwen 0.8B");
    let m = Qwen35F32Model::load(&gguf).expect("Qwen35F32Model::load");

    // Iterate every JSON fixture in tests/golden/.
    let mut entries: Vec<PathBuf> = fs::read_dir("tests/golden")
        .expect("tests/golden missing")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("json"))
        .collect();
    entries.sort();
    assert!(!entries.is_empty(), "no JSON fixtures in tests/golden/");

    for fixture in &entries {
        let text = fs::read_to_string(fixture)
            .unwrap_or_else(|e| panic!("read {}: {e}", fixture.display()));
        let g = parse_golden(&text);
        let name = fixture.file_name().unwrap().to_string_lossy().into_owned();
        check_against_golden(&name, &m, &g);
    }
}
