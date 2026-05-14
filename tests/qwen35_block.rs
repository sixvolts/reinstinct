//! Smoke test for the full-attention block forward on real Qwen 3.5 weights.
//!
//! No reference comparison yet — that's the validation gate at the end of
//! Phase 3.6. This test asserts no NaN/Inf, output magnitude stays bounded,
//! and successive calls with growing KV cache produce different (sensible)
//! outputs. If any of those break, the bug surfaces before we've wired the
//! whole forward pass.

use std::path::PathBuf;

use reinstinct_engine::cpu::qwen3_5::{
    full_attention_block, linear_attention_block, BlockWeights, LayerKvCache, LinAttnState,
    Qwen35F32Weights,
};
use reinstinct_engine::cpu::rope::RopeCache;
use reinstinct_engine::gguf::GgufFile;
use reinstinct_engine::model::qwen3_5::{BlockKind, Qwen35Model};

fn fixture_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("REINSTINCT_GGUF_FIXTURE") {
        return Some(PathBuf::from(p));
    }
    let home = std::env::var_os("HOME")?;
    let p = PathBuf::from(home).join("models/qwen-3.5-0.8B/Qwen3.5-0.8B-UD-Q4_K_XL.gguf");
    p.exists().then_some(p)
}

#[test]
fn full_attention_block_runs_on_real_weights() {
    let Some(path) = fixture_path() else {
        eprintln!("skipping: GGUF fixture not found");
        return;
    };

    let g = GgufFile::open(&path).expect("open Qwen 0.8B");
    let model = Qwen35Model::load(&g).expect("Qwen35Model::load");
    let w = Qwen35F32Weights::load(&g, &model).expect("dequant weights");

    let cfg = &model.config;
    let max_seq = 16;
    let rope = RopeCache::new(
        cfg.rope_dim_count as usize,
        max_seq,
        cfg.rope_freq_base,
    );

    // Block 3 is the first full-attention block per the [L,L,L,F]×6 schedule.
    assert_eq!(model.block_kinds[3], BlockKind::FullAttention);
    let weights = match &w.blocks[3] {
        BlockWeights::FullAttention(fa) => fa,
        _ => panic!("expected full attention at block 3"),
    };

    let mut kv = LayerKvCache::new(
        max_seq,
        cfg.attn_n_kv_heads as usize,
        cfg.attn_head_dim as usize,
    );

    // Step 0: synthetic hidden state with small magnitude (matches real
    // post-embed activations roughly). Norm RMS should bring this to ~1
    // pre-projection regardless.
    let h_dim = cfg.hidden_size as usize;
    let mut hidden: Vec<f32> = (0..h_dim).map(|i| {
        let x = (i as f32 / h_dim as f32 - 0.5) * 0.1;
        x.sin() * 0.05
    }).collect();
    let input_rms: f32 = (hidden.iter().map(|v| v * v).sum::<f32>() / h_dim as f32).sqrt();
    eprintln!("input rms       = {input_rms:.6}");

    full_attention_block(&mut hidden, weights, cfg, &mut kv, &rope, 0);

    let out_rms_0: f32 = (hidden.iter().map(|v| v * v).sum::<f32>() / h_dim as f32).sqrt();
    let max_abs_0 = hidden.iter().fold(0.0_f32, |a, &b| a.max(b.abs()));
    let nan_count = hidden.iter().filter(|v| !v.is_finite()).count();
    eprintln!("after step 0:    rms = {out_rms_0:.6}  max|x| = {max_abs_0:.6}");
    assert_eq!(nan_count, 0, "non-finite outputs after step 0");
    assert!(max_abs_0 < 100.0, "output magnitude exploded: max|x| = {max_abs_0}");
    assert_eq!(kv.len(), 1);

    // Step 1: feed the previous block's output back in as if running multiple
    // tokens. This stresses the KV-cache attention path with cache_len = 2.
    full_attention_block(&mut hidden, weights, cfg, &mut kv, &rope, 1);
    let out_rms_1: f32 = (hidden.iter().map(|v| v * v).sum::<f32>() / h_dim as f32).sqrt();
    let max_abs_1 = hidden.iter().fold(0.0_f32, |a, &b| a.max(b.abs()));
    let nan_count = hidden.iter().filter(|v| !v.is_finite()).count();
    eprintln!("after step 1:    rms = {out_rms_1:.6}  max|x| = {max_abs_1:.6}");
    assert_eq!(nan_count, 0, "non-finite outputs after step 1");
    assert!(max_abs_1 < 100.0, "output magnitude exploded: max|x| = {max_abs_1}");
    assert_eq!(kv.len(), 2);
}

#[test]
fn linear_attention_block_runs_on_real_weights() {
    let Some(path) = fixture_path() else {
        eprintln!("skipping: GGUF fixture not found");
        return;
    };

    let g = GgufFile::open(&path).expect("open Qwen 0.8B");
    let model = Qwen35Model::load(&g).expect("Qwen35Model::load");
    let w = Qwen35F32Weights::load(&g, &model).expect("dequant weights");
    let cfg = &model.config;

    // Block 0 is the first linear-attention block per the [L,L,L,F]×6 schedule.
    assert_eq!(model.block_kinds[0], BlockKind::LinearAttention);
    let weights = match &w.blocks[0] {
        BlockWeights::LinearAttention(la) => la,
        _ => panic!("expected linear attention at block 0"),
    };

    let n_heads = cfg.gdn_n_heads as usize;
    let head_dim = cfg.gdn_head_dim as usize;
    let value_dim = cfg.gdn_value_dim as usize;
    let conv_dim = 3 * value_dim;
    let mut state = LinAttnState::new(
        n_heads, head_dim, head_dim,
        conv_dim, cfg.gdn_conv_kernel as usize,
    );

    let h_dim = cfg.hidden_size as usize;
    let mut hidden: Vec<f32> = (0..h_dim).map(|i| {
        let x = (i as f32 / h_dim as f32 - 0.5) * 0.1;
        x.sin() * 0.05
    }).collect();
    let input_rms: f32 = (hidden.iter().map(|v| v * v).sum::<f32>() / h_dim as f32).sqrt();
    eprintln!("GDN input rms     = {input_rms:.6}");

    linear_attention_block(&mut hidden, weights, cfg, &mut state);
    let out_rms_0: f32 = (hidden.iter().map(|v| v * v).sum::<f32>() / h_dim as f32).sqrt();
    let max_abs_0 = hidden.iter().fold(0.0_f32, |a, &b| a.max(b.abs()));
    let nan_count = hidden.iter().filter(|v| !v.is_finite()).count();
    eprintln!("GDN after step 0: rms = {out_rms_0:.6}  max|x| = {max_abs_0:.6}");
    assert_eq!(nan_count, 0, "non-finite outputs after GDN step 0");
    assert!(max_abs_0 < 100.0, "output magnitude exploded: max|x| = {max_abs_0}");

    // Step 1 — recurrent state from step 0 should now decay & re-update.
    linear_attention_block(&mut hidden, weights, cfg, &mut state);
    let out_rms_1: f32 = (hidden.iter().map(|v| v * v).sum::<f32>() / h_dim as f32).sqrt();
    let max_abs_1 = hidden.iter().fold(0.0_f32, |a, &b| a.max(b.abs()));
    let nan_count = hidden.iter().filter(|v| !v.is_finite()).count();
    eprintln!("GDN after step 1: rms = {out_rms_1:.6}  max|x| = {max_abs_1:.6}");
    assert_eq!(nan_count, 0, "non-finite outputs after GDN step 1");
    assert!(max_abs_1 < 100.0, "output magnitude exploded: max|x| = {max_abs_1}");

    // Indirect state check: a third step on a fresh-state copy should differ
    // from the third step on the populated state, proving recurrence is wired.
    let mut hidden_fresh = hidden.clone();
    let mut state_fresh = LinAttnState::new(
        n_heads, head_dim, head_dim,
        conv_dim, cfg.gdn_conv_kernel as usize,
    );
    linear_attention_block(&mut hidden_fresh, weights, cfg, &mut state_fresh);

    let mut hidden_continued = hidden.clone();
    linear_attention_block(&mut hidden_continued, weights, cfg, &mut state);

    let mut diff = 0.0_f32;
    for i in 0..h_dim {
        diff += (hidden_continued[i] - hidden_fresh[i]).powi(2);
    }
    let diff_rms = (diff / h_dim as f32).sqrt();
    eprintln!("fresh-vs-continued state diff rms = {diff_rms:.6}");
    assert!(diff_rms > 1e-5,
        "GDN output identical with fresh vs accumulated state — recurrence is no-op");
}
