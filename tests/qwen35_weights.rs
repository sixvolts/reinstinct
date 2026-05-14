//! Eagerly dequantize all Qwen 3.5 0.8B weights into f32 buffers and
//! sanity-check shapes/sizes/values. ~3 GB host RAM; ~5–10 s in dev.
//!
//! Skipped when the GGUF fixture is absent.

use std::path::PathBuf;
use std::time::Instant;

use reinstinct_engine::cpu::qwen3_5::{BlockWeights, Qwen35F32Weights};
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
fn dequant_cache_loads_and_shapes_match_config() {
    let Some(path) = fixture_path() else {
        eprintln!("skipping: GGUF fixture not found");
        return;
    };

    let g = GgufFile::open(&path).expect("open Qwen 0.8B");
    let model = Qwen35Model::load(&g).expect("Qwen35Model::load");

    let t0 = Instant::now();
    let w = Qwen35F32Weights::load(&g, &model).expect("dequant cache");
    let elapsed = t0.elapsed();
    eprintln!("dequantized full Qwen 3.5 0.8B in {:.2} s", elapsed.as_secs_f32());

    let cfg = &model.config;
    let h  = cfg.hidden_size as usize;
    let v  = cfg.vocab_size as usize;
    let f  = cfg.ffn_size as usize;
    let vd = cfg.gdn_value_dim as usize;
    let nh = cfg.gdn_n_heads as usize;
    let kk = cfg.gdn_conv_kernel as usize;
    let hd = cfg.gdn_head_dim as usize;
    let an = cfg.attn_n_heads as usize;
    let av = cfg.attn_n_kv_heads as usize;
    let ad = cfg.attn_head_dim as usize;

    // Top-level
    assert_eq!(w.token_embd.len(), v * h, "token_embd shape");
    assert_eq!(w.output_norm.len(), h, "output_norm shape");
    assert!(w.output.is_none(), "tied embeddings → no separate output");

    assert_eq!(w.blocks.len(), cfg.block_count as usize);

    let mut n_lin = 0;
    let mut n_full = 0;
    for (i, blk) in w.blocks.iter().enumerate() {
        match (model.block_kinds[i], blk) {
            (BlockKind::LinearAttention, BlockWeights::LinearAttention(la)) => {
                n_lin += 1;
                assert_eq!(la.attn_norm.len(), h, "block {i} attn_norm");
                assert_eq!(la.attn_qkv.len(), h * 3 * vd, "block {i} attn_qkv (3×value_dim)");
                assert_eq!(la.attn_gate.len(), h * vd, "block {i} attn_gate");
                assert_eq!(la.ssm_alpha.len(), h * nh, "block {i} ssm_alpha");
                assert_eq!(la.ssm_beta.len(), h * nh, "block {i} ssm_beta");
                assert_eq!(la.ssm_a.len(), nh, "block {i} ssm_a");
                assert_eq!(la.ssm_dt_bias.len(), nh, "block {i} ssm_dt_bias");
                // ssm_conv1d: ggml shape [kernel, conv_dim] where conv_dim = 3*value_dim
                assert_eq!(la.ssm_conv1d.len(), kk * 3 * vd, "block {i} ssm_conv1d");
                assert_eq!(la.ssm_norm.len(), hd, "block {i} ssm_norm");
                assert_eq!(la.ssm_out.len(), vd * h, "block {i} ssm_out");
                assert_eq!(la.post_attention_norm.len(), h, "block {i} post_attention_norm");
                assert_eq!(la.ffn_gate.len(), h * f, "block {i} ffn_gate");
                assert_eq!(la.ffn_up.len(), h * f, "block {i} ffn_up");
                assert_eq!(la.ffn_down.len(), f * h, "block {i} ffn_down");
            }
            (BlockKind::FullAttention, BlockWeights::FullAttention(fa)) => {
                n_full += 1;
                assert_eq!(fa.attn_norm.len(), h, "block {i} attn_norm");
                // attn_q outputs 2 * n_heads * head_dim because Q + Q_gate are concatenated.
                assert_eq!(fa.attn_q.len(), h * 2 * an * ad, "block {i} attn_q (Q | Q_gate)");
                assert_eq!(fa.attn_k.len(), h * av * ad, "block {i} attn_k");
                assert_eq!(fa.attn_v.len(), h * av * ad, "block {i} attn_v");
                assert_eq!(fa.attn_q_norm.len(), ad, "block {i} attn_q_norm");
                assert_eq!(fa.attn_k_norm.len(), ad, "block {i} attn_k_norm");
                assert_eq!(fa.attn_output.len(), an * ad * h, "block {i} attn_output");
                assert_eq!(fa.post_attention_norm.len(), h, "block {i} post_attention_norm");
                assert_eq!(fa.ffn_gate.len(), h * f, "block {i} ffn_gate");
                assert_eq!(fa.ffn_up.len(), h * f, "block {i} ffn_up");
                assert_eq!(fa.ffn_down.len(), f * h, "block {i} ffn_down");
            }
            (k, _) => panic!("block {i} kind mismatch: schedule says {:?}, weights say something else", k),
        }
    }
    assert_eq!(n_lin, 18);
    assert_eq!(n_full, 6);

    // Spot sanity: every weight buffer should contain at least some non-zero values.
    let bad: Vec<&str> = match &w.blocks[0] {
        BlockWeights::LinearAttention(la) => {
            let mut out = Vec::new();
            for (name, buf) in [
                ("attn_norm", &la.attn_norm), ("attn_qkv", &la.attn_qkv),
                ("attn_gate", &la.attn_gate), ("ssm_out", &la.ssm_out),
                ("ffn_down", &la.ffn_down),
            ] {
                if buf.iter().all(|v| *v == 0.0) { out.push(name); }
            }
            out
        }
        _ => unreachable!(),
    };
    assert!(bad.is_empty(), "all-zero buffers in block 0: {bad:?}");
}
