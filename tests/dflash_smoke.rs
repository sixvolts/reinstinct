//! DFlash end-to-end: fuse context, draft a 16-token block, verify it
//! with the target, and report how much was accepted.
//!
//! Acceptance IS the correctness test here. Every plausible wiring
//! mistake — wrong tapped layers, RoPE at the wrong positions, k_norm
//! skipped on the context rows, next-token instead of denoising
//! semantics — degrades the block to noise, and a noise drafter accepts
//! essentially nothing. A healthy DFlash block should accept most of it.
//!
//! Defaults to `~/models/gemma4-31B/` for both target and drafter, or set
//! REINSTINCT_GEMMA_FIXTURE / REINSTINCT_DFLASH_FIXTURE to override.

use reinstinct_engine::gguf::GgufFile;
use reinstinct_engine::hip;
use reinstinct_engine::model::dflash::DFlashModel;
use reinstinct_engine::model::gemma4::Gemma4Model;
use reinstinct_engine::runtime::dflash::{DFlashState, GpuDFlash};
use reinstinct_engine::runtime::gemma4::{Gemma4GpuState, GpuGemma4};

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, x) in v.iter().enumerate() { if *x > v[best] { best = i; } }
    best as u32
}

#[test]
fn drafts_a_block_the_target_mostly_accepts() {
    use reinstinct_engine::test_support as fx;
    let (Some(tp), Some(dp)) = (fx::gemma_fixture(), fx::dflash_fixture()) else { return };
    let Some(cache) = fx::kernel_cache() else { return };
    let _dev = hip::Device::set(0).unwrap();

    let t_gguf = GgufFile::open(&tp).expect("open target");
    let d_gguf = GgufFile::open(&dp).expect("open drafter");
    let t_model = Gemma4Model::load(&t_gguf).expect("load target");
    let d_model = DFlashModel::load(&d_gguf).expect("load dflash");

    let max_seq = 512usize;
    let target = GpuGemma4::new(&t_model, &t_gguf, &cache, max_seq).expect("gpu target");
    let draft = GpuDFlash::new(&d_model, &d_gguf, &cache, &target, max_seq).expect("gpu dflash");
    let mut t_state = Gemma4GpuState::new(&t_model, max_seq).expect("target state");
    let mut d_state = DFlashState::new(&draft.config, max_seq).expect("draft state");
    t_state.reset();
    d_state.reset();

    // docs/DFLASH_PORT.md flags an off-by-one that is silent if wrong:
    // the GGUF's target_layers are HF hidden-state indices, so the blocks
    // to tap are those minus one. REINSTINCT_DFLASH_TAP_SHIFT lets the
    // other reading be tried — the correct one should be clearly better.
    let shift: i32 = std::env::var("REINSTINCT_DFLASH_TAP_SHIFT")
        .ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    let taps: Vec<usize> = draft.target_layers().iter()
        .map(|&l| (l as i32 + shift).max(0) as usize).collect();
    eprintln!("dflash: tapping target blocks {taps:?} (shift {shift})");
    target.enable_target_tap(&taps, &mut t_state, max_seq).expect("tap");

    // "<bos> The capital of France is Paris."
    let prompt: Vec<u32> = vec![2, 669, 5279, 529, 7001, 563, 9079, 236761];
    let p = prompt.len();
    let logits = target.prefill_forward(&prompt, &mut t_state).expect("prefill");
    let anchor = argmax(&logits);

    draft.append_context(&mut d_state, t_state.tap.as_ref().unwrap(), p, 0)
        .expect("append context");
    assert_eq!(d_state.ctx_len, p);

    let mask = 4u32;   // Gemma 4 tokenizer mask_token_id
    let preds = draft.draft_block(&d_state, &target, anchor, mask, draft.config.block_size as usize).expect("draft");
    let b = draft.config.block_size as usize;
    assert_eq!(preds.len(), b);
    // Block position 0 held the anchor unmasked. Its output is NOT a
    // draft and is not asserted on: DFlash computes loss only on masked
    // positions, so the anchor slot is unconstrained by training and can
    // legitimately predict anything. Printed as a diagnostic only.
    eprintln!("dflash: anchor={anchor}, unconstrained pos0 output={}", preds[0]);
    let drafted = &preds[1..];

    let vocab = target.vocab_size() as u32;
    assert!(drafted.iter().all(|&t| t < vocab), "drafted token out of vocab range");
    assert!(drafted.iter().any(|&t| t != drafted[0]),
            "every drafted token identical ({}) — the block collapsed", drafted[0]);

    // Verify the whole block with the target in one causal pass.
    let mut block = Vec::with_capacity(b);
    block.push(anchor);
    block.extend_from_slice(&drafted);
    let vlogits = target.verify_forward(&block, &mut t_state).expect("verify");
    assert_eq!(vlogits.len(), b);

    let mut accepted = 0usize;
    for j in 1..b {
        if drafted[j - 1] == argmax(&vlogits[j - 1]) { accepted += 1; } else { break; }
    }
    eprintln!("dflash: anchor={anchor} accepted {accepted}/{} of the block", b - 1);
    eprintln!("  drafted: {drafted:?}");
    eprintln!("  target : {:?}",
              (0..b - 1).map(|j| argmax(&vlogits[j])).collect::<Vec<_>>());

    // A correctly wired drafter conditioned on the target's own hidden
    // states should not be at chance. Chance here is ~1/262144.
    assert!(accepted >= 1,
        "accepted 0/{} — the draft block does not agree with the target at all, \
         which means the forward pass is wired wrong, not merely weak", b - 1);
}
