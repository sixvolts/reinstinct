//! DFlash GGUF parsing against the real checkpoint.
//!
//! Defaults to `~/models/gemma4-31B/`, or set `REINSTINCT_DFLASH_FIXTURE`
//! to override. Skips without it, like the other fixture-backed tests.

use reinstinct_engine::gguf::GgufFile;
use reinstinct_engine::model::dflash::{DFlashAttn, DFlashModel};

fn fixture() -> Option<std::path::PathBuf> {
    reinstinct_engine::test_support::dflash_fixture()
}

#[test]
fn loads_gemma4_31b_dflash_with_expected_topology() {
    let Some(path) = fixture() else {
        eprintln!("skip: DFlash GGUF not found");
        return;
    };
    let gguf = GgufFile::open(&path).expect("open");
    let m = DFlashModel::load(&gguf).expect("load dflash");
    let c = &m.config;

    assert_eq!(c.block_count, 5);
    assert_eq!(c.hidden_size, 5376, "draft width == target hidden");
    assert_eq!(c.ffn_size, 10752);
    assert_eq!(c.n_heads, 64);
    assert_eq!(c.n_kv_heads, 8);
    assert_eq!(c.head_dim, 128);
    assert_eq!(c.block_size, 16);
    assert_eq!(c.sliding_window, 2048);

    // Four sliding-causal layers then one unmasked — the asymmetry the
    // attention kernels have to honour.
    assert_eq!(c.attn_kinds, vec![
        DFlashAttn::SlidingCausal, DFlashAttn::SlidingCausal,
        DFlashAttn::SlidingCausal, DFlashAttn::SlidingCausal,
        DFlashAttn::FullBidirectional,
    ]);

    // The off-by-one: the GGUF holds HF hidden-state indices
    // [2,13,24,36,47,58]; upstream's target_layer_ids are [1,12,23,35,46,57]
    // and those are the block indices we must tap.
    assert_eq!(c.target_layers, vec![1, 12, 23, 35, 46, 57],
               "target_layers must be converted from hidden-state to block indices");

    // fc consumes six concatenated target hidden states.
    assert_eq!(c.fused_ctx_width(), 6 * 5376);

    // Pairing check against a 60-block target succeeds; a 24-block one
    // must not.
    assert!(c.validate_against_target(60).is_ok());
    assert!(c.validate_against_target(24).is_err(),
            "block 57 cannot be tapped from a 24-block target");
}

#[test]
fn rejects_a_non_dflash_file() {
    // Any non-DFlash GGUF will do; reuse the Gemma fixture when present.
    let Some(p) = reinstinct_engine::test_support::gemma_fixture() else { return };
    let gguf = GgufFile::open(&p).expect("open");
    assert!(DFlashModel::load(&gguf).is_err(),
            "a gemma4 file must not parse as dflash");
}
