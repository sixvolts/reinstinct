//! Load a Gemma 4 dense GGUF and validate parsed config + tensor inventory.

use std::path::PathBuf;

use reinstinct_engine::gguf::GgufFile;
use reinstinct_engine::model::gemma4::Gemma4Model;

fn fixture_path() -> Option<PathBuf> {
    reinstinct_engine::test_support::gemma_fixture()
}

#[test]
fn loads_gemma4_dense_with_consistent_topology() {
    let Some(path) = fixture_path() else {
        eprintln!("skipping: Gemma fixture not found");
        return;
    };

    let g = GgufFile::open(&path).expect("open GGUF");
    let model = match Gemma4Model::load(&g) {
        Ok(m) => m,
        Err(e) => panic!("Gemma4Model::load failed: {e}"),
    };
    let cfg = &model.config;

    assert!(!cfg.is_moe(), "31B fixture should be dense");
    assert!(cfg.block_count > 0);
    assert!(cfg.hidden_size > 0);
    assert!(cfg.vocab_size > 0);
    assert!(cfg.n_heads > 0);
    assert!(cfg.head_dim_full > 0);
    assert!(cfg.sliding_window > 0);
    assert!(cfg.rope_freq_base > 0.0);
    assert!(cfg.rms_norm_eps > 0.0);

    assert_eq!(cfg.attn_kinds.len(), cfg.block_count as usize);
    assert_eq!(cfg.kv_heads.len(), cfg.block_count as usize);

    eprintln!("loaded Gemma 4 dense: {} blocks, hidden={}, vocab={}, heads={}",
        cfg.block_count, cfg.hidden_size, cfg.vocab_size, cfg.n_heads);
}
