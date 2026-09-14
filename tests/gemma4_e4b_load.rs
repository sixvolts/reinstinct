//! Load a Gemma 4 E4B GGUF and validate parsed config.

use std::path::PathBuf;

use reinstinct_engine::gguf::GgufFile;
use reinstinct_engine::model::gemma4::Gemma4Model;

fn fixture_path() -> Option<PathBuf> {
    reinstinct_engine::test_support::gemma_e4b_fixture()
}

#[test]
fn loads_gemma4_e4b_with_consistent_topology() {
    let Some(path) = fixture_path() else {
        eprintln!("skipping: Gemma E4B fixture not found");
        return;
    };

    let g = GgufFile::open(&path).expect("open GGUF");
    let model = match Gemma4Model::load(&g) {
        Ok(m) => m,
        Err(e) => panic!("Gemma4Model::load failed: {e}"),
    };
    let cfg = &model.config;

    assert!(!cfg.is_moe(), "E4B fixture should be dense");
    assert!(cfg.block_count > 0);
    assert!(cfg.hidden_size > 0);
    assert!(cfg.vocab_size > 0);

    assert_eq!(cfg.attn_kinds.len(), cfg.block_count as usize);

    eprintln!("loaded Gemma 4 E4B: {} blocks, hidden={}, vocab={}",
        cfg.block_count, cfg.hidden_size, cfg.vocab_size);
}
