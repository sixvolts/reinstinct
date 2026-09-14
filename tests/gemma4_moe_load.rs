//! Load a Gemma 4 MoE (26B-A4B) GGUF and validate parsed config.

use std::path::PathBuf;

use reinstinct_engine::gguf::GgufFile;
use reinstinct_engine::model::gemma4::Gemma4Model;

fn fixture_path() -> Option<PathBuf> {
    reinstinct_engine::test_support::gemma_moe_fixture()
}

#[test]
fn loads_gemma4_moe_with_consistent_topology() {
    let Some(path) = fixture_path() else {
        eprintln!("skipping: Gemma MoE fixture not found");
        return;
    };

    let g = GgufFile::open(&path).expect("open GGUF");
    let model = match Gemma4Model::load(&g) {
        Ok(m) => m,
        Err(e) => panic!("Gemma4Model::load failed: {e}"),
    };
    let cfg = &model.config;

    assert!(cfg.is_moe(), "26B-A4B fixture should be MoE");
    assert!(cfg.expert_count > 0);
    assert!(cfg.expert_used_count > 0);
    assert!(cfg.expert_used_count <= cfg.expert_count);
    assert!(cfg.expert_ff_size > 0);

    assert!(cfg.block_count > 0);
    assert!(cfg.hidden_size > 0);
    assert!(cfg.vocab_size > 0);

    assert_eq!(cfg.attn_kinds.len(), cfg.block_count as usize);
    assert_eq!(cfg.kv_heads.len(), cfg.block_count as usize);

    eprintln!("loaded Gemma 4 MoE: {} blocks, hidden={}, vocab={}, experts={}/{}",
        cfg.block_count, cfg.hidden_size, cfg.vocab_size,
        cfg.expert_used_count, cfg.expert_count);
}
