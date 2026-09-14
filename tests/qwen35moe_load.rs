//! Load a Qwen 3.6 MoE GGUF and validate parsed config + block schedule.

use std::path::PathBuf;

use reinstinct_engine::gguf::GgufFile;
use reinstinct_engine::model::qwen3_5::{BlockKind, Qwen35Model};

fn fixture_path() -> Option<PathBuf> {
    reinstinct_engine::test_support::qwen_moe_fixture()
}

#[test]
fn loads_qwen_moe_with_consistent_topology() {
    let Some(path) = fixture_path() else {
        eprintln!("skipping: Qwen MoE fixture not found");
        return;
    };

    let g = GgufFile::open(&path).expect("open GGUF");
    let model = match Qwen35Model::load(&g) {
        Ok(m) => m,
        Err(e) => panic!("Qwen35Model::load failed: {e}"),
    };
    let cfg = &model.config;

    assert_eq!(cfg.arch, "qwen35moe");
    assert!(cfg.is_moe());
    let moe = cfg.moe.as_ref().expect("MoE config present");
    assert!(moe.n_expert > 0);
    assert!(moe.n_expert_used > 0);
    assert!(moe.n_expert_used <= moe.n_expert);
    assert!(moe.expert_ff > 0);
    assert!(moe.shared_expert_ff > 0);

    assert!(cfg.block_count > 0);
    assert!(cfg.hidden_size > 0);
    assert!(cfg.vocab_size > 0);

    assert_eq!(model.block_kinds.len(), cfg.block_count as usize);
    let n_full = model.block_kinds.iter()
        .filter(|k| **k == BlockKind::FullAttention).count();
    let n_linear = model.block_kinds.iter()
        .filter(|k| **k == BlockKind::LinearAttention).count();
    assert_eq!(n_full + n_linear, cfg.block_count as usize);
    assert!(n_full > 0);

    eprintln!("loaded MoE: {} blocks, hidden={}, vocab={}, experts={}/{}",
        cfg.block_count, cfg.hidden_size, cfg.vocab_size,
        moe.n_expert_used, moe.n_expert);
}
