//! Load a real Qwen 3.x GGUF as a typed Qwen35Model and validate that
//! the parsed config and block schedule are self-consistent.
//!
//! Skipped when the file is absent (set REINSTINCT_GGUF_FIXTURE to override).

use std::path::PathBuf;

use reinstinct_engine::gguf::GgufFile;
use reinstinct_engine::model::qwen3_5::{BlockKind, Qwen35Model};

fn fixture_path() -> Option<PathBuf> {
    reinstinct_engine::test_support::qwen_fixture()
}

#[test]
fn loads_qwen_dense_with_consistent_topology() {
    let Some(path) = fixture_path() else {
        eprintln!("skipping: GGUF fixture not found");
        return;
    };

    let g = GgufFile::open(&path).expect("open GGUF");
    let model = match Qwen35Model::load(&g) {
        Ok(m) => m,
        Err(e) => panic!("Qwen35Model::load failed: {e}"),
    };
    let cfg = &model.config;

    assert!(cfg.block_count > 0);
    assert!(cfg.hidden_size > 0);
    assert!(cfg.vocab_size > 0);
    assert!(cfg.attn_n_heads > 0);
    assert!(cfg.attn_n_kv_heads > 0);
    assert!(cfg.attn_n_heads % cfg.attn_n_kv_heads == 0);
    assert!(cfg.rope_freq_base > 0.0);
    assert!(cfg.rms_norm_eps > 0.0);

    let n_main = cfg.n_main_blocks() as usize;
    assert_eq!(model.block_kinds.len(), n_main);

    let n_full = model.block_kinds.iter()
        .filter(|k| **k == BlockKind::FullAttention).count();
    let n_linear = model.block_kinds.iter()
        .filter(|k| **k == BlockKind::LinearAttention).count();
    assert_eq!(n_full + n_linear, n_main);
    assert!(n_full > 0, "expected at least one full-attention block");

    if cfg.full_attention_interval > 0 {
        for (i, kind) in model.block_kinds.iter().enumerate() {
            let expected_full = (i as u32 + 1) % cfg.full_attention_interval == 0;
            if expected_full {
                assert_eq!(*kind, BlockKind::FullAttention,
                    "block {i} should be full attention (interval={})", cfg.full_attention_interval);
            } else {
                assert_eq!(*kind, BlockKind::LinearAttention,
                    "block {i} should be linear attention");
            }
        }
    }

    assert!(!cfg.is_moe(), "dense fixture should not be MoE");
    assert!(cfg.ffn_size > 0, "dense model should have ffn_size > 0");

    eprintln!("loaded: {} blocks ({} main + {} nextn), hidden={}, vocab={}, heads={}/{}",
        cfg.block_count, n_main, cfg.nextn_predict_layers,
        cfg.hidden_size, cfg.vocab_size,
        cfg.attn_n_heads, cfg.attn_n_kv_heads);
    eprintln!("block schedule: {} full, {} linear", n_full, n_linear);
}

#[test]
fn rejects_non_qwen35_architecture() {
    use std::io::Write;
    use memmap2::MmapMut;

    const GGUF_MAGIC: u32 = 0x4655_4747;
    let mut buf = Vec::<u8>::new();
    buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    buf.extend_from_slice(&3u32.to_le_bytes());     // version
    buf.extend_from_slice(&0u64.to_le_bytes());     // tensor_count
    buf.extend_from_slice(&1u64.to_le_bytes());     // metadata_kv_count
    let key = "general.architecture";
    buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
    buf.extend_from_slice(key.as_bytes());
    buf.extend_from_slice(&8u32.to_le_bytes());     // value_type = String
    let val = "llama";
    buf.extend_from_slice(&(val.len() as u64).to_le_bytes());
    buf.extend_from_slice(val.as_bytes());
    while buf.len() % 32 != 0 { buf.push(0); }

    let mut mmap = MmapMut::map_anon(buf.len()).unwrap();
    (&mut mmap[..]).write_all(&buf).unwrap();
    let mmap = mmap.make_read_only().unwrap();
    let g = GgufFile::from_mmap(mmap).expect("parse synthetic gguf");

    use reinstinct_engine::model::qwen3_5::Qwen35Error;
    match Qwen35Model::load(&g) {
        Ok(_) => panic!("expected WrongArchitecture error"),
        Err(Qwen35Error::WrongArchitecture { got, .. }) => assert_eq!(got, "llama"),
        Err(e) => panic!("expected WrongArchitecture, got {e}"),
    }
}
