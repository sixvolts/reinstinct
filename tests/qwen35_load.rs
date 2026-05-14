//! Load real Qwen 3.5 0.8B GGUF as a typed Qwen35Model and validate that
//! the parsed config + block schedule match the upstream HF config.json.
//!
//! Skipped when the file is absent (set REINSTINCT_GGUF_FIXTURE to override).

use std::path::PathBuf;

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
fn loads_qwen3_5_0_8b_with_expected_topology() {
    let Some(path) = fixture_path() else {
        eprintln!("skipping: GGUF fixture not found");
        return;
    };

    let g = GgufFile::open(&path).expect("open Qwen 0.8B");
    let model = match Qwen35Model::load(&g) {
        Ok(m) => m,
        Err(e) => panic!("Qwen35Model::load failed: {e}"),
    };
    let cfg = &model.config;

    // Values pinned to the upstream HuggingFace config.json for Qwen3.5-0.8B.
    assert_eq!(cfg.block_count, 24);
    assert_eq!(cfg.hidden_size, 1024);
    assert_eq!(cfg.ffn_size, 3584);
    assert_eq!(cfg.vocab_size, 248320);
    assert_eq!(cfg.context_length, 262144);
    assert_eq!(cfg.attn_n_heads, 8);
    assert_eq!(cfg.attn_n_kv_heads, 2);
    assert_eq!(cfg.attn_head_dim, 256);
    assert_eq!(cfg.gdn_value_dim, 2048);
    assert_eq!(cfg.gdn_n_heads, 16);
    assert_eq!(cfg.gdn_head_dim, 128);
    assert_eq!(cfg.gdn_conv_kernel, 4);
    assert_eq!(cfg.rope_dim_count, 64);
    assert_eq!(cfg.full_attention_interval, 4);
    assert!((cfg.rope_freq_base - 10_000_000.0).abs() < 1.0);
    assert!((cfg.rms_norm_eps - 1e-6).abs() < 1e-9);
    assert!(cfg.tied_embeddings, "embeddings should be tied (no `output.weight`)");

    // Layer schedule: L,L,L,F repeated 6 times.
    assert_eq!(model.block_kinds.len(), 24);
    let n_full = model.block_kinds.iter()
        .filter(|k| **k == BlockKind::FullAttention).count();
    assert_eq!(n_full, 6);
    for &i in &[3u32, 7, 11, 15, 19, 23] {
        assert_eq!(model.block_kinds[i as usize], BlockKind::FullAttention,
            "block {i} should be full attention");
    }
    for &i in &[0u32, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22] {
        assert_eq!(model.block_kinds[i as usize], BlockKind::LinearAttention,
            "block {i} should be linear attention");
    }

    eprintln!("loaded {:#?}", cfg);
    eprintln!("block schedule: {:?}", model.block_kinds);
}

#[test]
fn rejects_non_qwen35_architecture() {
    use std::io::Write;
    use memmap2::MmapMut;

    // Synthesize a minimal GGUF whose architecture is "llama" instead of "qwen35".
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
    // Pad to alignment 32 (no tensors so no data section needed, but harmless).
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
