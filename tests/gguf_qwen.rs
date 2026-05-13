//! Open the real Qwen 3.5 0.8B UD-Q4_K_XL file and validate that the
//! parser handles a production GGUF end-to-end.
//!
//! Skips when the file is absent so CI / offline dev still passes.
//! Override the path with `REINSTINCT_GGUF_FIXTURE=/abs/path.gguf`.

use std::collections::BTreeMap;
use std::path::PathBuf;

use reinstinct_engine::gguf::{GgmlType, GgufFile};

fn fixture_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("REINSTINCT_GGUF_FIXTURE") {
        return Some(PathBuf::from(p));
    }
    let home = std::env::var_os("HOME")?;
    let p = PathBuf::from(home).join("models/qwen-3.5-0.8B/Qwen3.5-0.8B-UD-Q4_K_XL.gguf");
    p.exists().then_some(p)
}

#[test]
fn qwen_3_5_0_8b_ud_q4_k_xl_loads() {
    let Some(path) = fixture_path() else {
        eprintln!("skipping: GGUF fixture not found (set REINSTINCT_GGUF_FIXTURE to enable)");
        return;
    };

    let g = GgufFile::open(&path).expect("open Qwen 0.8B");

    assert_eq!(g.header.version, 3);
    assert!(g.header.tensor_count > 0, "expected tensors in Qwen 0.8B");

    // Architecture should announce itself.
    let arch = g.metadata_get("general.architecture")
        .and_then(|v| v.as_str())
        .expect("general.architecture present");
    eprintln!("arch        = {arch}");
    eprintln!("tensors     = {}", g.header.tensor_count);
    eprintln!("metadata    = {} kv pairs", g.header.metadata_kv_count);
    eprintln!("alignment   = {}", g.alignment);
    eprintln!("data_offset = {} bytes", g.data_section_offset);

    // Type histogram — confirms which dequant kernels we'll need to ship
    // for this file. UD-Q4_K_XL files mix Q4_K, Q5_K, Q6_K, Q8_0, plus
    // F16 / F32 / BF16 for embeddings + norms.
    let mut hist: BTreeMap<String, (u64, u64)> = BTreeMap::new();
    for t in &g.tensors {
        let bytes = t.byte_size().unwrap();
        let key = format!("{:?}", t.ggml_type);
        let e = hist.entry(key).or_default();
        e.0 += 1;
        e.1 += bytes;
    }
    eprintln!("--- tensor type histogram ---");
    for (k, (count, bytes)) in &hist {
        eprintln!("  {k:6} {count:5} tensors  {:>10} MB", bytes / (1024 * 1024));
    }

    // Sanity: every tensor type that appears must have a known on-disk size,
    // which is implied by GgmlType already; just confirm shape lookup works.
    for t in &g.tensors {
        assert!(t.shape().iter().all(|d| *d > 0), "tensor {} has zero dim", t.name);
    }

    // The token embedding is conventionally one of the largest tensors.
    let embed_names = [
        "token_embd.weight",
        "tok_embeddings.weight",
        "model.embed_tokens.weight",
    ];
    let embed = embed_names.iter().find_map(|n| g.tensor(n));
    if let Some(t) = embed {
        eprintln!("embedding   = {} {:?} shape {:?}", t.name, t.ggml_type, t.shape());
    } else {
        eprintln!("note: no canonical embedding tensor name found");
    }

    // Spot-check: zero-copy data slice for the first tensor matches its
    // computed byte_size and lives entirely within the mmap.
    let first = &g.tensors[0];
    let bytes = g.tensor_data(&first.name).unwrap().expect("first tensor data");
    assert_eq!(bytes.len() as u64, first.byte_size().unwrap());

    // At minimum we expect Q4_K to dominate a UD-Q4_K_XL file.
    let q4k_count = g.tensors.iter().filter(|t| t.ggml_type == GgmlType::Q4_K).count();
    assert!(q4k_count > 0, "UD-Q4_K_XL file should contain Q4_K tensors");
}
