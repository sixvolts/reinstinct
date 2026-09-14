//! Open a real Qwen UD-Q4_K_XL file and validate that the parser
//! handles a production GGUF end-to-end.
//!
//! Skips when the file is absent so CI / offline dev still passes.
//! Override the path with `REINSTINCT_GGUF_FIXTURE=/abs/path.gguf`.

use std::collections::BTreeMap;
use std::path::PathBuf;

use reinstinct_engine::gguf::{GgmlType, GgufFile};

fn fixture_path() -> Option<PathBuf> {
    reinstinct_engine::test_support::qwen_fixture()
}

#[test]
fn qwen_ud_q4_k_xl_loads() {
    let Some(path) = fixture_path() else {
        eprintln!("skipping: GGUF fixture not found (set REINSTINCT_GGUF_FIXTURE to enable)");
        return;
    };

    let g = GgufFile::open(&path).expect("open GGUF");

    assert_eq!(g.header.version, 3);
    assert!(g.header.tensor_count > 0, "expected tensors");

    let arch = g.metadata_get("general.architecture")
        .and_then(|v| v.as_str())
        .expect("general.architecture present");
    eprintln!("arch        = {arch}");
    eprintln!("tensors     = {}", g.header.tensor_count);
    eprintln!("metadata    = {} kv pairs", g.header.metadata_kv_count);
    eprintln!("alignment   = {}", g.alignment);
    eprintln!("data_offset = {} bytes", g.data_section_offset);

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

    for t in &g.tensors {
        assert!(t.shape().iter().all(|d| *d > 0), "tensor {} has zero dim", t.name);
    }

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

    let first = &g.tensors[0];
    let bytes = g.tensor_data(&first.name).unwrap().expect("first tensor data");
    assert_eq!(bytes.len() as u64, first.byte_size().unwrap());

    let q4k_count = g.tensors.iter().filter(|t| t.ggml_type == GgmlType::Q4_K).count();
    assert!(q4k_count > 0, "UD-Q4_K_XL file should contain Q4_K tensors");
}
