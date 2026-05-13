//! Dequantize one real tensor of each type appearing in the Qwen 0.8B
//! UD-Q4_K_XL file and sanity-check the output. This is the first end-to-end
//! exercise of the quant module against production data.
//!
//! Skipped when the file is absent (set REINSTINCT_GGUF_FIXTURE to override).

use std::collections::HashSet;
use std::path::PathBuf;

use reinstinct_engine::gguf::{GgmlType, GgufFile};
use reinstinct_engine::quant::dequantize_tensor;

fn fixture_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("REINSTINCT_GGUF_FIXTURE") {
        return Some(PathBuf::from(p));
    }
    let home = std::env::var_os("HOME")?;
    let p = PathBuf::from(home).join("models/qwen-3.5-0.8B/Qwen3.5-0.8B-UD-Q4_K_XL.gguf");
    p.exists().then_some(p)
}

#[test]
fn dequantize_one_tensor_per_type() {
    let Some(path) = fixture_path() else {
        eprintln!("skipping: GGUF fixture not found");
        return;
    };

    let g = GgufFile::open(&path).expect("open Qwen 0.8B");

    let mut seen: HashSet<GgmlType> = HashSet::new();
    let mut tested = 0;

    for t in &g.tensors {
        if !seen.insert(t.ggml_type) {
            continue;
        }
        let bytes = g.tensor_data(&t.name).expect("tensor_data").expect("present");
        let values = dequantize_tensor(t, bytes)
            .unwrap_or_else(|e| panic!("dequant {} ({:?}): {e}", t.name, t.ggml_type));

        assert_eq!(values.len() as u64, t.n_elements(),
            "tensor {} ({:?}): output len {} != n_elements {}",
            t.name, t.ggml_type, values.len(), t.n_elements());

        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let mut sum_sq = 0.0_f64;
        let mut nan_or_inf = 0usize;
        for &v in &values {
            if !v.is_finite() {
                nan_or_inf += 1;
                continue;
            }
            min = min.min(v);
            max = max.max(v);
            sum_sq += (v as f64) * (v as f64);
        }
        let rms = (sum_sq / values.len() as f64).sqrt();
        eprintln!(
            "{:6} {:50} shape={:?} min={:.4} max={:.4} rms={:.4} bad={nan_or_inf}",
            format!("{:?}", t.ggml_type), t.name, t.shape(), min, max, rms,
        );
        assert_eq!(nan_or_inf, 0,
            "tensor {} ({:?}) contains {} non-finite values",
            t.name, t.ggml_type, nan_or_inf);
        // Loose sanity: RMS within [1e-6, 1e3] for any production weight tensor.
        assert!(rms > 1e-6 && rms < 1e3,
            "tensor {} ({:?}) rms {} outside reasonable range",
            t.name, t.ggml_type, rms);
        tested += 1;
    }

    // Make sure we exercised the ones we care about for Qwen 3.5.
    for required in [
        GgmlType::F32, GgmlType::F16,
        GgmlType::Q4_K, GgmlType::Q5_K, GgmlType::Q6_K, GgmlType::Q8_0,
        GgmlType::IQ4_XS,
    ] {
        assert!(seen.contains(&required),
            "expected {:?} to appear in Qwen 0.8B, but no tensor of that type was found",
            required);
    }

    assert_eq!(tested, seen.len());
    eprintln!("dequantized one tensor of each type: {} types total", tested);
}
