//! Pipeline parallelism must be invisible: a model split into stages
//! produces the same logits as the whole-model engine, for the batched
//! prefill and for graph-replayed decode. The stages run the same
//! kernels on the same bytes, so "the same" is bit-exact — the only new
//! operation is the peer copy of the hidden activation, which is exact.
//!
//! Runs a 2-stage split on devices `[0, 1]` when two GPUs are visible,
//! otherwise `[0, 0]` (two stages on one device — the handoff is then a
//! same-device copy, which still exercises every seam except PCIe).

use std::path::PathBuf;
use std::sync::Mutex;

use reinstinct_engine::gguf::GgufFile;
use reinstinct_engine::hip;
use reinstinct_engine::model::qwen3_5::Qwen35Model;
use reinstinct_engine::runtime::KernelCache;
use reinstinct_engine::runtime::pipeline::Qwen35Pipeline;

/// HIP graph capture in `Global` mode rejects "unsafe" API calls from
/// *any* thread while a capture is open, so two tests capturing in the
/// same process must not interleave. Serialise them.
static GPU: Mutex<()> = Mutex::new(());

fn fixture_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("REINSTINCT_GGUF_FIXTURE") {
        return Some(PathBuf::from(p));
    }
    let home = std::env::var_os("HOME")?;
    let p = PathBuf::from(home).join("models/qwen-3.5-0.8B/Qwen3.5-0.8B-UD-Q4_K_XL.gguf");
    if p.exists() { Some(p) } else { None }
}

#[test]
fn two_stage_pipeline_matches_single_engine() {
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let Some(p) = fixture_path() else { eprintln!("skipping"); return; };
    let n_dev = match hip::device_count() { Ok(n) if n >= 1 => n, _ => { eprintln!("skip: no HIP"); return; } };
    let devices: Vec<i32> = if n_dev >= 2 { vec![0, 1] } else { vec![0, 0] };

    hip::Device::set(0).unwrap();
    let g = GgufFile::open(&p).expect("open gguf");
    let model = Qwen35Model::load(&g).expect("load model");
    let cache = KernelCache::new().expect("kernel cache");
    let max_seq = 64;
    let prompt: Vec<u32> = vec![151644, 872, 198, 3838, 374, 279, 6722, 315, 9625, 30, 151645, 198];

    // Reference: the whole model on device 0.
    let single = Qwen35Pipeline::new(&model, &g, &cache, max_seq, &[0], None).expect("single");
    let mut s1 = single.new_state(&model, max_seq).expect("state");
    let ref_prefill = single.forward_tokens_batched(&prompt, &mut s1).expect("prefill");
    let g1 = single.capture_forward_graph(&mut s1).expect("graph");
    let mut ref_dec = Vec::new();
    let mut tok = argmax(&ref_prefill);
    for _ in 0..3 {
        let lg = single.forward_token_via_graph(&g1, tok, &mut s1).expect("decode");
        tok = argmax(&lg);
        ref_dec.push(lg);
    }

    // Two stages, planned by weight bytes.
    let pipe = Qwen35Pipeline::new(&model, &g, &cache, max_seq, &devices, None).expect("pipeline");
    let layout = pipe.layout();
    eprintln!("layout: {layout:?}");
    assert_eq!(layout.len(), 2);
    assert_eq!(layout[0].1.start, 0);
    assert_eq!(layout[0].1.end, layout[1].1.start);
    assert_eq!(layout[1].1.end, model.block_kinds.len());
    let mut s2 = pipe.new_state(&model, max_seq).expect("state");
    let got_prefill = pipe.forward_tokens_batched(&prompt, &mut s2).expect("prefill");
    assert_eq!(got_prefill.len(), ref_prefill.len());
    assert!(got_prefill.iter().zip(&ref_prefill).all(|(a, b)| a.to_bits() == b.to_bits()),
            "pipeline prefill logits differ from single-engine (max abs diff {:.3e})",
            max_abs_diff(&got_prefill, &ref_prefill));

    let g2 = pipe.capture_forward_graph(&mut s2).expect("graph");
    let mut tok = argmax(&got_prefill);
    for (i, want) in ref_dec.iter().enumerate() {
        let lg = pipe.forward_token_via_graph(&g2, tok, &mut s2).expect("decode");
        assert!(lg.iter().zip(want).all(|(a, b)| a.to_bits() == b.to_bits()),
                "decode step {i}: pipeline logits differ (max abs diff {:.3e})",
                max_abs_diff(&lg, want));
        tok = argmax(&lg);
    }
    assert_eq!(s2.pos, prompt.len() + 3);

    // An explicit split is honoured, and per-kernel decode agrees too.
    let n = model.block_kinds.len();
    let pipe_x = Qwen35Pipeline::new(&model, &g, &cache, max_seq, &devices, Some(&[1, n - 1]))
        .expect("explicit split");
    assert_eq!(pipe_x.layout()[0].1, 0..1);
    let mut s3 = pipe_x.new_state(&model, max_seq).expect("state");
    let lg = pipe_x.forward_tokens(&prompt, &mut s3).expect("sequential");
    // Sequential (per-token) prefill and batched prefill use different
    // kernels, so compare by argmax rather than bits.
    assert_eq!(argmax(&lg), argmax(&ref_prefill));
}

/// Micro-batched prefill: a prompt long enough to split into several
/// chunks must land on the same logits as the one-shot single engine.
/// Later chunks attend to earlier ones through the KV cache with a
/// different tiling of the same sums, so compare to a tolerance rather
/// than bits, and require the same argmax through a few decode steps.
#[test]
fn micro_batched_prefill_matches_one_shot() {
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let Some(p) = fixture_path() else { eprintln!("skipping"); return; };
    let n_dev = match hip::device_count() { Ok(n) if n >= 1 => n, _ => { eprintln!("skip: no HIP"); return; } };
    let devices: Vec<i32> = if n_dev >= 2 { vec![0, 1] } else { vec![0, 0] };

    hip::Device::set(0).unwrap();
    let g = GgufFile::open(&p).expect("open gguf");
    let model = Qwen35Model::load(&g).expect("load model");
    let cache = KernelCache::new().expect("kernel cache");
    let max_seq = 512;
    // 300 tokens -> chunks of 128, 128, 44.
    let base: Vec<u32> = vec![151644, 872, 198, 3838, 374, 279, 6722, 315, 9625, 30, 11, 323];
    let prompt: Vec<u32> = (0..300).map(|i| base[i % base.len()] + (i as u32 % 7)).collect();

    let single = Qwen35Pipeline::new(&model, &g, &cache, max_seq, &[0], None).expect("single");
    let mut s1 = single.new_state(&model, max_seq).expect("state");
    let ref_lg = single.forward_tokens_batched(&prompt, &mut s1).expect("prefill");

    let pipe = Qwen35Pipeline::new(&model, &g, &cache, max_seq, &devices, None).expect("pipeline");
    let mut s2 = pipe.new_state(&model, max_seq).expect("state");
    let got = pipe.forward_tokens_batched(&prompt, &mut s2).expect("chunked prefill");
    assert_eq!(s2.pos, prompt.len());
    let scale = ref_lg.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let diff = max_abs_diff(&got, &ref_lg);
    eprintln!("chunked prefill: max |Δlogit| = {diff:.3e} (scale {scale:.2})");
    assert!(diff <= 1e-3 * scale, "chunked prefill logits differ: {diff:.3e} vs scale {scale:.2}");
    assert_eq!(argmax(&got), argmax(&ref_lg));

    // The state left behind is a valid mid-sequence state.
    let g1 = single.capture_forward_graph(&mut s1).expect("graph");
    let g2 = pipe.capture_forward_graph(&mut s2).expect("graph");
    let mut tok = argmax(&ref_lg);
    for i in 0..4 {
        let a = single.forward_token_via_graph(&g1, tok, &mut s1).expect("decode");
        let b = pipe.forward_token_via_graph(&g2, tok, &mut s2).expect("decode");
        assert_eq!(argmax(&a), argmax(&b), "decode step {i} diverged after chunked prefill");
        tok = argmax(&a);
    }
}

fn argmax(v: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, x) in v.iter().enumerate() { if *x > v[best] { best = i; } }
    best as u32
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
}
