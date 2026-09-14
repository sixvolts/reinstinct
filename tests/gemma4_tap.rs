//! DFlash target-context tap: does `enable_target_tap` capture the layer
//! outputs it claims to?
//!
//! Needs a real Gemma 4 GGUF — defaults to `~/models/gemma4-31B/`, or
//! set `REINSTINCT_GEMMA_FIXTURE` to override.  Skips without it or
//! without a GPU, matching the rest of the GPU test suite.

use reinstinct_engine::model::gemma4::Gemma4Model;
use reinstinct_engine::runtime::gemma4::{Gemma4GpuState, GpuGemma4};
use reinstinct_engine::hip;
use reinstinct_engine::gguf::GgufFile;

fn fixture() -> Option<std::path::PathBuf> {
    reinstinct_engine::test_support::gemma_fixture()
}

/// Tapping the LAST block must reproduce `last_prenorm_hidden()` exactly:
/// both are the final block's output at the last position, arrived at by
/// different routes (the tap kernel vs. the verify path's own D2D copy).
/// Any slot/stride mistake in the gather shows up here immediately.
#[test]
fn tap_of_final_block_matches_prenorm_hidden() {
    let Some(path) = fixture() else {
        eprintln!("skip: Gemma 4 GGUF not found");
        return;
    };
    let Some(cache) = reinstinct_engine::test_support::kernel_cache() else { return };
    let _dev = hip::Device::set(0).unwrap();

    let gguf = GgufFile::open(&path).expect("open gguf");
    let model = Gemma4Model::load(&gguf).expect("load gemma4");
    let max_seq = 256usize;
    let gm = GpuGemma4::new(&model, &gguf, &cache, max_seq).expect("gpu gemma4");
    let mut state = Gemma4GpuState::new(&model, max_seq).expect("state");
    state.reset();

    let h = gm.hidden_size();
    let last_block = gm.block_count() - 1;
    // Two taps, so the test also covers a non-zero slot and a stride > hidden.
    let taps = [0usize, last_block];
    gm.enable_target_tap(&taps, &mut state, max_seq).expect("enable tap");
    let stride = taps.len() * h;
    assert_eq!(state.tap_stride, stride);

    let tokens: Vec<u32> = vec![2, 669, 5279, 529, 7001, 563, 9079, 236761];
    let p = tokens.len();
    gm.prefill_forward(&tokens, &mut state).expect("prefill");
    // The verify path is the one that maintains hidden_a, so drive it.
    gm.verify_forward(&tokens, &mut state).expect("verify");

    let mut tap_host = vec![0.0f32; max_seq * stride];
    state.tap.as_ref().unwrap().copy_to_host(&mut tap_host).expect("tap readback");
    let mut prenorm = vec![0.0f32; h];
    gm.last_prenorm_hidden().copy_to_host(&mut prenorm).expect("hidden_a readback");

    // slot 1 == last block, row p-1 == last position.
    let base = (p - 1) * stride + 1 * h;
    let tapped = &tap_host[base..base + h];
    let mismatches = tapped.iter().zip(&prenorm).filter(|(a, b)| a != b).count();
    assert_eq!(mismatches, 0,
        "tap of block {last_block} differs from last_prenorm_hidden in {mismatches}/{h} elements");

    // Slot 0 (block 0) must be populated and must NOT equal the last
    // block — otherwise a stride bug could make every slot alias.
    let slot0 = &tap_host[(p - 1) * stride..(p - 1) * stride + h];
    assert!(slot0.iter().any(|v| *v != 0.0), "tap slot 0 is all zeros");
    assert!(slot0 != tapped, "tap slots 0 and 1 alias — stride is wrong");
}
