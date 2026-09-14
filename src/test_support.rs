//! Fixture discovery shared by the GPU / model tests.
//!
//! Every test that needs a GGUF or a GPU skips when it is missing, so a
//! checkout without models passes green vacuously. On a box that *does*
//! have them, set `REINSTINCT_REQUIRE_FIXTURES=1` and a missing fixture
//! or device is a failure instead of a silent skip — the difference
//! between "the golden test passed" and "the golden test never ran".

use std::path::PathBuf;

fn required() -> bool {
    std::env::var_os("REINSTINCT_REQUIRE_FIXTURES").is_some()
}

/// Report a missing fixture: a panic under `REINSTINCT_REQUIRE_FIXTURES`,
/// a "skip:" line otherwise. Returns `None` for the caller to return on.
pub fn missing<T>(what: &str) -> Option<T> {
    if required() {
        panic!("REINSTINCT_REQUIRE_FIXTURES is set and {what} is missing");
    }
    eprintln!("skip: {what}");
    None
}

/// A GGUF fixture named by `env_var`, falling back to `$HOME/<home_rel>`
/// when given. `None` (after reporting) when neither exists.
pub fn gguf_fixture(env_var: &str, home_rel: Option<&str>) -> Option<PathBuf> {
    if let Some(p) = std::env::var_os(env_var).map(PathBuf::from) {
        if p.exists() { return Some(p); }
        return missing(&format!("{env_var}={} does not exist", p.display()));
    }
    if let Some(rel) = home_rel {
        if let Some(home) = std::env::var_os("HOME") {
            let p = PathBuf::from(home).join(rel);
            if p.exists() { return Some(p); }
        }
        return missing(&format!("no {env_var}, and $HOME/{rel} not present"));
    }
    missing(&format!("set {env_var} to a GGUF"))
}

/// Qwen dense fixture — Qwen 3.8 27B by default.
pub fn qwen_fixture() -> Option<PathBuf> {
    gguf_fixture(
        "REINSTINCT_GGUF_FIXTURE",
        Some("models/qwen-3.8-27B/Qwen3.8-27B-UD-Q4_K_XL.gguf"),
    )
}

/// Qwen MoE fixture — Qwen 3.6 35B-A3B.
pub fn qwen_moe_fixture() -> Option<PathBuf> {
    gguf_fixture(
        "REINSTINCT_QWEN_MOE_FIXTURE",
        Some("models/qwen-3.6-35B-A3B/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"),
    )
}

/// Gemma 4 dense fixture — 31B by default.
pub fn gemma_fixture() -> Option<PathBuf> {
    gguf_fixture(
        "REINSTINCT_GEMMA_FIXTURE",
        Some("models/gemma4-31B/gemma-4-31B-it-UD-Q4_K_XL.gguf"),
    )
}

/// Gemma 4 MoE fixture — 26B-A4B.
pub fn gemma_moe_fixture() -> Option<PathBuf> {
    gguf_fixture(
        "REINSTINCT_GEMMA_MOE_FIXTURE",
        Some("models/gemma4-26B/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf"),
    )
}

/// Gemma 4 E4B fixture — small dense model.
pub fn gemma_e4b_fixture() -> Option<PathBuf> {
    gguf_fixture(
        "REINSTINCT_GEMMA_E4B_FIXTURE",
        Some("models/gemma4-E4B/gemma-4-E4B-it-UD-Q4_K_XL.gguf"),
    )
}

/// A DFlash drafter GGUF (`REINSTINCT_DFLASH_FIXTURE`).
pub fn dflash_fixture() -> Option<PathBuf> {
    gguf_fixture(
        "REINSTINCT_DFLASH_FIXTURE",
        Some("models/gemma4-31B/gemma4-31b-it-dflash-Q8_0.gguf"),
    )
}

/// Gemma 4 MTP assistant GGUF.
pub fn gemma4_assistant_fixture() -> Option<PathBuf> {
    gguf_fixture(
        "REINSTINCT_GEMMA4_ASSISTANT_FIXTURE",
        Some("models/gemma4-31B/mtp-gemma-4-31B-it-Q8_0.gguf"),
    )
}

/// `Some(())` when a HIP device is present; reports and `None` otherwise.
pub fn gpu() -> Option<()> {
    if crate::hip::device_count().ok().unwrap_or(0) >= 1 { Some(()) }
    else { missing("no HIP device") }
}

/// A kernel cache, or a reported skip when the toolchain is absent.
pub fn kernel_cache() -> Option<crate::runtime::KernelCache> {
    gpu()?;
    match crate::runtime::KernelCache::new() {
        Ok(c) => Some(c),
        Err(e) => missing(&format!("kernel cache: {e}")),
    }
}
