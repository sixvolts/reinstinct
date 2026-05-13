//! Runtime `dlopen` of `libamdhip64.so` and safe Rust wrappers.
//!
//! No link-time ROCm dependency — the engine binary runs against any
//! ROCm 5.7 / 6.x / 7.x runtime that exposes `libamdhip64.so`.
