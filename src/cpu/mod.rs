//! Pure-Rust CPU forward-pass oracle.
//!
//! These reference implementations are the correctness oracle for the HIP
//! kernels in `kernels/`. They prioritize clarity over speed: every op is
//! a standalone function, all math in f32, no SIMD intrinsics, no parallelism.
//! When CPU logits match HF transformers, the model is *understood* and we
//! can swap in HIP kernels piece by piece against the same oracle.

pub mod conv1d;
pub mod ops;
pub mod qwen3_5;
pub mod rope;
