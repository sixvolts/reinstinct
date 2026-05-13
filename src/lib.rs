//! Custom HIP inference engine for AMD MI50/MI60 (gfx906).
//!
//! See `gfx906-inference-engine-design.md` at the repo root for the
//! architecture and rationale behind the module layout below.

pub mod gguf;
pub mod hip;
pub mod model;
pub mod quant;
pub mod runtime;
