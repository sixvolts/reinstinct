//! Gemma 4: typed config + per-layer schedule for the GQA + alternating
//! sliding-window architecture.
//!
//! Unlike Qwen 3.5 (one repeating L/F pattern) Gemma 4 carries two
//! per-layer arrays in GGUF metadata:
//!   - `gemma4.attention.head_count_kv`        — KV head count per layer
//!   - `gemma4.attention.sliding_window_pattern` — bool per layer
//!
//! A layer is *sliding-window* when its pattern bool is true and
//! *full-attention* when false (the reference 31B runs 5 SWA then 1
//! full, repeating). The two attention kinds differ in head dim, RoPE
//! frequency base, and rotated-dim count:
//!
//!   kind     head_dim  rope_freq_base  rope_dim
//!   sliding    256        1e4            256
//!   full       512        1e6            512
//!
//! Per-block tensors (sandwich norms — Gemma norms attention and FFN on
//! both sides): attn_norm, attn_{q,k,v,q_norm,k_norm,output},
//! post_attention_norm, ffn_norm, ffn_{gate,up,down}, post_ffw_norm,
//! layer_output_scale.
//!
//! This module only parses + validates; the forward path is built on top.

use thiserror::Error;

use crate::gguf::{GgufFile, MetaValue};

const ARCH: &str = "gemma4";

#[derive(Debug, Error)]
pub enum Gemma4Error {
    #[error("not a Gemma 4 file: general.architecture = {got:?}, expected {expected:?}")]
    WrongArchitecture { got: String, expected: &'static str },

    #[error("missing required GGUF metadata key: {0}")]
    MissingMetadata(&'static str),

    #[error("metadata key {key} has wrong type (expected {expected})")]
    WrongMetadataType { key: &'static str, expected: &'static str },

    #[error("metadata array {key} has {got} entries, expected {expected}")]
    WrongArrayLength { key: &'static str, got: usize, expected: usize },

    #[error("missing required tensor: {0}")]
    MissingTensor(String),
}

type Result<T> = std::result::Result<T, Gemma4Error>;

/// Attention kind for a single layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AttnKind {
    /// Local sliding-window attention (256 head dim, 1e4 RoPE base).
    Sliding,
    /// Global full attention (512 head dim, 1e6 RoPE base).
    Full,
}

/// Gemma 4 hyperparameters from `gemma4.*` GGUF metadata.
#[derive(Debug, Clone)]
pub struct Gemma4Config {
    pub block_count: u32,
    pub hidden_size: u32,
    pub ffn_size: u32,
    pub vocab_size: u32,
    pub context_length: u32,
    pub rms_norm_eps: f32,
    pub eos_token_id: u32,

    /// Query head count — constant across layers.
    pub n_heads: u32,
    pub head_dim_full: u32,
    pub head_dim_swa: u32,
    pub sliding_window: u32,

    pub rope_freq_base: f32,
    pub rope_freq_base_swa: f32,
    pub rope_dim_full: u32,
    pub rope_dim_swa: u32,

    /// Final-logit soft-cap value (`logits = cap·tanh(logits/cap)`).
    pub final_logit_softcapping: f32,

    pub tied_embeddings: bool,

    /// KV head count per layer (GQA group size varies by layer).
    pub kv_heads: Vec<u32>,
    /// Attention kind per layer.
    pub attn_kinds: Vec<AttnKind>,

    /// MoE: routed expert count (0 ⇒ dense model, e.g. the 31B).
    pub expert_count: u32,
    /// MoE: experts activated per token (top-k of `expert_count`).
    pub expert_used_count: u32,
    /// MoE: per-expert FFN intermediate size.
    pub expert_ff_size: u32,
}

impl Gemma4Config {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let arch = require_str(gguf, "general.architecture")?;
        if arch != ARCH {
            return Err(Gemma4Error::WrongArchitecture { got: arch.to_owned(), expected: ARCH });
        }

        let block_count    = require_u32(gguf, "gemma4.block_count")?;
        let hidden_size    = require_u32(gguf, "gemma4.embedding_length")?;
        let ffn_size       = require_u32(gguf, "gemma4.feed_forward_length")?;
        let context_length = require_u32(gguf, "gemma4.context_length")?;
        let rms_norm_eps   = require_f32(gguf, "gemma4.attention.layer_norm_rms_epsilon")?;
        let n_heads        = require_u32(gguf, "gemma4.attention.head_count")?;
        let head_dim_full  = require_u32(gguf, "gemma4.attention.key_length")?;
        let head_dim_swa   = require_u32(gguf, "gemma4.attention.key_length_swa")?;
        let sliding_window = require_u32(gguf, "gemma4.attention.sliding_window")?;
        let rope_freq_base     = require_f32(gguf, "gemma4.rope.freq_base")?;
        let rope_freq_base_swa = require_f32(gguf, "gemma4.rope.freq_base_swa")?;
        let rope_dim_full  = require_u32(gguf, "gemma4.rope.dimension_count")?;
        let rope_dim_swa   = require_u32(gguf, "gemma4.rope.dimension_count_swa")?;
        let final_logit_softcapping = require_f32(gguf, "gemma4.final_logit_softcapping")?;
        let eos_token_id   = require_u32(gguf, "tokenizer.ggml.eos_token_id")?;

        // Per-layer KV head counts.
        let kv_heads = read_u32_vec(gguf, "gemma4.attention.head_count_kv")?;
        if kv_heads.len() != block_count as usize {
            return Err(Gemma4Error::WrongArrayLength {
                key: "gemma4.attention.head_count_kv",
                got: kv_heads.len(), expected: block_count as usize });
        }

        // Per-layer sliding-window pattern (true = sliding, false = full).
        let pattern = read_bool_vec(gguf, "gemma4.attention.sliding_window_pattern")?;
        if pattern.len() != block_count as usize {
            return Err(Gemma4Error::WrongArrayLength {
                key: "gemma4.attention.sliding_window_pattern",
                got: pattern.len(), expected: block_count as usize });
        }
        let attn_kinds: Vec<AttnKind> = pattern.iter()
            .map(|&b| if b { AttnKind::Sliding } else { AttnKind::Full })
            .collect();

        // Vocab from the token embedding tensor shape.
        let token_embd = gguf.tensor("token_embd.weight")
            .ok_or_else(|| Gemma4Error::MissingTensor("token_embd.weight".into()))?;
        let vocab_size = *token_embd.shape().get(1).ok_or_else(||
            Gemma4Error::MissingTensor("token_embd.weight (2D)".into()))? as u32;

        let tied_embeddings = gguf.tensor("output.weight").is_none();

        // MoE metadata — absent on the dense 31B.
        let expert_count      = optional_u32(gguf, "gemma4.expert_count").unwrap_or(0);
        let expert_used_count = optional_u32(gguf, "gemma4.expert_used_count").unwrap_or(0);
        let expert_ff_size    = optional_u32(gguf, "gemma4.expert_feed_forward_length").unwrap_or(0);

        Ok(Self {
            block_count, hidden_size, ffn_size, vocab_size, context_length,
            rms_norm_eps, eos_token_id,
            n_heads, head_dim_full, head_dim_swa, sliding_window,
            rope_freq_base, rope_freq_base_swa, rope_dim_full, rope_dim_swa,
            final_logit_softcapping, tied_embeddings,
            kv_heads, attn_kinds,
            expert_count, expert_used_count, expert_ff_size,
        })
    }

    /// True for MoE models (the 26B-A4B); false for the dense 31B.
    pub fn is_moe(&self) -> bool {
        self.expert_count > 0
    }

    /// Head dim for the given layer's attention kind.
    pub fn head_dim(&self, layer: usize) -> u32 {
        match self.attn_kinds[layer] {
            AttnKind::Sliding => self.head_dim_swa,
            AttnKind::Full    => self.head_dim_full,
        }
    }

    /// RoPE frequency base for the given layer.
    pub fn rope_base(&self, layer: usize) -> f32 {
        match self.attn_kinds[layer] {
            AttnKind::Sliding => self.rope_freq_base_swa,
            AttnKind::Full    => self.rope_freq_base,
        }
    }
}

/// Loaded Gemma 4 model: config + validated tensor presence.
#[derive(Debug, Clone)]
pub struct Gemma4Model {
    pub config: Gemma4Config,
}

impl Gemma4Model {
    pub fn load(gguf: &GgufFile) -> Result<Self> {
        let config = Gemma4Config::from_gguf(gguf)?;
        let model = Self { config };
        model.validate_tensor_presence(gguf)?;
        Ok(model)
    }

    fn validate_tensor_presence(&self, gguf: &GgufFile) -> Result<()> {
        require_tensor(gguf, "token_embd.weight")?;
        require_tensor(gguf, "output_norm.weight")?;
        if !self.config.tied_embeddings {
            require_tensor(gguf, "output.weight")?;
        }
        let moe = self.config.is_moe();
        for (layer, &kind) in self.config.attn_kinds.iter().enumerate() {
            for name in expected_tensors(layer as u32, kind, moe) {
                require_tensor(gguf, &name)?;
            }
        }
        Ok(())
    }
}

/// GGUF tensor names a Gemma 4 block requires.
///
/// Sandwich norms on both sub-layers. Note the asymmetry: sliding-window
/// layers carry a separate `attn_v` projection, but full-attention
/// layers do **not** — they ship only `attn_k`. (How the V stream is
/// reconstructed on the full layers is a forward-pass concern resolved
/// when the Gemma 4 oracle is built.)
pub fn expected_tensors(layer: u32, kind: AttnKind, moe: bool) -> Vec<String> {
    let mut names = vec![
        format!("blk.{layer}.attn_norm.weight"),
        format!("blk.{layer}.attn_q.weight"),
        format!("blk.{layer}.attn_k.weight"),
        format!("blk.{layer}.attn_q_norm.weight"),
        format!("blk.{layer}.attn_k_norm.weight"),
        format!("blk.{layer}.attn_output.weight"),
        format!("blk.{layer}.post_attention_norm.weight"),
        // Shared MLP — present on both dense and MoE layers.
        format!("blk.{layer}.ffn_norm.weight"),
        format!("blk.{layer}.ffn_gate.weight"),
        format!("blk.{layer}.ffn_up.weight"),
        format!("blk.{layer}.ffn_down.weight"),
        format!("blk.{layer}.post_ffw_norm.weight"),
        format!("blk.{layer}.layer_output_scale.weight"),
    ];
    if kind == AttnKind::Sliding {
        names.push(format!("blk.{layer}.attn_v.weight"));
    }
    if moe {
        // Dual-FFN: the shared MLP gets a post_ffw_norm_1; the routed
        // expert block adds its own pre/post norm, router, and experts.
        names.push(format!("blk.{layer}.post_ffw_norm_1.weight"));
        names.push(format!("blk.{layer}.pre_ffw_norm_2.weight"));
        names.push(format!("blk.{layer}.post_ffw_norm_2.weight"));
        names.push(format!("blk.{layer}.ffn_gate_inp.weight"));
        names.push(format!("blk.{layer}.ffn_gate_inp.scale"));
        names.push(format!("blk.{layer}.ffn_gate_up_exps.weight"));
        names.push(format!("blk.{layer}.ffn_down_exps.weight"));
        names.push(format!("blk.{layer}.ffn_down_exps.scale"));
    }
    names
}

// ---- helpers -----------------------------------------------------------------

fn require_metadata<'a>(gguf: &'a GgufFile, key: &'static str) -> Result<&'a MetaValue> {
    gguf.metadata_get(key).ok_or(Gemma4Error::MissingMetadata(key))
}

fn require_str<'a>(gguf: &'a GgufFile, key: &'static str) -> Result<&'a str> {
    require_metadata(gguf, key)?
        .as_str()
        .ok_or(Gemma4Error::WrongMetadataType { key, expected: "string" })
}

fn optional_u32(gguf: &GgufFile, key: &str) -> Option<u32> {
    gguf.metadata_get(key).and_then(|v| v.as_u32())
}

fn require_u32(gguf: &GgufFile, key: &'static str) -> Result<u32> {
    require_metadata(gguf, key)?
        .as_u32()
        .ok_or(Gemma4Error::WrongMetadataType { key, expected: "u32-compatible integer" })
}

fn require_f32(gguf: &GgufFile, key: &'static str) -> Result<f32> {
    require_metadata(gguf, key)?
        .as_f32()
        .ok_or(Gemma4Error::WrongMetadataType { key, expected: "f32-compatible float" })
}

fn read_u32_vec(gguf: &GgufFile, key: &'static str) -> Result<Vec<u32>> {
    let (_, arr) = require_metadata(gguf, key)?
        .as_array()
        .ok_or(Gemma4Error::WrongMetadataType { key, expected: "array" })?;
    arr.iter()
        .map(|e| e.as_u32().ok_or(Gemma4Error::WrongMetadataType {
            key, expected: "array of integers" }))
        .collect()
}

fn read_bool_vec(gguf: &GgufFile, key: &'static str) -> Result<Vec<bool>> {
    let (_, arr) = require_metadata(gguf, key)?
        .as_array()
        .ok_or(Gemma4Error::WrongMetadataType { key, expected: "array" })?;
    arr.iter()
        .map(|e| match e {
            MetaValue::Bool(b) => Ok(*b),
            _ => Err(Gemma4Error::WrongMetadataType { key, expected: "array of bools" }),
        })
        .collect()
}

fn require_tensor(gguf: &GgufFile, name: &str) -> Result<()> {
    if gguf.tensor(name).is_none() {
        return Err(Gemma4Error::MissingTensor(name.to_owned()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture_path() -> Option<PathBuf> {
        let home = std::env::var_os("HOME")?;
        let p = PathBuf::from(home).join("models/gemma4-31b/gemma-4-31B-it-UD-Q4_K_XL.gguf");
        p.exists().then_some(p)
    }

    #[test]
    fn loads_gemma4_31b_config_and_schedule() {
        let Some(p) = fixture_path() else { eprintln!("skip: no gemma4-31B fixture"); return };
        let gguf = GgufFile::open(&p).expect("open gguf");
        let model = Gemma4Model::load(&gguf).expect("load gemma4");
        let c = &model.config;
        eprintln!("gemma4: {} blocks, hidden {}, ffn {}, vocab {}",
                  c.block_count, c.hidden_size, c.ffn_size, c.vocab_size);
        assert_eq!(c.block_count, 60);
        assert_eq!(c.hidden_size, 5376);
        assert_eq!(c.n_heads, 32);
        assert_eq!(c.head_dim_full, 512);
        assert_eq!(c.head_dim_swa, 256);
        assert_eq!(c.kv_heads.len(), 60);
        assert_eq!(c.attn_kinds.len(), 60);
        // Reference pattern: 5 sliding, then 1 full, repeating.
        for layer in 0..60 {
            let expect = if (layer + 1) % 6 == 0 { AttnKind::Full } else { AttnKind::Sliding };
            assert_eq!(c.attn_kinds[layer], expect, "layer {layer} attn kind");
        }
        // Full layers have 4 KV heads, sliding have 16 (reference 31B).
        for layer in 0..60 {
            let want_kv = if c.attn_kinds[layer] == AttnKind::Full { 4 } else { 16 };
            assert_eq!(c.kv_heads[layer], want_kv, "layer {layer} kv heads");
        }
        assert_eq!(c.final_logit_softcapping, 30.0);
        assert!(c.tied_embeddings);
    }

    #[test]
    fn loads_gemma4_26b_moe_config() {
        let home = std::env::var_os("HOME").map(PathBuf::from);
        let Some(home) = home else { return };
        let p = home.join("models/gemma4-26B/gemma-4-26B-A4B-it-UD-Q6_K_XL.gguf");
        if !p.exists() { eprintln!("skip: no gemma4-26B fixture"); return; }
        let gguf = GgufFile::open(&p).expect("open gguf");
        let model = Gemma4Model::load(&gguf).expect("load gemma4 26B");
        let c = &model.config;
        assert!(c.is_moe());
        assert_eq!(c.block_count, 30);
        assert_eq!(c.hidden_size, 2816);
        assert_eq!(c.expert_count, 128);
        assert_eq!(c.expert_used_count, 8);
        assert_eq!(c.expert_ff_size, 704);
        assert_eq!(c.ffn_size, 2112);   // shared MLP intermediate
        eprintln!("gemma4-26B MoE: {} layers, {} experts, {} used, expert_ff {}",
                  c.block_count, c.expert_count, c.expert_used_count, c.expert_ff_size);
    }

    #[test]
    fn rejects_non_gemma4() {
        let home = std::env::var_os("HOME").map(PathBuf::from);
        let Some(home) = home else { return };
        let qwen = home.join("models/qwen-3.5-0.8B/Qwen3.5-0.8B-UD-Q4_K_XL.gguf");
        if !qwen.exists() { eprintln!("skip: no qwen fixture"); return; }
        let gguf = GgufFile::open(&qwen).unwrap();
        assert!(matches!(Gemma4Model::load(&gguf),
                         Err(Gemma4Error::WrongArchitecture { .. })));
    }
}
