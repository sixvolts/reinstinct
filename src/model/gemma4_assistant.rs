//! Gemma 4 Assistant: the MTP / speculative-decoding drafter that ships
//! alongside each `gemma-4-*-it` target.
//!
//! Arch (`general.architecture = "gemma4_assistant"`):
//! - 4 transformer blocks (always); SWA/full alternation per layer.
//! - **No K/V projections per block** — `attention.shared_kv_layers = 4`
//!   plus `attention.k_eq_v = true` means every drafter layer reads
//!   K (= V) directly from the target's KV cache.
//! - Two top-level MTP tensors:
//!     `mtp.pre_projection`   `[n_embd_backbone*2, hidden]`  — combines
//!         target-side signals (10752 → 1024 on the 31B drafter).
//!     `mtp.post_projection`  `[hidden, n_embd_backbone]`    — projects
//!         the drafter's hidden back to backbone dim for the next step.
//! - `requires_target_arch` names the matching target architecture.
//!
//! This module only parses + validates the GGUF; the forward path is
//! built on top.

use thiserror::Error;

use crate::gguf::{GgufFile, MetaValue};
use crate::model::gemma4::AttnKind;

const ARCH: &str = "gemma4_assistant";

#[derive(Debug, Error)]
pub enum Gemma4AssistantError {
    #[error("not a Gemma 4 assistant file: general.architecture = {got:?}, expected {expected:?}")]
    WrongArchitecture { got: String, expected: &'static str },

    #[error("missing required GGUF metadata key: {0}")]
    MissingMetadata(&'static str),

    #[error("metadata key {key} has wrong type (expected {expected})")]
    WrongMetadataType { key: &'static str, expected: &'static str },

    #[error("metadata array {key} has {got} entries, expected {expected}")]
    WrongArrayLength { key: &'static str, got: usize, expected: usize },

    #[error("missing required tensor: {0}")]
    MissingTensor(String),

    #[error("unexpected per-block tensor present: {0} \
             (drafter blocks should have Q-only attention)")]
    UnexpectedTensor(String),
}

type Result<T> = std::result::Result<T, Gemma4AssistantError>;

/// Gemma 4 Assistant hyperparameters from `gemma4_assistant.*` metadata.
#[derive(Debug, Clone)]
pub struct Gemma4AssistantConfig {
    pub block_count: u32,
    pub hidden_size: u32,
    pub ffn_size: u32,
    pub vocab_size: u32,
    pub context_length: u32,
    pub rms_norm_eps: f32,
    pub eos_token_id: u32,

    pub n_heads: u32,
    pub head_dim_full: u32,
    pub head_dim_swa: u32,
    pub sliding_window: u32,
    pub rope_freq_base: f32,
    pub rope_freq_base_swa: f32,
    pub rope_dim_full: u32,
    pub rope_dim_swa: u32,

    pub kv_heads: Vec<u32>,
    pub attn_kinds: Vec<AttnKind>,

    /// Target architecture this drafter binds to (e.g. "gemma4").
    pub requires_target_arch: String,
    /// Width of the target hidden state piped into pre_projection
    /// (= target's `embedding_length`).
    pub n_embd_backbone: u32,
    /// Clustered output-head parameters (set on all -assistant ckpts;
    /// only actually consumed by the E*B drafters in HF's reference
    /// implementation — flagged here for downstream dispatch).
    pub n_centroids: u32,
    pub centroid_top_k: u32,
    /// True ⇒ K and V are the same tensor (gemma4 convention).
    pub k_eq_v: bool,
    /// How many of the drafter's own layers share KV with the target;
    /// equals `block_count` on every released drafter (= all layers
    /// borrow KV from the target).
    pub shared_kv_layers: u32,
}

impl Gemma4AssistantConfig {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let arch = require_str(gguf, "general.architecture")?;
        if arch != ARCH {
            return Err(Gemma4AssistantError::WrongArchitecture {
                got: arch.to_owned(), expected: ARCH });
        }

        let p = "gemma4_assistant";
        let block_count    = require_u32(gguf, &format!("{p}.block_count"))?;
        let hidden_size    = require_u32(gguf, &format!("{p}.embedding_length"))?;
        let ffn_size       = require_u32(gguf, &format!("{p}.feed_forward_length"))?;
        let context_length = require_u32(gguf, &format!("{p}.context_length"))?;
        let rms_norm_eps   = require_f32(gguf, &format!("{p}.attention.layer_norm_rms_epsilon"))?;
        let n_heads        = require_u32(gguf, &format!("{p}.attention.head_count"))?;
        let head_dim_full  = require_u32(gguf, &format!("{p}.attention.key_length"))?;
        let head_dim_swa   = require_u32(gguf, &format!("{p}.attention.key_length_swa"))?;
        let sliding_window = require_u32(gguf, &format!("{p}.attention.sliding_window"))?;
        let rope_freq_base     = require_f32(gguf, &format!("{p}.rope.freq_base"))?;
        let rope_freq_base_swa = require_f32(gguf, &format!("{p}.rope.freq_base_swa"))?;
        let rope_dim_full  = require_u32(gguf, &format!("{p}.rope.dimension_count"))?;
        let rope_dim_swa   = require_u32(gguf, &format!("{p}.rope.dimension_count_swa"))?;
        let eos_token_id   = require_u32(gguf, "tokenizer.ggml.eos_token_id")?;

        let kv_heads = read_u32_vec_or_broadcast(
            gguf, &format!("{p}.attention.head_count_kv"), block_count as usize)?;
        if kv_heads.len() != block_count as usize {
            return Err(Gemma4AssistantError::WrongArrayLength {
                key: "gemma4_assistant.attention.head_count_kv",
                got: kv_heads.len(), expected: block_count as usize });
        }
        let pattern = read_bool_vec(gguf, &format!("{p}.attention.sliding_window_pattern"))?;
        if pattern.len() != block_count as usize {
            return Err(Gemma4AssistantError::WrongArrayLength {
                key: "gemma4_assistant.attention.sliding_window_pattern",
                got: pattern.len(), expected: block_count as usize });
        }
        let attn_kinds: Vec<AttnKind> = pattern.iter()
            .map(|&b| if b { AttnKind::Sliding } else { AttnKind::Full })
            .collect();

        let token_embd = gguf.tensor("token_embd.weight")
            .ok_or_else(|| Gemma4AssistantError::MissingTensor("token_embd.weight".into()))?;
        let vocab_size = *token_embd.shape().get(1).ok_or_else(||
            Gemma4AssistantError::MissingTensor("token_embd.weight (2D)".into()))? as u32;

        let n_embd_backbone = require_u32(gguf, &format!("{p}.n_embd_backbone"))?;
        let n_centroids    = optional_u32(gguf, &format!("{p}.n_centroids")).unwrap_or(0);
        let centroid_top_k = optional_u32(gguf, &format!("{p}.centroid_top_k")).unwrap_or(0);
        let k_eq_v = match gguf.metadata_get(&format!("{p}.attention.k_eq_v")) {
            Some(MetaValue::Bool(b)) => *b,
            None => false,
            _ => return Err(Gemma4AssistantError::WrongMetadataType {
                key: "gemma4_assistant.attention.k_eq_v", expected: "bool" }),
        };
        let shared_kv_layers = optional_u32(
            gguf, &format!("{p}.attention.shared_kv_layers")).unwrap_or(0);

        let requires_target_arch = match gguf.metadata_get(
            &format!("{p}.requires_target_arch"))
        {
            Some(MetaValue::String(s)) => s.clone(),
            _ => String::new(),
        };

        Ok(Self {
            block_count, hidden_size, ffn_size, vocab_size, context_length,
            rms_norm_eps, eos_token_id,
            n_heads, head_dim_full, head_dim_swa, sliding_window,
            rope_freq_base, rope_freq_base_swa, rope_dim_full, rope_dim_swa,
            kv_heads, attn_kinds,
            requires_target_arch, n_embd_backbone, n_centroids, centroid_top_k,
            k_eq_v, shared_kv_layers,
        })
    }

    pub fn head_dim(&self, layer: usize) -> u32 {
        match self.attn_kinds[layer] {
            AttnKind::Sliding => self.head_dim_swa,
            AttnKind::Full    => self.head_dim_full,
        }
    }

    pub fn rope_base(&self, layer: usize) -> f32 {
        match self.attn_kinds[layer] {
            AttnKind::Sliding => self.rope_freq_base_swa,
            AttnKind::Full    => self.rope_freq_base,
        }
    }
}

/// Loaded Gemma 4 Assistant model — config + validated tensor presence.
#[derive(Debug, Clone)]
pub struct Gemma4AssistantModel {
    pub config: Gemma4AssistantConfig,
}

impl Gemma4AssistantModel {
    pub fn load(gguf: &GgufFile) -> Result<Self> {
        let config = Gemma4AssistantConfig::from_gguf(gguf)?;
        let model = Self { config };
        model.validate_tensor_presence(gguf)?;
        Ok(model)
    }

    fn validate_tensor_presence(&self, gguf: &GgufFile) -> Result<()> {
        // Top-level
        require_tensor(gguf, "token_embd.weight")?;
        require_tensor(gguf, "output_norm.weight")?;
        require_tensor(gguf, "mtp.pre_projection.weight")?;
        require_tensor(gguf, "mtp.post_projection.weight")?;

        // Per-block: Q-only attention (no K/V projections) + sandwich
        // norms + FFN + layer scale.
        for layer in 0..self.config.block_count {
            for stem in &[
                "attn_norm.weight",
                "attn_q.weight",
                "attn_q_norm.weight",
                "attn_output.weight",
                "post_attention_norm.weight",
                "ffn_norm.weight",
                "ffn_gate.weight",
                "ffn_up.weight",
                "ffn_down.weight",
                "post_ffw_norm.weight",
                "layer_output_scale.weight",
            ] {
                require_tensor(gguf, &format!("blk.{layer}.{stem}"))?;
            }
            // A K/V projection would mean the drafter has its own KV
            // cache, which contradicts shared_kv_layers == block_count.
            for forbidden in &["attn_k.weight", "attn_v.weight", "attn_k_norm.weight"] {
                let name = format!("blk.{layer}.{forbidden}");
                if gguf.tensor(&name).is_some() {
                    return Err(Gemma4AssistantError::UnexpectedTensor(name));
                }
            }
        }

        // Cross-check shape: pre_projection is [n_embd_backbone*2, hidden],
        // post_projection is [hidden, n_embd_backbone].
        let pre = gguf.tensor("mtp.pre_projection.weight").unwrap();
        let expect_pre_in = (self.config.n_embd_backbone * 2) as u64;
        if pre.shape() != [expect_pre_in, self.config.hidden_size as u64] {
            return Err(Gemma4AssistantError::WrongArrayLength {
                key: "mtp.pre_projection.weight.shape",
                got: pre.shape().len(),
                expected: 2,
            });
        }
        let post = gguf.tensor("mtp.post_projection.weight").unwrap();
        if post.shape() != [self.config.hidden_size as u64,
                            self.config.n_embd_backbone as u64] {
            return Err(Gemma4AssistantError::WrongArrayLength {
                key: "mtp.post_projection.weight.shape",
                got: post.shape().len(),
                expected: 2,
            });
        }
        Ok(())
    }
}

// --- metadata helpers (mirror src/model/gemma4.rs) ---

fn require_str<'a>(gguf: &'a GgufFile, key: &'static str) -> Result<&'a str> {
    match gguf.metadata_get(key) {
        Some(MetaValue::String(s)) => Ok(s),
        Some(_) => Err(Gemma4AssistantError::WrongMetadataType { key, expected: "string" }),
        None => Err(Gemma4AssistantError::MissingMetadata(key)),
    }
}

fn require_u32(gguf: &GgufFile, key: &str) -> Result<u32> {
    let static_key: &'static str = Box::leak(key.to_string().into_boxed_str());
    match gguf.metadata_get(key) {
        Some(v) => v.as_u32()
            .ok_or(Gemma4AssistantError::WrongMetadataType {
                key: static_key, expected: "u32" }),
        None => Err(Gemma4AssistantError::MissingMetadata(static_key)),
    }
}

fn require_f32(gguf: &GgufFile, key: &str) -> Result<f32> {
    let static_key: &'static str = Box::leak(key.to_string().into_boxed_str());
    match gguf.metadata_get(key) {
        Some(MetaValue::F32(v)) => Ok(*v),
        Some(_) => Err(Gemma4AssistantError::WrongMetadataType {
            key: static_key, expected: "f32" }),
        None => Err(Gemma4AssistantError::MissingMetadata(static_key)),
    }
}

fn optional_u32(gguf: &GgufFile, key: &str) -> Option<u32> {
    gguf.metadata_get(key).and_then(|v| v.as_u32())
}

fn read_u32_vec_or_broadcast(gguf: &GgufFile, key: &str, n: usize) -> Result<Vec<u32>> {
    let static_key: &'static str = Box::leak(key.to_string().into_boxed_str());
    match gguf.metadata_get(key) {
        Some(MetaValue::Array { values, .. }) => {
            let mut out = Vec::with_capacity(values.len());
            for v in values {
                out.push(v.as_u32().ok_or(Gemma4AssistantError::WrongMetadataType {
                    key: static_key, expected: "u32 array" })?);
            }
            Ok(out)
        }
        Some(v) => match v.as_u32() {
            Some(x) => Ok(vec![x; n]),
            None => Err(Gemma4AssistantError::WrongMetadataType {
                key: static_key, expected: "u32 or u32 array" }),
        }
        None => Err(Gemma4AssistantError::MissingMetadata(static_key)),
    }
}

fn read_bool_vec(gguf: &GgufFile, key: &str) -> Result<Vec<bool>> {
    let static_key: &'static str = Box::leak(key.to_string().into_boxed_str());
    match gguf.metadata_get(key) {
        Some(MetaValue::Array { values, .. }) => {
            let mut out = Vec::with_capacity(values.len());
            for v in values {
                match v {
                    MetaValue::Bool(b) => out.push(*b),
                    _ => return Err(Gemma4AssistantError::WrongMetadataType {
                        key: static_key, expected: "bool array" }),
                }
            }
            Ok(out)
        }
        Some(_) => Err(Gemma4AssistantError::WrongMetadataType {
            key: static_key, expected: "bool array" }),
        None => Err(Gemma4AssistantError::MissingMetadata(static_key)),
    }
}

fn require_tensor(gguf: &GgufFile, name: &str) -> Result<()> {
    if gguf.tensor(name).is_none() {
        return Err(Gemma4AssistantError::MissingTensor(name.into()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_gemma4_31b_assistant_config() {
        let path = std::path::PathBuf::from(
            "/home/sixvolts/models/gemma4-mtp/gemma-4-31B-it-assistant.Q4_K_M.gguf");
        if !path.exists() {
            eprintln!("skipping: {} not present", path.display());
            return;
        }
        let g = GgufFile::open(&path).expect("open");
        let m = Gemma4AssistantModel::load(&g).expect("load");
        let c = &m.config;
        assert_eq!(c.block_count, 4);
        assert_eq!(c.hidden_size, 1024);
        assert_eq!(c.ffn_size, 8192);
        assert_eq!(c.n_heads, 32);
        assert_eq!(c.n_embd_backbone, 5376);
        assert!(c.k_eq_v);
        assert_eq!(c.shared_kv_layers, 4);
        assert_eq!(c.requires_target_arch, "gemma4");
        assert_eq!(c.vocab_size, 262144);
        assert_eq!(c.kv_heads.len(), 4);
        assert_eq!(c.attn_kinds.len(), 4);
    }
}
