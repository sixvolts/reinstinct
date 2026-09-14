//! DFlash block-diffusion drafter (`general.architecture = "dflash"`).
//!
//! Unlike the `gemma4-assistant` drafter, which proposes tokens one at a
//! time, DFlash denoises a whole `block_size` block in a single forward
//! pass: position 0 holds an anchor token we already know, the rest are
//! `MASK`, and all of them are predicted at once. See
//! `docs/DFLASH_PORT.md` for the full spec and its sources.
//!
//! Two things about this file are worth knowing up front.
//!
//! **The draft body is Qwen3-shaped, not Gemma-shaped**, even when the
//! target is Gemma 4 — the upstream config says `model_type: qwen3` with
//! `hidden_act: silu`. So the blocks are pre-norm (`attn_norm`,
//! `ffn_norm`) with a SwiGLU MLP, and there are no Gemma sandwich norms.
//! The 11-tensor block layout is validated below precisely so a
//! Gemma-shaped checkpoint cannot be loaded by mistake.
//!
//! **The drafter owns no embedding and no output head.** It borrows the
//! target's, so `token_embd.weight` and `output.weight` are expected to
//! be *absent*; a file carrying them is not this architecture.

use thiserror::Error;

use crate::gguf::{GgufFile, MetaValue};

const ARCH: &str = "dflash";

#[derive(Debug, Error)]
pub enum DFlashError {
    #[error("not a DFlash file: general.architecture = {got:?}, expected {expected:?}")]
    WrongArchitecture { got: String, expected: &'static str },

    #[error("missing required GGUF metadata key: {0}")]
    MissingMetadata(String),

    #[error("metadata key {key} has wrong type (expected {expected})")]
    WrongMetadataType { key: String, expected: &'static str },

    #[error("metadata array {key} has {got} entries, expected {expected}")]
    WrongArrayLength { key: String, got: usize, expected: usize },

    #[error("missing required tensor: {0}")]
    MissingTensor(String),

    #[error("unexpected tensor present: {0} \
             (a DFlash drafter shares the target's embedding and LM head, \
              so it should carry neither)")]
    UnexpectedTensor(String),

    #[error("tensor {name} has shape {got:?}, expected {expected:?}")]
    WrongTensorShape { name: String, got: Vec<u64>, expected: Vec<u64> },

    #[error("target_layers refers to block {got}, but the target has only \
             {target_blocks} blocks")]
    TargetLayerOutOfRange { got: u32, target_blocks: u32 },
}

type Result<T> = std::result::Result<T, DFlashError>;

/// Per-block attention flavour. Mirrors the upstream `layer_types`:
/// `sliding_attention` layers are causal with a window; `full_attention`
/// layers carry **no mask at all** and see the whole `[ctx | block]` span
/// bidirectionally. That asymmetry is load-bearing — see
/// `Qwen3DFlashAttention` in the reference, where `is_causal` is derived
/// from the layer type rather than set independently.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DFlashAttn {
    /// Causal, restricted to `sliding_window` keys.
    SlidingCausal,
    /// Unmasked in both directions across context and block.
    FullBidirectional,
}

/// DFlash hyperparameters from `dflash.*` metadata.
#[derive(Debug, Clone)]
pub struct DFlashConfig {
    pub block_count: u32,
    /// Draft hidden width. Equals the target's `embedding_length`, since
    /// the context features are projected into the draft's own width.
    pub hidden_size: u32,
    pub ffn_size: u32,
    pub context_length: u32,
    pub rms_norm_eps: f32,

    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
    pub sliding_window: u32,
    pub rope_freq_base: f32,

    pub attn_kinds: Vec<DFlashAttn>,

    /// Tokens denoised per draft pass (16 upstream).
    pub block_size: u32,
    /// Target blocks whose output feeds `fc`, as **block indices**.
    ///
    /// The GGUF stores HF `hidden_states` indices, which are one higher
    /// because `hidden_states[0]` is the embedding output. That subtraction
    /// happens here, once, so every consumer downstream gets block indices
    /// and nobody has to remember. Getting this wrong does not crash — it
    /// silently conditions the drafter on the wrong layers and costs
    /// acceptance rate.
    pub target_layers: Vec<u32>,
}

impl DFlashConfig {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let arch = require_str(gguf, "general.architecture")?;
        if arch != ARCH {
            return Err(DFlashError::WrongArchitecture {
                got: arch.to_owned(), expected: ARCH });
        }
        let p = ARCH;

        let block_count    = require_u32(gguf, &format!("{p}.block_count"))?;
        let hidden_size    = require_u32(gguf, &format!("{p}.embedding_length"))?;
        let ffn_size       = require_u32(gguf, &format!("{p}.feed_forward_length"))?;
        let context_length = require_u32(gguf, &format!("{p}.context_length"))?;
        let rms_norm_eps   = require_f32(gguf, &format!("{p}.attention.layer_norm_rms_epsilon"))?;
        let n_heads        = require_u32(gguf, &format!("{p}.attention.head_count"))?;
        let n_kv_heads     = require_u32(gguf, &format!("{p}.attention.head_count_kv"))?;
        let head_dim       = require_u32(gguf, &format!("{p}.attention.key_length"))?;
        let sliding_window = require_u32(gguf, &format!("{p}.attention.sliding_window"))?;
        let rope_freq_base = require_f32(gguf, &format!("{p}.rope.freq_base"))?;
        let block_size     = require_u32(gguf, &format!("{p}.block_size"))?;

        // Value length is carried separately but DFlash never uses a
        // different one; a mismatch means this is not the layout we think.
        let v_len = require_u32(gguf, &format!("{p}.attention.value_length"))?;
        if v_len != head_dim {
            return Err(DFlashError::WrongArrayLength {
                key: "dflash.attention.value_length".into(),
                got: v_len as usize, expected: head_dim as usize });
        }

        let pattern = read_bool_vec(gguf, &format!("{p}.attention.sliding_window_pattern"))?;
        if pattern.len() != block_count as usize {
            return Err(DFlashError::WrongArrayLength {
                key: "dflash.attention.sliding_window_pattern".into(),
                got: pattern.len(), expected: block_count as usize });
        }
        let attn_kinds = pattern.iter()
            .map(|&b| if b { DFlashAttn::SlidingCausal } else { DFlashAttn::FullBidirectional })
            .collect();

        // HF hidden-state index → block index. See the field doc.
        let raw = read_u32_vec(gguf, &format!("{p}.target_layers"))?;
        let mut target_layers = Vec::with_capacity(raw.len());
        for v in raw {
            if v == 0 {
                // hidden_states[0] is the embedding output, not a block.
                return Err(DFlashError::WrongMetadataType {
                    key: "dflash.target_layers".into(),
                    expected: "hidden-state indices >= 1" });
            }
            target_layers.push(v - 1);
        }

        Ok(Self {
            block_count, hidden_size, ffn_size, context_length, rms_norm_eps,
            n_heads, n_kv_heads, head_dim, sliding_window, rope_freq_base,
            attn_kinds, block_size, target_layers,
        })
    }

    /// Feature width `fc` consumes: `target_layers.len() * hidden_size`.
    pub fn fused_ctx_width(&self) -> u64 {
        self.target_layers.len() as u64 * self.hidden_size as u64
    }

    /// Check the tapped layers exist in a target with `target_blocks`
    /// blocks. Call once the target is known — the drafter GGUF alone
    /// cannot tell whether it was paired correctly.
    pub fn validate_against_target(&self, target_blocks: u32) -> Result<()> {
        for &l in &self.target_layers {
            if l >= target_blocks {
                return Err(DFlashError::TargetLayerOutOfRange {
                    got: l, target_blocks });
            }
        }
        Ok(())
    }
}

/// A parsed DFlash drafter: config plus a validated tensor inventory.
#[derive(Debug, Clone)]
pub struct DFlashModel {
    pub config: DFlashConfig,
}

impl DFlashModel {
    pub fn load(gguf: &GgufFile) -> Result<Self> {
        let config = DFlashConfig::from_gguf(gguf)?;
        let model = Self { config };
        model.validate_tensors(gguf)?;
        Ok(model)
    }

    fn validate_tensors(&self, gguf: &GgufFile) -> Result<()> {
        let c = &self.config;
        let h = c.hidden_size as u64;
        let q_dim = (c.n_heads * c.head_dim) as u64;
        let kv_dim = (c.n_kv_heads * c.head_dim) as u64;
        let ff = c.ffn_size as u64;

        // Top level: the cross-layer fusion and the two norms.
        shape(gguf, "fc.weight", &[c.fused_ctx_width(), h])?;
        shape(gguf, "enc.output_norm.weight", &[h])?;
        shape(gguf, "output_norm.weight", &[h])?;

        // Borrowed from the target — their presence means this is a
        // standalone LM, not a DFlash drafter.
        for name in ["token_embd.weight", "output.weight"] {
            if gguf.tensor(name).is_some() {
                return Err(DFlashError::UnexpectedTensor(name.into()));
            }
        }

        for l in 0..c.block_count {
            shape(gguf, &format!("blk.{l}.attn_norm.weight"), &[h])?;
            shape(gguf, &format!("blk.{l}.attn_q.weight"), &[h, q_dim])?;
            shape(gguf, &format!("blk.{l}.attn_k.weight"), &[h, kv_dim])?;
            shape(gguf, &format!("blk.{l}.attn_v.weight"), &[h, kv_dim])?;
            shape(gguf, &format!("blk.{l}.attn_output.weight"), &[q_dim, h])?;
            // Per-head-dim norms, Qwen3 style.
            shape(gguf, &format!("blk.{l}.attn_q_norm.weight"), &[c.head_dim as u64])?;
            shape(gguf, &format!("blk.{l}.attn_k_norm.weight"), &[c.head_dim as u64])?;
            shape(gguf, &format!("blk.{l}.ffn_norm.weight"), &[h])?;
            shape(gguf, &format!("blk.{l}.ffn_gate.weight"), &[h, ff])?;
            shape(gguf, &format!("blk.{l}.ffn_up.weight"), &[h, ff])?;
            shape(gguf, &format!("blk.{l}.ffn_down.weight"), &[ff, h])?;

            // Gemma sandwich norms would mean a Gemma-shaped block, which
            // this is not; DFlash2's convs and codebooks would mean the
            // richer architecture, which the runtime does not implement.
            for extra in ["post_attention_norm.weight", "post_ffw_norm.weight",
                          "attention_conv.weight", "mlp_conv.weight"] {
                let name = format!("blk.{l}.{extra}");
                if gguf.tensor(&name).is_some() {
                    return Err(DFlashError::UnexpectedTensor(name));
                }
            }
        }
        Ok(())
    }
}

fn shape(gguf: &GgufFile, name: &str, expected: &[u64]) -> Result<()> {
    let info = gguf.tensor(name)
        .ok_or_else(|| DFlashError::MissingTensor(name.to_string()))?;
    if info.shape() != expected {
        return Err(DFlashError::WrongTensorShape {
            name: name.to_string(),
            got: info.shape().to_vec(),
            expected: expected.to_vec(),
        });
    }
    Ok(())
}

fn require_str<'a>(gguf: &'a GgufFile, key: &str) -> Result<&'a str> {
    match gguf.metadata_get(key) {
        Some(MetaValue::String(s)) => Ok(s),
        Some(_) => Err(DFlashError::WrongMetadataType { key: key.into(), expected: "string" }),
        None => Err(DFlashError::MissingMetadata(key.into())),
    }
}

fn require_u32(gguf: &GgufFile, key: &str) -> Result<u32> {
    match gguf.metadata_get(key) {
        Some(v) => v.as_u32().ok_or(DFlashError::WrongMetadataType {
            key: key.into(), expected: "u32" }),
        None => Err(DFlashError::MissingMetadata(key.into())),
    }
}

fn require_f32(gguf: &GgufFile, key: &str) -> Result<f32> {
    match gguf.metadata_get(key) {
        Some(MetaValue::F32(v)) => Ok(*v),
        Some(_) => Err(DFlashError::WrongMetadataType {
            key: key.into(), expected: "f32" }),
        None => Err(DFlashError::MissingMetadata(key.into())),
    }
}

fn read_u32_vec(gguf: &GgufFile, key: &str) -> Result<Vec<u32>> {
    match gguf.metadata_get(key) {
        Some(MetaValue::Array { values, .. }) => {
            let mut out = Vec::with_capacity(values.len());
            for v in values {
                out.push(v.as_u32().ok_or(DFlashError::WrongMetadataType {
                    key: key.into(), expected: "u32 array" })?);
            }
            Ok(out)
        }
        Some(_) => Err(DFlashError::WrongMetadataType {
            key: key.into(), expected: "u32 array" }),
        None => Err(DFlashError::MissingMetadata(key.into())),
    }
}

fn read_bool_vec(gguf: &GgufFile, key: &str) -> Result<Vec<bool>> {
    match gguf.metadata_get(key) {
        Some(MetaValue::Array { values, .. }) => {
            let mut out = Vec::with_capacity(values.len());
            for v in values {
                match v {
                    MetaValue::Bool(b) => out.push(*b),
                    _ => return Err(DFlashError::WrongMetadataType {
                        key: key.into(), expected: "bool array" }),
                }
            }
            Ok(out)
        }
        Some(_) => Err(DFlashError::WrongMetadataType {
            key: key.into(), expected: "bool array" }),
        None => Err(DFlashError::MissingMetadata(key.into())),
    }
}
