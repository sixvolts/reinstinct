//! Qwen 3.5 / 3.6: typed config + tensor binding for the hybrid
//! Gated-DeltaNet / GQA-attention architecture.
//!
//! Handles two GGUF architectures that share the GDN + full-attention
//! layer stack and differ only in the FFN:
//!   * `qwen35`    — dense SwiGLU FFN (`ffn_gate/up/down`).
//!   * `qwen35moe` — MoE FFN: 256 routed experts + a gated shared expert.
//!
//! Layer schedule alternates `linear_attention` and `full_attention` per
//! `<arch>.full_attention_interval` (pattern `[L,L,L,F]…`).
//!
//! See `project_qwen35_tensor_map.md` for the GGUF→PyTorch mapping.

use thiserror::Error;

use crate::gguf::{GgmlType, GgufFile, MetaValue};

#[derive(Debug, Error)]
pub enum Qwen35Error {
    #[error("unsupported architecture {got:?} (expected qwen35 or qwen35moe)")]
    WrongArchitecture { got: String },

    #[error("missing required GGUF metadata key: {0}")]
    MissingMetadata(String),

    #[error("metadata key {key} has wrong type (expected {expected})")]
    WrongMetadataType { key: String, expected: &'static str },

    #[error("missing required tensor: {0}")]
    MissingTensor(String),

    #[error("tensor {name} has unexpected type {got:?} (expected one of {expected:?})")]
    WrongTensorType { name: String, got: GgmlType, expected: Vec<GgmlType> },

    #[error("tensor {name} has unexpected shape {got:?} (expected {expected:?})")]
    WrongTensorShape { name: String, got: Vec<u64>, expected: Vec<u64> },
}

type Result<T> = std::result::Result<T, Qwen35Error>;

/// MoE FFN hyperparameters — present only for the `qwen35moe` arch.
#[derive(Debug, Clone)]
pub struct MoeConfig {
    /// Total routed experts (`expert_count`).
    pub n_expert: u32,
    /// Experts selected per token (`expert_used_count`).
    pub n_expert_used: u32,
    /// Routed-expert intermediate width (`expert_feed_forward_length`).
    pub expert_ff: u32,
    /// Shared-expert intermediate width (`expert_shared_feed_forward_length`).
    pub shared_expert_ff: u32,
}

/// Hyperparameters loaded from `<arch>.*` GGUF metadata plus a few
/// inferred values (vocab from `token_embd`, tied embeddings from the
/// absence of `output.weight`).
#[derive(Debug, Clone)]
pub struct Qwen35Config {
    /// `"qwen35"` or `"qwen35moe"`.
    pub arch: String,

    // Topology
    pub block_count: u32,
    pub hidden_size: u32,
    /// Dense FFN intermediate width — 0 on the MoE arch (no dense FFN).
    pub ffn_size: u32,
    pub vocab_size: u32,
    pub context_length: u32,
    pub rms_norm_eps: f32,
    pub eos_token_id: u32,

    // Full-attention block
    pub attn_n_heads: u32,
    pub attn_n_kv_heads: u32,
    pub attn_head_dim: u32,

    // Linear-attention (Gated DeltaNet), GQA.
    pub gdn_value_dim: u32,
    pub gdn_n_heads: u32,
    pub gdn_n_k_heads: u32,
    pub gdn_head_dim: u32,
    pub gdn_conv_kernel: u32,

    // RoPE
    pub rope_freq_base: f32,
    pub rope_dim_count: u32,
    pub rope_dim_sections: [u32; 4],

    // Layer schedule
    pub full_attention_interval: u32,

    // Embedding
    pub tied_embeddings: bool,

    /// `Some` on `qwen35moe`.
    pub moe: Option<MoeConfig>,

    /// Number of trailing blocks that are MTP "next-N" predictor heads.
    /// 0 for the base model; 1 for Unsloth's Qwen 3.6 MTP release
    /// (block_count = 65 with the last block being the MTP layer:
    /// full-attention transformer block + `nextn.{eh_proj, enorm,
    /// hnorm, shared_head_norm}` weights, run separately from the
    /// main forward as a 1-token drafter).
    pub nextn_predict_layers: u32,
}

impl Qwen35Config {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let arch = require_str(gguf, "general.architecture")?.to_owned();
        let is_moe = match arch.as_str() {
            "qwen35"    => false,
            "qwen35moe" => true,
            _ => return Err(Qwen35Error::WrongArchitecture { got: arch }),
        };
        let p = arch.as_str();   // metadata key prefix

        let block_count             = require_u32(gguf, &k(p, "block_count"))?;
        let hidden_size             = require_u32(gguf, &k(p, "embedding_length"))?;
        let context_length          = require_u32(gguf, &k(p, "context_length"))?;
        let rms_norm_eps            = require_f32(gguf, &k(p, "attention.layer_norm_rms_epsilon"))?;
        let attn_n_heads            = require_u32(gguf, &k(p, "attention.head_count"))?;
        let attn_n_kv_heads         = require_u32(gguf, &k(p, "attention.head_count_kv"))?;
        let attn_head_dim           = require_u32(gguf, &k(p, "attention.key_length"))?;
        let gdn_value_dim           = require_u32(gguf, &k(p, "ssm.inner_size"))?;
        let gdn_n_heads             = require_u32(gguf, &k(p, "ssm.time_step_rank"))?;
        let gdn_n_k_heads           = require_u32(gguf, &k(p, "ssm.group_count"))?;
        let gdn_head_dim            = require_u32(gguf, &k(p, "ssm.state_size"))?;
        let gdn_conv_kernel         = require_u32(gguf, &k(p, "ssm.conv_kernel"))?;
        let rope_freq_base          = require_f32(gguf, &k(p, "rope.freq_base"))?;
        let rope_dim_count          = require_u32(gguf, &k(p, "rope.dimension_count"))?;
        let full_attention_interval = require_u32(gguf, &k(p, "full_attention_interval"))?;
        let eos_token_id            = require_u32(gguf, "tokenizer.ggml.eos_token_id")?;
        let rope_dim_sections       = read_u32_array(gguf, &k(p, "rope.dimension_sections"))?;
        let nextn_predict_layers    = gguf.metadata_get(&k(p, "nextn_predict_layers"))
            .and_then(|v| v.as_u32()).unwrap_or(0);

        // Dense FFN width — absent on the MoE arch.
        let ffn_size = gguf.metadata_get(&k(p, "feed_forward_length"))
            .and_then(|v| v.as_u32()).unwrap_or(0);

        let moe = if is_moe {
            Some(MoeConfig {
                n_expert:         require_u32(gguf, &k(p, "expert_count"))?,
                n_expert_used:    require_u32(gguf, &k(p, "expert_used_count"))?,
                expert_ff:        require_u32(gguf, &k(p, "expert_feed_forward_length"))?,
                shared_expert_ff: require_u32(gguf, &k(p, "expert_shared_feed_forward_length"))?,
            })
        } else {
            None
        };

        // Vocab size from the token embedding tensor (authoritative).
        let token_embd = gguf.tensor("token_embd.weight")
            .ok_or_else(|| Qwen35Error::MissingTensor("token_embd.weight".into()))?;
        if token_embd.shape().len() != 2 || token_embd.shape()[0] != hidden_size as u64 {
            return Err(Qwen35Error::WrongTensorShape {
                name: "token_embd.weight".into(),
                got: token_embd.shape().to_vec(),
                expected: vec![hidden_size as u64, 0],
            });
        }
        let vocab_size = token_embd.shape()[1] as u32;
        let tied_embeddings = gguf.tensor("output.weight").is_none();

        Ok(Self {
            arch,
            block_count, hidden_size, ffn_size, vocab_size, context_length,
            rms_norm_eps, eos_token_id,
            attn_n_heads, attn_n_kv_heads, attn_head_dim,
            gdn_value_dim, gdn_n_heads, gdn_n_k_heads, gdn_head_dim, gdn_conv_kernel,
            rope_freq_base, rope_dim_count, rope_dim_sections,
            nextn_predict_layers,
            full_attention_interval, tied_embeddings, moe,
        })
    }

    /// Per-layer block type. The trailing `nextn_predict_layers` blocks
    /// are MTP next-N heads (NextN); blocks before that follow the
    /// `(idx+1) % interval == 0 ⇒ FullAttention` schedule.
    pub fn block_kind(&self, layer_idx: u32) -> BlockKind {
        if layer_idx >= self.block_count - self.nextn_predict_layers {
            return BlockKind::NextN;
        }
        if (layer_idx + 1) % self.full_attention_interval == 0 {
            BlockKind::FullAttention
        } else {
            BlockKind::LinearAttention
        }
    }

    pub fn is_moe(&self) -> bool { self.moe.is_some() }

    /// True for models with one or more trailing MTP next-N predictor
    /// blocks (Unsloth's Qwen 3.6 MTP release).
    pub fn has_nextn(&self) -> bool { self.nextn_predict_layers > 0 }

    /// Number of "main forward" blocks — total minus any trailing NextN
    /// MTP heads. The drafter loop runs blocks 0..n_main_blocks for the
    /// main step and the trailing block(s) only on demand.
    pub fn n_main_blocks(&self) -> u32 { self.block_count - self.nextn_predict_layers }

    /// Linear-attention key/value head dim (K/Q and V share it).
    pub fn gdn_value_head_dim(&self) -> u32 { self.gdn_head_dim }

    /// Linear-attention key/query projection width = n_k_heads × head_dim.
    pub fn gdn_key_dim(&self) -> u32 { self.gdn_n_k_heads * self.gdn_head_dim }

    /// Linear-attention input-projection output dim = q ‖ k ‖ v widths.
    pub fn gdn_qkv_concat_dim(&self) -> u32 {
        2 * self.gdn_key_dim() + self.gdn_value_dim
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlockKind {
    LinearAttention,
    FullAttention,
    /// MTP next-N predictor head. Same tensor layout as `FullAttention`
    /// plus the `nextn.{eh_proj, enorm, hnorm, shared_head_norm}`
    /// weights. Not run during the main forward — invoked by the
    /// spec-decode drafter to produce a +2 token candidate.
    NextN,
}

/// Loaded model: config + per-layer block schedule. Tensor data stays in
/// the underlying `GgufFile` mmap.
#[derive(Debug, Clone)]
pub struct Qwen35Model {
    pub config: Qwen35Config,
    /// Block kinds for the MAIN FORWARD only — size `n_main_blocks()`.
    /// On MTP-enabled GGUFs (`nextn_predict_layers > 0`) the trailing
    /// NextN blocks are NOT in this vec; load them via
    /// `mtp_block_kinds()` / `expected_tensors(idx, NextN, ...)`.
    pub block_kinds: Vec<BlockKind>,
}

impl Qwen35Model {
    pub fn load(gguf: &GgufFile) -> Result<Self> {
        let config = Qwen35Config::from_gguf(gguf)?;
        let block_kinds: Vec<BlockKind> = (0..config.n_main_blocks())
            .map(|i| config.block_kind(i))
            .collect();
        let model = Self { config, block_kinds };
        model.validate_tensor_presence(gguf)?;
        Ok(model)
    }

    /// Layer indices and kinds for the trailing MTP NextN blocks.
    /// Empty on base models.
    pub fn mtp_block_kinds(&self) -> Vec<(u32, BlockKind)> {
        (self.config.n_main_blocks()..self.config.block_count)
            .map(|i| (i, self.config.block_kind(i)))
            .collect()
    }

    fn validate_tensor_presence(&self, gguf: &GgufFile) -> Result<()> {
        require_tensor(gguf, "token_embd.weight")?;
        require_tensor(gguf, "output_norm.weight")?;
        if !self.config.tied_embeddings {
            require_tensor(gguf, "output.weight")?;
        }
        // Main-forward blocks.
        for (i, &kind) in self.block_kinds.iter().enumerate() {
            for name in expected_tensors(i as u32, kind, self.config.is_moe()) {
                require_tensor(gguf, &name)?;
            }
        }
        // MTP NextN blocks (if any).
        for (i, kind) in self.mtp_block_kinds() {
            for name in expected_tensors(i, kind, self.config.is_moe()) {
                require_tensor(gguf, &name)?;
            }
        }
        Ok(())
    }
}

/// GGUF tensor names a block of the given kind requires. `moe` selects
/// the MoE FFN tensor set (routed experts + shared expert) over the
/// dense `ffn_gate/up/down`.
pub fn expected_tensors(layer: u32, kind: BlockKind, moe: bool) -> Vec<String> {
    let mut names = vec![
        format!("blk.{layer}.attn_norm.weight"),
        format!("blk.{layer}.post_attention_norm.weight"),
    ];
    if moe {
        names.extend([
            format!("blk.{layer}.ffn_gate_inp.weight"),
            format!("blk.{layer}.ffn_gate_exps.weight"),
            format!("blk.{layer}.ffn_up_exps.weight"),
            format!("blk.{layer}.ffn_down_exps.weight"),
            format!("blk.{layer}.ffn_gate_inp_shexp.weight"),
            format!("blk.{layer}.ffn_gate_shexp.weight"),
            format!("blk.{layer}.ffn_up_shexp.weight"),
            format!("blk.{layer}.ffn_down_shexp.weight"),
        ]);
    } else {
        names.extend([
            format!("blk.{layer}.ffn_gate.weight"),
            format!("blk.{layer}.ffn_up.weight"),
            format!("blk.{layer}.ffn_down.weight"),
        ]);
    }
    match kind {
        BlockKind::LinearAttention => {
            names.extend([
                format!("blk.{layer}.attn_qkv.weight"),
                format!("blk.{layer}.attn_gate.weight"),
                format!("blk.{layer}.ssm_a"),
                format!("blk.{layer}.ssm_alpha.weight"),
                format!("blk.{layer}.ssm_beta.weight"),
                format!("blk.{layer}.ssm_conv1d.weight"),
                format!("blk.{layer}.ssm_dt.bias"),
                format!("blk.{layer}.ssm_norm.weight"),
                format!("blk.{layer}.ssm_out.weight"),
            ]);
        }
        BlockKind::FullAttention => {
            names.extend([
                format!("blk.{layer}.attn_q.weight"),
                format!("blk.{layer}.attn_k.weight"),
                format!("blk.{layer}.attn_v.weight"),
                format!("blk.{layer}.attn_q_norm.weight"),
                format!("blk.{layer}.attn_k_norm.weight"),
                format!("blk.{layer}.attn_output.weight"),
            ]);
        }
        BlockKind::NextN => {
            // Same attention layout as FullAttention plus the MTP-specific
            // weights. The eh_proj concatenates (enorm(embed), hnorm(prev
            // hidden)) — output dim is 2*hidden, input is hidden.
            names.extend([
                format!("blk.{layer}.attn_q.weight"),
                format!("blk.{layer}.attn_k.weight"),
                format!("blk.{layer}.attn_v.weight"),
                format!("blk.{layer}.attn_q_norm.weight"),
                format!("blk.{layer}.attn_k_norm.weight"),
                format!("blk.{layer}.attn_output.weight"),
                format!("blk.{layer}.nextn.eh_proj.weight"),
                format!("blk.{layer}.nextn.enorm.weight"),
                format!("blk.{layer}.nextn.hnorm.weight"),
                format!("blk.{layer}.nextn.shared_head_norm.weight"),
            ]);
        }
    }
    names
}

// ---- helpers -----------------------------------------------------------------

/// Build a `<prefix>.<suffix>` metadata key.
fn k(prefix: &str, suffix: &str) -> String {
    format!("{prefix}.{suffix}")
}

fn require_metadata<'a>(gguf: &'a GgufFile, key: &str) -> Result<&'a MetaValue> {
    gguf.metadata_get(key).ok_or_else(|| Qwen35Error::MissingMetadata(key.to_owned()))
}

fn require_str<'a>(gguf: &'a GgufFile, key: &str) -> Result<&'a str> {
    require_metadata(gguf, key)?
        .as_str()
        .ok_or_else(|| Qwen35Error::WrongMetadataType { key: key.to_owned(), expected: "string" })
}

fn require_u32(gguf: &GgufFile, key: &str) -> Result<u32> {
    require_metadata(gguf, key)?
        .as_u32()
        .ok_or_else(|| Qwen35Error::WrongMetadataType {
            key: key.to_owned(), expected: "u32-compatible integer" })
}

fn require_f32(gguf: &GgufFile, key: &str) -> Result<f32> {
    require_metadata(gguf, key)?
        .as_f32()
        .ok_or_else(|| Qwen35Error::WrongMetadataType {
            key: key.to_owned(), expected: "f32-compatible float" })
}

fn read_u32_array(gguf: &GgufFile, key: &str) -> Result<[u32; 4]> {
    let v = require_metadata(gguf, key)?;
    let (_, arr) = v.as_array()
        .ok_or_else(|| Qwen35Error::WrongMetadataType { key: key.to_owned(), expected: "array" })?;
    let mut out = [0u32; 4];
    for (i, slot) in out.iter_mut().enumerate() {
        if let Some(elem) = arr.get(i) {
            *slot = elem.as_u32().ok_or_else(|| Qwen35Error::WrongMetadataType {
                key: key.to_owned(), expected: "array of u32-compatible integers" })?;
        }
    }
    Ok(out)
}

fn require_tensor(gguf: &GgufFile, name: &str) -> Result<()> {
    if gguf.tensor(name).is_none() {
        return Err(Qwen35Error::MissingTensor(name.to_owned()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_config() -> Qwen35Config {
        Qwen35Config {
            arch: "qwen35".into(),
            block_count: 24, hidden_size: 1024, ffn_size: 3584, vocab_size: 248320,
            context_length: 262144, rms_norm_eps: 1e-6, eos_token_id: 248046,
            attn_n_heads: 8, attn_n_kv_heads: 2, attn_head_dim: 256,
            gdn_value_dim: 2048, gdn_n_heads: 16, gdn_n_k_heads: 16,
            gdn_head_dim: 128, gdn_conv_kernel: 4,
            rope_freq_base: 1e7, rope_dim_count: 64, rope_dim_sections: [11, 11, 10, 0],
            full_attention_interval: 4, tied_embeddings: true, moe: None,
            nextn_predict_layers: 0,
        }
    }

    #[test]
    fn block_schedule_matches_full_attention_interval() {
        let cfg = synth_config();
        let kinds: Vec<BlockKind> = (0..cfg.block_count).map(|i| cfg.block_kind(i)).collect();
        assert_eq!(kinds.len(), 24);
        for i in 0..24 {
            let expected = if (i + 1) % 4 == 0 {
                BlockKind::FullAttention
            } else {
                BlockKind::LinearAttention
            };
            assert_eq!(kinds[i as usize], expected, "block {i}");
        }
        assert_eq!(kinds.iter().filter(|k| **k == BlockKind::FullAttention).count(), 6);
    }

    #[test]
    fn linear_attention_block_has_ssm_tensors() {
        let names = expected_tensors(0, BlockKind::LinearAttention, false);
        assert!(names.iter().any(|n| n == "blk.0.attn_qkv.weight"));
        assert!(names.iter().any(|n| n == "blk.0.ssm_conv1d.weight"));
        assert!(names.iter().any(|n| n == "blk.0.ffn_gate.weight"));
        assert!(!names.iter().any(|n| n == "blk.0.attn_q.weight"));
    }

    #[test]
    fn full_attention_block_has_qkv_split_and_qk_norm() {
        let names = expected_tensors(3, BlockKind::FullAttention, false);
        assert!(names.iter().any(|n| n == "blk.3.attn_q.weight"));
        assert!(names.iter().any(|n| n == "blk.3.attn_q_norm.weight"));
        assert!(!names.iter().any(|n| n.contains("ssm_")));
    }

    #[test]
    fn moe_block_has_expert_tensors() {
        let names = expected_tensors(0, BlockKind::LinearAttention, true);
        assert!(names.iter().any(|n| n == "blk.0.ffn_gate_exps.weight"));
        assert!(names.iter().any(|n| n == "blk.0.ffn_down_exps.weight"));
        assert!(names.iter().any(|n| n == "blk.0.ffn_gate_inp.weight"));
        assert!(names.iter().any(|n| n == "blk.0.ffn_gate_shexp.weight"));
        assert!(names.iter().any(|n| n == "blk.0.ffn_gate_inp_shexp.weight"));
        assert!(!names.iter().any(|n| n == "blk.0.ffn_gate.weight"));
    }
}
