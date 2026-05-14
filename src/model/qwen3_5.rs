//! Qwen 3.5: typed config + tensor binding for the hybrid Gated-DeltaNet
//! / GQA-attention architecture.
//!
//! Layer schedule alternates `linear_attention` and `full_attention` per
//! `qwen35.full_attention_interval` (4 in the reference 0.8B model — pattern
//! `[L,L,L,F]×6`). The two block types load distinct tensor sets; the
//! validator below enforces full presence at load time.
//!
//! See `gfx906-inference-engine-design.md` for the broader engine plan and
//! the project memory `project_qwen35_tensor_map.md` for the GGUF→PyTorch
//! parameter mapping this module implements.

use thiserror::Error;

use crate::gguf::{GgmlType, GgufFile, MetaValue};

const ARCH: &str = "qwen35";

#[derive(Debug, Error)]
pub enum Qwen35Error {
    #[error("not a Qwen 3.5 file: general.architecture = {got:?}, expected {expected:?}")]
    WrongArchitecture { got: String, expected: &'static str },

    #[error("missing required GGUF metadata key: {0}")]
    MissingMetadata(&'static str),

    #[error("metadata key {key} has wrong type (expected {expected})")]
    WrongMetadataType { key: &'static str, expected: &'static str },

    #[error("missing required tensor: {0}")]
    MissingTensor(String),

    #[error("tensor {name} has unexpected type {got:?} (expected one of {expected:?})")]
    WrongTensorType { name: String, got: GgmlType, expected: Vec<GgmlType> },

    #[error("tensor {name} has unexpected shape {got:?} (expected {expected:?})")]
    WrongTensorShape { name: String, got: Vec<u64>, expected: Vec<u64> },
}

type Result<T> = std::result::Result<T, Qwen35Error>;

/// Hyperparameters loaded from `qwen35.*` GGUF metadata plus a few inferred
/// values (vocab size from `token_embd` shape, tied embeddings from absence
/// of `output.weight`).
#[derive(Debug, Clone)]
pub struct Qwen35Config {
    // Topology
    pub block_count: u32,
    pub hidden_size: u32,
    pub ffn_size: u32,
    pub vocab_size: u32,
    pub context_length: u32,
    pub rms_norm_eps: f32,
    pub eos_token_id: u32,

    // Full-attention block (every Nth block per full_attention_interval)
    pub attn_n_heads: u32,
    pub attn_n_kv_heads: u32,
    pub attn_head_dim: u32,

    // Linear-attention (Gated DeltaNet) block
    /// value_dim = num_v_heads × value_head_dim. From `qwen35.ssm.inner_size`.
    pub gdn_value_dim: u32,
    /// = num_v_heads. From `qwen35.ssm.group_count`.
    pub gdn_n_heads: u32,
    /// Per-head dim. Both K and V share this in Qwen 3.5. From `qwen35.ssm.state_size`.
    pub gdn_head_dim: u32,
    pub gdn_conv_kernel: u32,

    // RoPE
    pub rope_freq_base: f32,
    /// Number of head-dim values that get rotated (partial RoPE). For Qwen 3.5
    /// 0.8B this is 64 of 256 → partial_rotary_factor = 0.25.
    pub rope_dim_count: u32,
    /// M-RoPE per-axis section sizes. Stored as a fixed array padded with 0.
    /// Used for vision tokens; pure-text inference treats all as one axis.
    pub rope_dim_sections: [u32; 4],

    // Layer schedule
    pub full_attention_interval: u32,

    // Embedding
    pub tied_embeddings: bool,
}

impl Qwen35Config {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let arch = require_str(gguf, "general.architecture")?;
        if arch != ARCH {
            return Err(Qwen35Error::WrongArchitecture {
                got: arch.to_owned(),
                expected: ARCH,
            });
        }

        let block_count             = require_u32(gguf, "qwen35.block_count")?;
        let hidden_size             = require_u32(gguf, "qwen35.embedding_length")?;
        let ffn_size                = require_u32(gguf, "qwen35.feed_forward_length")?;
        let context_length          = require_u32(gguf, "qwen35.context_length")?;
        let rms_norm_eps            = require_f32(gguf, "qwen35.attention.layer_norm_rms_epsilon")?;
        let attn_n_heads            = require_u32(gguf, "qwen35.attention.head_count")?;
        let attn_n_kv_heads         = require_u32(gguf, "qwen35.attention.head_count_kv")?;
        let attn_head_dim           = require_u32(gguf, "qwen35.attention.key_length")?;
        let gdn_value_dim           = require_u32(gguf, "qwen35.ssm.inner_size")?;
        let gdn_n_heads             = require_u32(gguf, "qwen35.ssm.group_count")?;
        let gdn_head_dim            = require_u32(gguf, "qwen35.ssm.state_size")?;
        let gdn_conv_kernel         = require_u32(gguf, "qwen35.ssm.conv_kernel")?;
        let rope_freq_base          = require_f32(gguf, "qwen35.rope.freq_base")?;
        let rope_dim_count          = require_u32(gguf, "qwen35.rope.dimension_count")?;
        let full_attention_interval = require_u32(gguf, "qwen35.full_attention_interval")?;
        let eos_token_id            = require_u32(gguf, "tokenizer.ggml.eos_token_id")?;

        let rope_dim_sections = read_u32_array(gguf, "qwen35.rope.dimension_sections")?;

        // Vocab size: read from the token embedding tensor's shape rather
        // than trusting metadata, since they should agree but the tensor
        // is the authoritative source for what the model actually projects to.
        let token_embd = gguf.tensor("token_embd.weight")
            .ok_or_else(|| Qwen35Error::MissingTensor("token_embd.weight".into()))?;
        if token_embd.shape().len() != 2 {
            return Err(Qwen35Error::WrongTensorShape {
                name: "token_embd.weight".into(),
                got: token_embd.shape().to_vec(),
                expected: vec![hidden_size as u64, 0],
            });
        }
        if token_embd.shape()[0] != hidden_size as u64 {
            return Err(Qwen35Error::WrongTensorShape {
                name: "token_embd.weight".into(),
                got: token_embd.shape().to_vec(),
                expected: vec![hidden_size as u64, token_embd.shape()[1]],
            });
        }
        let vocab_size = token_embd.shape()[1] as u32;

        // Tied embeddings if there's no separate output projection.
        let tied_embeddings = gguf.tensor("output.weight").is_none();

        Ok(Self {
            block_count, hidden_size, ffn_size, vocab_size, context_length,
            rms_norm_eps, eos_token_id,
            attn_n_heads, attn_n_kv_heads, attn_head_dim,
            gdn_value_dim, gdn_n_heads, gdn_head_dim, gdn_conv_kernel,
            rope_freq_base, rope_dim_count, rope_dim_sections,
            full_attention_interval,
            tied_embeddings,
        })
    }

    /// Per-layer block type. Pattern is `[L]×(N-1) [F]` repeated
    /// `block_count / full_attention_interval` times — i.e. blocks where
    /// `(idx + 1) % full_attention_interval == 0` are full attention.
    pub fn block_kind(&self, layer_idx: u32) -> BlockKind {
        if (layer_idx + 1) % self.full_attention_interval == 0 {
            BlockKind::FullAttention
        } else {
            BlockKind::LinearAttention
        }
    }

    /// Linear-attention key/value head dim (both equal in Qwen 3.5).
    pub fn gdn_value_head_dim(&self) -> u32 {
        self.gdn_value_dim / self.gdn_n_heads
    }

    /// Linear-attention input projection output dim:
    ///   2 × key_dim + value_dim = 3 × value_dim   (since num_k_heads = num_v_heads, head_dim equal)
    pub fn gdn_qkv_concat_dim(&self) -> u32 {
        2 * self.gdn_value_dim + self.gdn_value_dim
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlockKind {
    LinearAttention,
    FullAttention,
}

/// Loaded Qwen 3.5 model: config + per-layer block schedule. Tensor data
/// stays in the underlying `GgufFile` mmap; this struct only holds the
/// parsed metadata and validates that every expected weight is present.
#[derive(Debug, Clone)]
pub struct Qwen35Model {
    pub config: Qwen35Config,
    pub block_kinds: Vec<BlockKind>,
}

impl Qwen35Model {
    pub fn load(gguf: &GgufFile) -> Result<Self> {
        let config = Qwen35Config::from_gguf(gguf)?;
        let block_kinds: Vec<BlockKind> = (0..config.block_count)
            .map(|i| config.block_kind(i))
            .collect();
        let model = Self { config, block_kinds };
        model.validate_tensor_presence(gguf)?;
        Ok(model)
    }

    fn validate_tensor_presence(&self, gguf: &GgufFile) -> Result<()> {
        require_tensor(gguf, "token_embd.weight")?;
        require_tensor(gguf, "output_norm.weight")?;
        if !self.config.tied_embeddings {
            require_tensor(gguf, "output.weight")?;
        }
        for (i, &kind) in self.block_kinds.iter().enumerate() {
            for name in expected_tensors(i as u32, kind) {
                require_tensor(gguf, &name)?;
            }
        }
        Ok(())
    }
}

/// Iterate the GGUF tensor names a block of the given kind requires.
pub fn expected_tensors(layer: u32, kind: BlockKind) -> Vec<String> {
    let mut names = vec![
        format!("blk.{layer}.attn_norm.weight"),
        format!("blk.{layer}.post_attention_norm.weight"),
        format!("blk.{layer}.ffn_gate.weight"),
        format!("blk.{layer}.ffn_up.weight"),
        format!("blk.{layer}.ffn_down.weight"),
    ];
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
    }
    names
}

// ---- helpers -----------------------------------------------------------------

fn require_metadata<'a>(gguf: &'a GgufFile, key: &'static str) -> Result<&'a MetaValue> {
    gguf.metadata_get(key).ok_or(Qwen35Error::MissingMetadata(key))
}

fn require_str<'a>(gguf: &'a GgufFile, key: &'static str) -> Result<&'a str> {
    require_metadata(gguf, key)?
        .as_str()
        .ok_or(Qwen35Error::WrongMetadataType { key, expected: "string" })
}

fn require_u32(gguf: &GgufFile, key: &'static str) -> Result<u32> {
    require_metadata(gguf, key)?
        .as_u32()
        .ok_or(Qwen35Error::WrongMetadataType { key, expected: "u32-compatible integer" })
}

fn require_f32(gguf: &GgufFile, key: &'static str) -> Result<f32> {
    require_metadata(gguf, key)?
        .as_f32()
        .ok_or(Qwen35Error::WrongMetadataType { key, expected: "f32-compatible float" })
}

fn read_u32_array(gguf: &GgufFile, key: &'static str) -> Result<[u32; 4]> {
    let v = require_metadata(gguf, key)?;
    let (_, arr) = v.as_array()
        .ok_or(Qwen35Error::WrongMetadataType { key, expected: "array" })?;
    let mut out = [0u32; 4];
    for (i, slot) in out.iter_mut().enumerate() {
        if let Some(elem) = arr.get(i) {
            *slot = elem.as_u32().ok_or(Qwen35Error::WrongMetadataType {
                key, expected: "array of u32-compatible integers",
            })?;
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

    #[test]
    fn block_schedule_matches_full_attention_interval() {
        // Synthetic config matching the 0.8B layout.
        let cfg = Qwen35Config {
            block_count: 24, hidden_size: 1024, ffn_size: 3584, vocab_size: 248320,
            context_length: 262144, rms_norm_eps: 1e-6, eos_token_id: 248046,
            attn_n_heads: 8, attn_n_kv_heads: 2, attn_head_dim: 256,
            gdn_value_dim: 2048, gdn_n_heads: 16, gdn_head_dim: 128, gdn_conv_kernel: 4,
            rope_freq_base: 1e7, rope_dim_count: 64, rope_dim_sections: [11, 11, 10, 0],
            full_attention_interval: 4, tied_embeddings: true,
        };
        let kinds: Vec<BlockKind> = (0..cfg.block_count).map(|i| cfg.block_kind(i)).collect();
        // L,L,L,F repeated 6 times → 24 total
        assert_eq!(kinds.len(), 24);
        for i in 0..24 {
            let expected = if (i + 1) % 4 == 0 {
                BlockKind::FullAttention
            } else {
                BlockKind::LinearAttention
            };
            assert_eq!(kinds[i as usize], expected, "block {i}");
        }
        let n_full = kinds.iter().filter(|k| **k == BlockKind::FullAttention).count();
        assert_eq!(n_full, 6);
    }

    #[test]
    fn linear_attention_block_has_ssm_tensors() {
        let names = expected_tensors(0, BlockKind::LinearAttention);
        assert!(names.iter().any(|n| n == "blk.0.attn_qkv.weight"));
        assert!(names.iter().any(|n| n == "blk.0.attn_gate.weight"));
        assert!(names.iter().any(|n| n == "blk.0.ssm_conv1d.weight"));
        assert!(names.iter().any(|n| n == "blk.0.ssm_out.weight"));
        assert!(!names.iter().any(|n| n == "blk.0.attn_q.weight"));
    }

    #[test]
    fn full_attention_block_has_qkv_split_and_qk_norm() {
        let names = expected_tensors(3, BlockKind::FullAttention);
        assert!(names.iter().any(|n| n == "blk.3.attn_q.weight"));
        assert!(names.iter().any(|n| n == "blk.3.attn_k.weight"));
        assert!(names.iter().any(|n| n == "blk.3.attn_v.weight"));
        assert!(names.iter().any(|n| n == "blk.3.attn_q_norm.weight"));
        assert!(names.iter().any(|n| n == "blk.3.attn_k_norm.weight"));
        assert!(names.iter().any(|n| n == "blk.3.attn_output.weight"));
        assert!(!names.iter().any(|n| n.contains("ssm_")));
    }
}
