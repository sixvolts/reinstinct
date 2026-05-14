//! CPU oracle for Qwen 3.5 inference: weight dequant cache + (later) forward pass.
//!
//! The current cache is **eager**: every weight tensor in the GGUF file is
//! dequantized to f32 at load time. For 0.8B at UD-Q4_K_XL this consumes
//! ~3 GB of host RAM. That cost is acceptable for an oracle whose purpose is
//! correctness validation; the production HIP path will keep weights quantized
//! in HBM and dequantize per-tile inside fused kernels.

use crate::gguf::{GgufError, GgufFile, TensorInfo};
use crate::model::qwen3_5::{BlockKind, Qwen35Error, Qwen35Model};
use crate::quant::dequantize_tensor;

/// All Qwen 3.5 weights as f32 buffers, organized by block.
pub struct Qwen35F32Weights {
    pub token_embd: Vec<f32>,           // ggml [hidden, vocab] → flat w[v*hidden + h]
    pub output_norm: Vec<f32>,           // [hidden]
    /// Separate output projection if `tied_embeddings == false`.
    pub output: Option<Vec<f32>>,
    pub blocks: Vec<BlockWeights>,
}

pub enum BlockWeights {
    LinearAttention(LinAttnWeights),
    FullAttention(FullAttnWeights),
}

/// Linear-attention (Gated DeltaNet) block — 14 weight tensors per layer.
///
/// Shapes annotated for Qwen 3.5 0.8B (hidden=1024, ffn=3584,
/// gdn_value_dim=2048, gdn_n_heads=16, gdn_head_dim=128, kernel=4).
pub struct LinAttnWeights {
    pub attn_norm: Vec<f32>,            // [hidden]
    pub attn_qkv: Vec<f32>,             // [hidden, 3*value_dim]
    pub attn_gate: Vec<f32>,            // [hidden, value_dim]
    pub ssm_alpha: Vec<f32>,            // [hidden, n_heads]
    pub ssm_beta: Vec<f32>,             // [hidden, n_heads]
    pub ssm_a: Vec<f32>,                // [n_heads]                — A_log diagonal
    pub ssm_dt_bias: Vec<f32>,          // [n_heads]
    pub ssm_conv1d: Vec<f32>,           // [n_channels, kernel]     — depthwise
    pub ssm_norm: Vec<f32>,             // [head_v_dim]             — per-head RMSNormGated weight
    pub ssm_out: Vec<f32>,              // [value_dim, hidden]
    pub post_attention_norm: Vec<f32>,  // [hidden]
    pub ffn_gate: Vec<f32>,             // [hidden, ffn]
    pub ffn_up: Vec<f32>,               // [hidden, ffn]
    pub ffn_down: Vec<f32>,             // [ffn, hidden]
}

/// Full-attention block — GQA + QK-norm + output gate (Q proj outputs 2× width).
pub struct FullAttnWeights {
    pub attn_norm: Vec<f32>,            // [hidden]
    pub attn_q: Vec<f32>,               // [hidden, 2 * n_heads * head_dim]   — concat(Q, Q_gate)
    pub attn_k: Vec<f32>,               // [hidden, n_kv_heads * head_dim]
    pub attn_v: Vec<f32>,               // [hidden, n_kv_heads * head_dim]
    pub attn_q_norm: Vec<f32>,          // [head_dim]                          — per-head Q RMSNorm
    pub attn_k_norm: Vec<f32>,          // [head_dim]
    pub attn_output: Vec<f32>,          // [n_heads * head_dim, hidden]
    pub post_attention_norm: Vec<f32>,  // [hidden]
    pub ffn_gate: Vec<f32>,             // [hidden, ffn]
    pub ffn_up: Vec<f32>,               // [hidden, ffn]
    pub ffn_down: Vec<f32>,             // [ffn, hidden]
}

#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error(transparent)]
    Qwen35(#[from] Qwen35Error),

    #[error(transparent)]
    Gguf(#[from] GgufError),
}

impl Qwen35F32Weights {
    /// Eagerly dequantize every Qwen 3.5 weight tensor in the file.
    ///
    /// Caller must have already loaded `Qwen35Model::load(&gguf)` to get the
    /// validated config + block schedule.
    pub fn load(gguf: &GgufFile, model: &Qwen35Model) -> Result<Self, LoadError> {
        let token_embd = dequant_named(gguf, "token_embd.weight")?;
        let output_norm = dequant_named(gguf, "output_norm.weight")?;
        let output = if model.config.tied_embeddings {
            None
        } else {
            Some(dequant_named(gguf, "output.weight")?)
        };

        let mut blocks = Vec::with_capacity(model.block_kinds.len());
        for (i, &kind) in model.block_kinds.iter().enumerate() {
            let layer = i as u32;
            blocks.push(match kind {
                BlockKind::LinearAttention => BlockWeights::LinearAttention(
                    load_linear_attention(gguf, layer)?
                ),
                BlockKind::FullAttention => BlockWeights::FullAttention(
                    load_full_attention(gguf, layer)?
                ),
            });
        }

        Ok(Self { token_embd, output_norm, output, blocks })
    }
}

fn load_linear_attention(gguf: &GgufFile, layer: u32) -> Result<LinAttnWeights, LoadError> {
    Ok(LinAttnWeights {
        attn_norm:           dequant_named(gguf, &format!("blk.{layer}.attn_norm.weight"))?,
        attn_qkv:            dequant_named(gguf, &format!("blk.{layer}.attn_qkv.weight"))?,
        attn_gate:           dequant_named(gguf, &format!("blk.{layer}.attn_gate.weight"))?,
        ssm_alpha:           dequant_named(gguf, &format!("blk.{layer}.ssm_alpha.weight"))?,
        ssm_beta:            dequant_named(gguf, &format!("blk.{layer}.ssm_beta.weight"))?,
        ssm_a:               dequant_named(gguf, &format!("blk.{layer}.ssm_a"))?,
        ssm_dt_bias:         dequant_named(gguf, &format!("blk.{layer}.ssm_dt.bias"))?,
        ssm_conv1d:          dequant_named(gguf, &format!("blk.{layer}.ssm_conv1d.weight"))?,
        ssm_norm:            dequant_named(gguf, &format!("blk.{layer}.ssm_norm.weight"))?,
        ssm_out:             dequant_named(gguf, &format!("blk.{layer}.ssm_out.weight"))?,
        post_attention_norm: dequant_named(gguf, &format!("blk.{layer}.post_attention_norm.weight"))?,
        ffn_gate:            dequant_named(gguf, &format!("blk.{layer}.ffn_gate.weight"))?,
        ffn_up:              dequant_named(gguf, &format!("blk.{layer}.ffn_up.weight"))?,
        ffn_down:            dequant_named(gguf, &format!("blk.{layer}.ffn_down.weight"))?,
    })
}

fn load_full_attention(gguf: &GgufFile, layer: u32) -> Result<FullAttnWeights, LoadError> {
    Ok(FullAttnWeights {
        attn_norm:           dequant_named(gguf, &format!("blk.{layer}.attn_norm.weight"))?,
        attn_q:              dequant_named(gguf, &format!("blk.{layer}.attn_q.weight"))?,
        attn_k:              dequant_named(gguf, &format!("blk.{layer}.attn_k.weight"))?,
        attn_v:              dequant_named(gguf, &format!("blk.{layer}.attn_v.weight"))?,
        attn_q_norm:         dequant_named(gguf, &format!("blk.{layer}.attn_q_norm.weight"))?,
        attn_k_norm:         dequant_named(gguf, &format!("blk.{layer}.attn_k_norm.weight"))?,
        attn_output:         dequant_named(gguf, &format!("blk.{layer}.attn_output.weight"))?,
        post_attention_norm: dequant_named(gguf, &format!("blk.{layer}.post_attention_norm.weight"))?,
        ffn_gate:            dequant_named(gguf, &format!("blk.{layer}.ffn_gate.weight"))?,
        ffn_up:              dequant_named(gguf, &format!("blk.{layer}.ffn_up.weight"))?,
        ffn_down:            dequant_named(gguf, &format!("blk.{layer}.ffn_down.weight"))?,
    })
}

fn dequant_named(gguf: &GgufFile, name: &str) -> Result<Vec<f32>, LoadError> {
    let info: &TensorInfo = gguf.tensor(name).ok_or_else(|| {
        // Should never trigger after Qwen35Model::load — kept for defense in depth.
        LoadError::Gguf(GgufError::Io(std::io::Error::other(
            format!("tensor {name} not present"),
        )))
    })?;
    let bytes = gguf.tensor_data(name)?.expect("tensor_data missing after lookup");
    Ok(dequantize_tensor(info, bytes)?)
}
