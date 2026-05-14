//! CPU oracle for Qwen 3.5 inference: weight dequant cache + forward pass.
//!
//! The current cache is **eager**: every weight tensor in the GGUF file is
//! dequantized to f32 at load time. For 0.8B at UD-Q4_K_XL this consumes
//! ~3 GB of host RAM. That cost is acceptable for an oracle whose purpose is
//! correctness validation; the production HIP path will keep weights quantized
//! in HBM and dequantize per-tile inside fused kernels.

use crate::cpu::ops;
use crate::cpu::rope::{apply_rope, RopeCache};
use crate::gguf::{GgufError, GgufFile, TensorInfo};
use crate::model::qwen3_5::{BlockKind, Qwen35Config, Qwen35Error, Qwen35Model};
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

// ---- Full-attention block ------------------------------------------------

/// Per-layer KV cache for full-attention blocks. Holds K and V up to a
/// pre-allocated maximum sequence length; only positions `[0, len)` are
/// valid at any given time.
pub struct LayerKvCache {
    k: Vec<f32>,
    v: Vec<f32>,
    pub max_seq: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    len: usize,
}

impl LayerKvCache {
    pub fn new(max_seq: usize, n_kv_heads: usize, head_dim: usize) -> Self {
        let n = max_seq * n_kv_heads * head_dim;
        Self {
            k: vec![0.0; n],
            v: vec![0.0; n],
            max_seq, n_kv_heads, head_dim, len: 0,
        }
    }

    pub fn len(&self) -> usize { self.len }
    pub fn is_empty(&self) -> bool { self.len == 0 }
    pub fn reset(&mut self) { self.len = 0; }

    /// Append one timestep's K and V (each [n_kv_heads * head_dim] flat).
    pub fn push(&mut self, k_step: &[f32], v_step: &[f32]) {
        let stride = self.n_kv_heads * self.head_dim;
        assert_eq!(k_step.len(), stride);
        assert_eq!(v_step.len(), stride);
        assert!(self.len < self.max_seq, "KV cache overflow");
        let off = self.len * stride;
        self.k[off..off + stride].copy_from_slice(k_step);
        self.v[off..off + stride].copy_from_slice(v_step);
        self.len += 1;
    }

    /// Slice the K row for KV-head `kv_h` at position `t`.
    fn k_at(&self, t: usize, kv_h: usize) -> &[f32] {
        let stride = self.n_kv_heads * self.head_dim;
        let off = t * stride + kv_h * self.head_dim;
        &self.k[off..off + self.head_dim]
    }

    fn v_at(&self, t: usize, kv_h: usize) -> &[f32] {
        let stride = self.n_kv_heads * self.head_dim;
        let off = t * stride + kv_h * self.head_dim;
        &self.v[off..off + self.head_dim]
    }
}

/// One step of the full-attention block (Qwen 3.5: GQA + per-head QK-norm
/// + per-head Q-side output gate + partial RoPE on first `rope_dim_count`
/// values of each head).
///
/// `out` receives the attention block's contribution (BEFORE residual add
/// against `hidden`). The caller does the residual.
#[allow(clippy::too_many_arguments)]
pub fn full_attention_step(
    hidden: &[f32],                     // [hidden_size]
    weights: &FullAttnWeights,
    config: &Qwen35Config,
    layer_kv: &mut LayerKvCache,
    rope: &RopeCache,
    pos: usize,
    out: &mut [f32],                    // [hidden_size]
) {
    let h_dim = config.hidden_size as usize;
    let head = config.attn_head_dim as usize;
    let nq = config.attn_n_heads as usize;
    let nkv = config.attn_n_kv_heads as usize;
    let groups = nq / nkv;
    let q_dim = nq * head;              // 2048
    let kv_dim = nkv * head;            // 512
    let scaling = (head as f32).powf(-0.5);

    assert_eq!(hidden.len(), h_dim);
    assert_eq!(out.len(), h_dim);

    let mut normed = vec![0.0_f32; h_dim];
    ops::rmsnorm(hidden, &weights.attn_norm, config.rms_norm_eps, &mut normed);

    // q_raw layout per token: 2 * nq * head = nq * (head * 2). The PyTorch
    // .view(..., -1, head*2) followed by chunk(2, dim=-1) means: for head h,
    // bytes [h*head*2 .. h*head*2 + head] = Q, [h*head*2+head .. h*head*2+head*2] = gate.
    let mut q_raw = vec![0.0_f32; 2 * q_dim];
    let mut k_raw = vec![0.0_f32; kv_dim];
    let mut v_raw = vec![0.0_f32; kv_dim];
    ops::matvec(&normed, &weights.attn_q, h_dim, 2 * q_dim, &mut q_raw);
    ops::matvec(&normed, &weights.attn_k, h_dim, kv_dim, &mut k_raw);
    ops::matvec(&normed, &weights.attn_v, h_dim, kv_dim, &mut v_raw);

    // Split q_raw into per-head Q + gate. Apply q_norm to Q (1+w semantics)
    // and RoPE in place on the first rope_dim_count values.
    let mut q = vec![0.0_f32; q_dim];      // [nq * head]
    let mut gate = vec![0.0_f32; q_dim];   // [nq * head]
    let mut tmp = vec![0.0_f32; head];
    for h in 0..nq {
        let src = &q_raw[h * 2 * head..(h + 1) * 2 * head];
        ops::rmsnorm(&src[..head], &weights.attn_q_norm, config.rms_norm_eps, &mut tmp);
        let dst = &mut q[h * head..(h + 1) * head];
        dst.copy_from_slice(&tmp);
        apply_rope(dst, rope, pos);

        // Gate is the second half of this head's slot in q_raw.
        gate[h * head..(h + 1) * head].copy_from_slice(&src[head..]);
    }

    // K: per-kv-head normalize + RoPE.
    let mut k_norm_buf = vec![0.0_f32; kv_dim];
    for h in 0..nkv {
        let src = &k_raw[h * head..(h + 1) * head];
        let dst = &mut k_norm_buf[h * head..(h + 1) * head];
        ops::rmsnorm(src, &weights.attn_k_norm, config.rms_norm_eps, dst);
        apply_rope(dst, rope, pos);
    }

    // Append (k, v) for this position to the KV cache.
    layer_kv.push(&k_norm_buf, &v_raw);
    let cache_len = layer_kv.len();

    // Attention: per Q head, compute scores against all cached K[t], softmax,
    // weighted sum of V[t]. Causal mask = "all positions in [0, cache_len)
    // are valid" since we just pushed pos and have only ever pushed positions
    // <= pos.
    let mut attn_concat = vec![0.0_f32; q_dim];
    let mut scores = vec![0.0_f32; cache_len];
    for h in 0..nq {
        let kv_h = h / groups;
        let q_h = &q[h * head..(h + 1) * head];

        for t in 0..cache_len {
            let k_t = layer_kv.k_at(t, kv_h);
            let mut acc = 0.0_f32;
            for d in 0..head {
                acc += q_h[d] * k_t[d];
            }
            scores[t] = acc * scaling;
        }
        ops::softmax(&mut scores);

        let head_out = &mut attn_concat[h * head..(h + 1) * head];
        head_out.fill(0.0);
        for t in 0..cache_len {
            let v_t = layer_kv.v_at(t, kv_h);
            let s = scores[t];
            for d in 0..head {
                head_out[d] += s * v_t[d];
            }
        }
    }

    // Output gate: attn_concat *= sigmoid(gate). Then o_proj.
    for i in 0..q_dim {
        attn_concat[i] *= ops::sigmoid(gate[i]);
    }
    ops::matvec(&attn_concat, &weights.attn_output, q_dim, h_dim, out);
}

// ---- SwiGLU FFN (shared by both block types) ------------------------------

/// SwiGLU feed-forward: `down( silu(gate(x)) * up(x) )`.
/// `out` is the FFN's contribution; caller adds residual.
pub fn swiglu_ffn(
    x: &[f32],                     // [hidden]
    gate_w: &[f32],                // [hidden, ffn]
    up_w: &[f32],                  // [hidden, ffn]
    down_w: &[f32],                // [ffn, hidden]
    hidden_size: usize,
    ffn_size: usize,
    out: &mut [f32],               // [hidden]
) {
    assert_eq!(x.len(), hidden_size);
    assert_eq!(out.len(), hidden_size);
    let mut gate = vec![0.0_f32; ffn_size];
    let mut up   = vec![0.0_f32; ffn_size];
    ops::matvec(x, gate_w, hidden_size, ffn_size, &mut gate);
    ops::matvec(x, up_w,   hidden_size, ffn_size, &mut up);
    ops::swiglu_mul(&mut gate, &up);
    ops::matvec(&gate, down_w, ffn_size, hidden_size, out);
}

// ---- Block forward (norm → attn → residual → norm → ffn → residual) -------

/// One full-attention block applied in place to `hidden_inout`. Combines:
///   x ← x + attn(rmsnorm(x))
///   x ← x + ffn(rmsnorm(x))
pub fn full_attention_block(
    hidden_inout: &mut [f32],         // [hidden]
    weights: &FullAttnWeights,
    config: &Qwen35Config,
    layer_kv: &mut LayerKvCache,
    rope: &RopeCache,
    pos: usize,
) {
    let h = config.hidden_size as usize;
    let f = config.ffn_size as usize;
    assert_eq!(hidden_inout.len(), h);

    // Sub-layer 1: attention with pre-norm.
    let mut attn_out = vec![0.0_f32; h];
    full_attention_step(hidden_inout, weights, config, layer_kv, rope, pos, &mut attn_out);
    ops::add_(hidden_inout, &attn_out);

    // Sub-layer 2: FFN with pre-norm.
    let mut normed = vec![0.0_f32; h];
    ops::rmsnorm(hidden_inout, &weights.post_attention_norm, config.rms_norm_eps, &mut normed);
    let mut ffn_out = vec![0.0_f32; h];
    swiglu_ffn(&normed, &weights.ffn_gate, &weights.ffn_up, &weights.ffn_down, h, f, &mut ffn_out);
    ops::add_(hidden_inout, &ffn_out);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kv_cache_push_read_round_trip() {
        let mut c = LayerKvCache::new(8, 2, 4);
        // Push two timesteps.
        c.push(&[1.0, 2.0, 3.0, 4.0,    5.0, 6.0, 7.0, 8.0],
               &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]);
        c.push(&[-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0],
               &[-10.0, -20.0, -30.0, -40.0, -50.0, -60.0, -70.0, -80.0]);

        assert_eq!(c.len(), 2);
        assert_eq!(c.k_at(0, 0), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(c.k_at(0, 1), &[5.0, 6.0, 7.0, 8.0]);
        assert_eq!(c.v_at(1, 0), &[-10.0, -20.0, -30.0, -40.0]);
        assert_eq!(c.v_at(1, 1), &[-50.0, -60.0, -70.0, -80.0]);

        c.reset();
        assert_eq!(c.len(), 0);
    }

    #[test]
    fn swiglu_ffn_with_zero_gate_outputs_zero() {
        // gate(x) = 0 → silu(0)*up = 0 → down(0) = 0 regardless of up/down weights.
        let h = 4;
        let f = 6;
        let x = vec![0.5_f32, -1.0, 0.25, 2.0];
        let gate_w = vec![0.0_f32; h * f];
        let up_w   = vec![1.0_f32; h * f];
        let down_w = vec![1.0_f32; f * h];
        let mut out = vec![0.0_f32; h];
        swiglu_ffn(&x, &gate_w, &up_w, &down_w, h, f, &mut out);
        for v in &out {
            assert_eq!(*v, 0.0, "expected 0 with zero gate, got {v}");
        }
    }
}
