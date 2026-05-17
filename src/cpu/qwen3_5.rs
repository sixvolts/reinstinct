//! CPU oracle for Qwen 3.5 inference: weight dequant cache + forward pass.
//!
//! The current cache is **eager**: every weight tensor in the GGUF file is
//! dequantized to f32 at load time. For 0.8B at UD-Q4_K_XL this consumes
//! ~3 GB of host RAM. That cost is acceptable for an oracle whose purpose is
//! correctness validation; the production HIP path will keep weights quantized
//! in HBM and dequantize per-tile inside fused kernels.

use crate::cpu::conv1d::Conv1dState;
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

pub fn load_linear_attention(gguf: &GgufFile, layer: u32) -> Result<LinAttnWeights, LoadError> {
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

pub fn load_full_attention(gguf: &GgufFile, layer: u32) -> Result<FullAttnWeights, LoadError> {
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

pub fn dequant_named(gguf: &GgufFile, name: &str) -> Result<Vec<f32>, LoadError> {
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

// ---- Linear-attention (Gated DeltaNet) block ------------------------------

/// Per-layer state for a Gated-DeltaNet block.
///
/// Two pieces:
/// - `recurrent`: the per-head outer-product memory matrix S of shape
///   [n_heads, k_head_dim, v_head_dim] = [16, 128, 128] for Qwen 3.5 0.8B,
///   stored fp32 per the model's `mamba_ssm_dtype: "float32"`.
/// - `conv`: depthwise causal Conv1D ring buffer over the conv_dim
///   (= 2*key_dim + value_dim = 6144) channels of mixed_qkv.
pub struct LinAttnState {
    recurrent: Vec<f32>,
    pub conv: Conv1dState,
    pub n_heads: usize,
    pub k_head_dim: usize,
    pub v_head_dim: usize,
}

impl LinAttnState {
    pub fn new(
        n_heads: usize, k_head_dim: usize, v_head_dim: usize,
        conv_channels: usize, conv_kernel: usize,
    ) -> Self {
        Self {
            recurrent: vec![0.0; n_heads * k_head_dim * v_head_dim],
            conv: Conv1dState::new(conv_channels, conv_kernel),
            n_heads, k_head_dim, v_head_dim,
        }
    }

    pub fn reset(&mut self) {
        for v in self.recurrent.iter_mut() { *v = 0.0; }
        self.conv.reset();
    }

}

/// One step of the Gated-DeltaNet block (the `linear_attention` block type).
///
/// Implements `Qwen3_5GatedDeltaNet.forward` for `seq_len == 1` (decode form),
/// using `torch_recurrent_gated_delta_rule` semantics:
///
/// 1. project hidden → `mixed_qkv` [conv_dim], `z` [value_dim], `a` and `b` [n_heads]
/// 2. depthwise causal Conv1D + SiLU on mixed_qkv
/// 3. split mixed_qkv into Q | K | V (each [n_heads × head_dim])
/// 4. l2-norm Q and K per-head, scale Q by 1/√head_dim
/// 5. β = sigmoid(b), g = -exp(A_log) * softplus(a + dt_bias)
/// 6. per head: state ← state·exp(g); kv_mem = state · k; δ = (v - kv_mem)·β;
///    state += k ⊗ δ; out = state · q
/// 7. apply RMSNormGated per head with z as the gate
/// 8. project value_dim → hidden via ssm_out
///
/// `out` receives the block's contribution; caller adds residual.
pub fn linear_attention_step(
    hidden: &[f32],
    weights: &LinAttnWeights,
    config: &Qwen35Config,
    state: &mut LinAttnState,
    out: &mut [f32],
) {
    let h_dim = config.hidden_size as usize;
    let n_heads = config.gdn_n_heads as usize;       // value heads
    let n_k_heads = config.gdn_n_k_heads as usize;   // key/query heads (GQA)
    let head_dim = config.gdn_head_dim as usize;     // shared by K/Q and V
    let value_dim = config.gdn_value_dim as usize;   // n_heads * head_dim
    let key_dim = config.gdn_key_dim() as usize;     // n_k_heads * head_dim
    let conv_dim = 2 * key_dim + value_dim;
    let scale = (head_dim as f32).powf(-0.5);

    assert_eq!(hidden.len(), h_dim);
    assert_eq!(out.len(), h_dim);
    assert_eq!(state.n_heads, n_heads);
    assert_eq!(state.k_head_dim, head_dim);
    assert_eq!(state.v_head_dim, head_dim);

    let mut normed = vec![0.0_f32; h_dim];
    ops::rmsnorm(hidden, &weights.attn_norm, config.rms_norm_eps, &mut normed);

    // Projections (single-token, so all are matvec).
    let mut mixed_qkv = vec![0.0_f32; conv_dim];
    let mut z = vec![0.0_f32; value_dim];
    let mut a = vec![0.0_f32; n_heads];
    let mut b = vec![0.0_f32; n_heads];
    ops::matvec(&normed, &weights.attn_qkv,  h_dim, conv_dim,  &mut mixed_qkv);
    ops::matvec(&normed, &weights.attn_gate, h_dim, value_dim, &mut z);
    ops::matvec(&normed, &weights.ssm_alpha, h_dim, n_heads,   &mut a);
    ops::matvec(&normed, &weights.ssm_beta,  h_dim, n_heads,   &mut b);

    // Causal Conv1D + SiLU on mixed_qkv (per channel, kernel=4).
    let mut conv_out = vec![0.0_f32; conv_dim];
    state.conv.step(&mixed_qkv, &weights.ssm_conv1d, &mut conv_out);
    for v in conv_out.iter_mut() { *v = ops::silu(*v); }

    // Split into Q | K | V. Q/K are key_dim wide (n_k_heads heads),
    // V is value_dim wide (n_heads heads).
    let q_slice = &conv_out[0..key_dim];
    let k_slice = &conv_out[key_dim..2 * key_dim];
    let v_slice = &conv_out[2 * key_dim..2 * key_dim + value_dim];

    // Per-head L2-norm of Q and K (n_k_heads heads), then Q-side scale.
    let mut q = vec![0.0_f32; key_dim];
    let mut k = vec![0.0_f32; key_dim];
    let v: &[f32] = v_slice;
    for h in 0..n_k_heads {
        let off = h * head_dim;
        ops::l2norm(&q_slice[off..off + head_dim], 1e-6, &mut q[off..off + head_dim]);
        ops::l2norm(&k_slice[off..off + head_dim], 1e-6, &mut k[off..off + head_dim]);
    }
    for x in q.iter_mut() { *x *= scale; }

    // Per-head decay and beta. Note that `convert_hf_to_gguf.py` stores
    // `ssm_a` as `-exp(A_log)` already (Qwen3NextModel.modify_tensors does
    // `data_torch = -torch.exp(data_torch)` for `.A_log`), so we multiply
    // ssm_a directly rather than re-applying `-exp`.
    //   beta_h  = sigmoid(b_h)
    //   g_h     = ssm_a_h * softplus(a_h + dt_bias_h)        (already negative)
    //   decay_h = exp(g_h)                                    (∈ (0, 1])
    let mut decay = vec![0.0_f32; n_heads];
    let mut beta = vec![0.0_f32; n_heads];
    for h in 0..n_heads {
        let g = weights.ssm_a[h] * ops::softplus(a[h] + weights.ssm_dt_bias[h]);
        decay[h] = g.exp();
        beta[h] = ops::sigmoid(b[h]);
    }

    // Recurrent gated delta rule, parallel across heads. Each head owns a
    // disjoint slice of the recurrent matrix and produces a disjoint chunk
    // of core_attn_out, so the outer loop has no cross-head dependencies.
    let mut core_attn_out = vec![0.0_f32; value_dim];
    {
        use rayon::prelude::*;
        let head_state_size = head_dim * head_dim;
        state.recurrent
            .par_chunks_exact_mut(head_state_size)
            .zip(core_attn_out.par_chunks_exact_mut(head_dim))
            .enumerate()
            .for_each(|(h, (s, head_out))| {
                // GQA: the value heads are tiled over key heads — value
                // head h pairs with key head h % n_k_heads (not blocked).
                let kh = h % n_k_heads;
                let q_h = &q[kh * head_dim..(kh + 1) * head_dim];
                let k_h = &k[kh * head_dim..(kh + 1) * head_dim];
                let v_h = &v[h * head_dim..(h + 1) * head_dim];
                let decay_h = decay[h];
                let beta_h = beta[h];

                // state *= decay
                for x in s.iter_mut() { *x *= decay_h; }

                // kv_mem[v] = sum_k state[k, v] * k_h[k]
                let mut kv_mem = [0.0_f32; 256]; // assume head_dim ≤ 256
                let kv_mem = &mut kv_mem[..head_dim];
                kv_mem.fill(0.0);
                for kk in 0..head_dim {
                    let kv = k_h[kk];
                    let row = &s[kk * head_dim..(kk + 1) * head_dim];
                    for vv in 0..head_dim {
                        kv_mem[vv] += row[vv] * kv;
                    }
                }

                // delta[v] = (v_h[v] - kv_mem[v]) * beta_h
                let mut delta = [0.0_f32; 256];
                let delta = &mut delta[..head_dim];
                for vv in 0..head_dim {
                    delta[vv] = (v_h[vv] - kv_mem[vv]) * beta_h;
                }

                // state[k, v] += k_h[k] * delta[v]   (rank-1 outer product)
                for kk in 0..head_dim {
                    let kv = k_h[kk];
                    let row = &mut s[kk * head_dim..(kk + 1) * head_dim];
                    for vv in 0..head_dim {
                        row[vv] += kv * delta[vv];
                    }
                }

                // out[h, v] = sum_k state[k, v] * q_h[k]
                head_out.fill(0.0);
                for kk in 0..head_dim {
                    let qv = q_h[kk];
                    let row = &s[kk * head_dim..(kk + 1) * head_dim];
                    for vv in 0..head_dim {
                        head_out[vv] += row[vv] * qv;
                    }
                }
            });
    }

    // Per-head RMSNormGated with z as gate (`Qwen3_5RMSNormGated`):
    //   out_h = (rmsnorm_no_shift(out_h) * weight) * silu(z_h)
    let mut normed_out = vec![0.0_f32; value_dim];
    for h in 0..n_heads {
        let off = h * head_dim;
        ops::rmsnorm_gated(
            &core_attn_out[off..off + head_dim],
            &z[off..off + head_dim],
            &weights.ssm_norm,
            config.rms_norm_eps,
            &mut normed_out[off..off + head_dim],
        );
    }

    // Project back to hidden_size.
    ops::matvec(&normed_out, &weights.ssm_out, value_dim, h_dim, out);
}

// ---- End-to-end model state + forward -------------------------------------

/// Per-prompt state for a Qwen 3.5 forward pass. Holds one slot per block:
/// `kv_caches[i]` is `Some` iff block `i` is full-attention; `gdn_states[i]`
/// is `Some` iff it's linear-attention. The shared `rope` table and `pos`
/// counter are advanced once per token.
pub struct Qwen35F32State {
    pub kv_caches: Vec<Option<LayerKvCache>>,
    pub gdn_states: Vec<Option<LinAttnState>>,
    pub rope: RopeCache,
    pub pos: usize,
}

impl Qwen35F32State {
    pub fn new(config: &Qwen35Config, block_kinds: &[BlockKind], max_seq: usize) -> Self {
        let conv_dim = config.gdn_qkv_concat_dim() as usize;
        let mut kv_caches = Vec::with_capacity(block_kinds.len());
        let mut gdn_states = Vec::with_capacity(block_kinds.len());
        for &kind in block_kinds {
            match kind {
                BlockKind::FullAttention => {
                    kv_caches.push(Some(LayerKvCache::new(
                        max_seq,
                        config.attn_n_kv_heads as usize,
                        config.attn_head_dim as usize,
                    )));
                    gdn_states.push(None);
                }
                BlockKind::LinearAttention => {
                    kv_caches.push(None);
                    gdn_states.push(Some(LinAttnState::new(
                        config.gdn_n_heads as usize,
                        config.gdn_head_dim as usize,
                        config.gdn_head_dim as usize,
                        conv_dim,
                        config.gdn_conv_kernel as usize,
                    )));
                }
            }
        }
        let rope = RopeCache::new(
            config.rope_dim_count as usize,
            max_seq,
            config.rope_freq_base,
        );
        Self { kv_caches, gdn_states, rope, pos: 0 }
    }

    pub fn reset(&mut self) {
        for c in self.kv_caches.iter_mut().flatten() { c.reset(); }
        for s in self.gdn_states.iter_mut().flatten() { s.reset(); }
        self.pos = 0;
    }
}

/// Loaded Qwen 3.5 model: typed config + schedule + dequantized weights.
pub struct Qwen35F32Model {
    pub model: Qwen35Model,
    pub weights: Qwen35F32Weights,
}

impl Qwen35F32Model {
    pub fn load(gguf: &GgufFile) -> Result<Self, LoadError> {
        let model = Qwen35Model::load(gguf)?;
        let weights = Qwen35F32Weights::load(gguf, &model)?;
        Ok(Self { model, weights })
    }

    pub fn new_state(&self, max_seq: usize) -> Qwen35F32State {
        Qwen35F32State::new(&self.model.config, &self.model.block_kinds, max_seq)
    }

    /// Run a multi-token prompt through the model, advancing `state.pos` and
    /// accumulating KV / GDN state per token. Returns logits for the LAST
    /// token only (next-token prediction).
    ///
    /// Currently implemented as a tight loop of `forward_token`. A future
    /// optimization would batch the per-block matvecs across all N tokens
    /// (one matmul instead of N matvecs), but the per-token state update
    /// (rank-1 GDN, KV append) is inherently sequential, so the speedup
    /// is bounded by how much of each block is matvec vs state work.
    pub fn forward_tokens(&self, tokens: &[u32], state: &mut Qwen35F32State) -> Vec<f32> {
        assert!(!tokens.is_empty(), "forward_tokens needs at least one token");
        let mut last = Vec::new();
        for &t in tokens {
            last = self.forward_token(t, state);
        }
        last
    }

    /// Forward one token at the current `state.pos`. Advances `state.pos` by 1.
    /// Returns the next-token logit vector of length `vocab_size`.
    ///
    /// Embedding lookup: `token_embd` has ggml shape `[hidden, vocab]`. With
    /// the leftmost-fast convention, the row for token `v` is contiguous at
    /// flat offset `v * hidden` for `hidden` floats — the natural embedding
    /// row layout. The same tensor doubles as the output projection when
    /// `tied_embeddings == true`.
    pub fn forward_token(&self, token_id: u32, state: &mut Qwen35F32State) -> Vec<f32> {
        self.forward_token_traced(token_id, state, None)
    }

    /// Same as `forward_token` but optionally records per-stage timings into
    /// `trace`. Pass `None` for the no-overhead path.
    pub fn forward_token_traced(
        &self, token_id: u32, state: &mut Qwen35F32State, mut trace: Option<&mut ForwardTrace>,
    ) -> Vec<f32> {
        let cfg = &self.model.config;
        let h_dim = cfg.hidden_size as usize;
        let v_dim = cfg.vocab_size as usize;

        assert!((token_id as usize) < v_dim,
            "token_id {} out of range [0, {})", token_id, v_dim);

        // Embedding lookup.
        let t0 = std::time::Instant::now();
        let row_off = token_id as usize * h_dim;
        let mut hidden = self.weights.token_embd[row_off..row_off + h_dim].to_vec();
        if let Some(t) = trace.as_deref_mut() { t.embed_ns = t0.elapsed().as_nanos() as u64; }

        // 24 blocks dispatched by kind.
        if let Some(t) = trace.as_deref_mut() {
            t.block_ns.resize(self.model.block_kinds.len(), 0);
        }
        for (i, &kind) in self.model.block_kinds.iter().enumerate() {
            let tb = std::time::Instant::now();
            match (kind, &self.weights.blocks[i]) {
                (BlockKind::FullAttention, BlockWeights::FullAttention(w)) => {
                    let kv = state.kv_caches[i].as_mut()
                        .expect("kv cache slot Some for full-attention block");
                    full_attention_block(&mut hidden, w, cfg, kv, &state.rope, state.pos);
                }
                (BlockKind::LinearAttention, BlockWeights::LinearAttention(w)) => {
                    let s = state.gdn_states[i].as_mut()
                        .expect("gdn state slot Some for linear-attention block");
                    linear_attention_block(&mut hidden, w, cfg, s);
                }
                _ => unreachable!("block kind mismatch — Qwen35Model::load validates this"),
            }
            if let Some(t) = trace.as_deref_mut() { t.block_ns[i] = tb.elapsed().as_nanos() as u64; }
        }

        // Final RMSNorm + tied (or untied) output projection.
        let tn = std::time::Instant::now();
        let mut normed = vec![0.0_f32; h_dim];
        ops::rmsnorm(&hidden, &self.weights.output_norm, cfg.rms_norm_eps, &mut normed);
        if let Some(t) = trace.as_deref_mut() { t.output_norm_ns = tn.elapsed().as_nanos() as u64; }

        let tp = std::time::Instant::now();
        let proj: &[f32] = self.weights.output.as_deref()
            .unwrap_or(self.weights.token_embd.as_slice());
        let mut logits = vec![0.0_f32; v_dim];
        ops::matvec(&normed, proj, h_dim, v_dim, &mut logits);
        if let Some(t) = trace.as_deref_mut() { t.output_proj_ns = tp.elapsed().as_nanos() as u64; }

        state.pos += 1;
        logits
    }
}

/// Per-stage timing record populated by `forward_token_traced`.
#[derive(Debug, Default, Clone)]
pub struct ForwardTrace {
    pub embed_ns: u64,
    pub block_ns: Vec<u64>,
    pub output_norm_ns: u64,
    pub output_proj_ns: u64,
}

impl ForwardTrace {
    pub fn total_ns(&self) -> u64 {
        self.embed_ns
            + self.block_ns.iter().sum::<u64>()
            + self.output_norm_ns
            + self.output_proj_ns
    }
}

/// Linear-attention (GDN) block: norm → GDN step → residual → norm → SwiGLU → residual.
pub fn linear_attention_block(
    hidden_inout: &mut [f32],
    weights: &LinAttnWeights,
    config: &Qwen35Config,
    state: &mut LinAttnState,
) {
    let h = config.hidden_size as usize;
    let f = config.ffn_size as usize;
    assert_eq!(hidden_inout.len(), h);

    let mut attn_out = vec![0.0_f32; h];
    linear_attention_step(hidden_inout, weights, config, state, &mut attn_out);
    ops::add_(hidden_inout, &attn_out);

    let mut normed = vec![0.0_f32; h];
    ops::rmsnorm(hidden_inout, &weights.post_attention_norm, config.rms_norm_eps, &mut normed);
    let mut ffn_out = vec![0.0_f32; h];
    swiglu_ffn(&normed, &weights.ffn_gate, &weights.ffn_up, &weights.ffn_down, h, f, &mut ffn_out);
    ops::add_(hidden_inout, &ffn_out);
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
