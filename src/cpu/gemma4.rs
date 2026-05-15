//! Gemma 4 CPU forward-pass oracle.
//!
//! Reference implementation matching llama.cpp's `models/gemma4.cpp`
//! (dense path — the 31B has no MoE and no per-layer input embeddings).
//!
//! Per-token forward:
//!   x = embed(token) · √hidden
//!   for each layer:
//!     n   = rmsnorm(x, attn_norm)
//!     q   = proj(n, attn_q);  per-head: q = rmsnorm(q, attn_q_norm); rope(q)
//!     k   = proj(n, attn_k);  v = attn_v ? proj(n, attn_v) : k
//!     per-head: k = rmsnorm(k, attn_k_norm); rope(k);  v = rmsnorm_plain(v)
//!     a   = attention(q, K_cache, V_cache)         scale 1.0, SWA or full
//!     a   = proj(a, attn_output)
//!     a   = rmsnorm(a, post_attention_norm)
//!     x   = x + a
//!     n2  = rmsnorm(x, ffn_norm)
//!     f   = proj(gelu(proj(n2, ffn_gate)) * proj(n2, ffn_up), ffn_down)
//!     f   = rmsnorm(f, post_ffw_norm)
//!     x   = x + f
//!     x   = x · layer_output_scale
//!   x = rmsnorm(x, output_norm)
//!   logits = proj(x, token_embd)            tied embeddings
//!   logits = cap · tanh(logits / cap)       final-logit soft-cap
//!
//! Weights are dequantised lazily per layer — the 31B's fp32 footprint
//! (~124 GB) far exceeds host RAM, so each layer's weights are
//! materialised on demand and dropped after use (~2.5 GB peak).

use crate::cpu::ops;
use crate::cpu::rope::{RopeCache, apply_rope};
use crate::gguf::GgufFile;
use crate::model::gemma4::{AttnKind, Gemma4Model};
use crate::quant;

/// Loaded Gemma 4 model for CPU inference. Holds the mmap'd GGUF and
/// dequantises weights on demand.
pub struct Gemma4CpuModel {
    pub model: Gemma4Model,
    gguf: GgufFile,
    /// token_embd dequantised once (also the tied output projection).
    token_embd: Vec<f32>,
    output_norm: Vec<f32>,
}

/// Dequantised weights for one transformer layer. `ffn_*` is the shared
/// MLP — present on every layer; `moe` is set only on MoE layers.
struct LayerWeights {
    attn_norm:    Vec<f32>,
    attn_q:       Vec<f32>,
    attn_k:       Vec<f32>,
    /// `None` on full-attention layers — V reuses the K projection.
    attn_v:       Option<Vec<f32>>,
    attn_q_norm:  Vec<f32>,
    attn_k_norm:  Vec<f32>,
    attn_output:  Vec<f32>,
    post_attention_norm: Vec<f32>,
    ffn_norm:     Vec<f32>,
    ffn_gate:     Vec<f32>,
    ffn_up:       Vec<f32>,
    ffn_down:     Vec<f32>,
    post_ffw_norm: Vec<f32>,
    layer_output_scale: f32,
    moe:          Option<MoeWeights>,
}

/// MoE-layer-only weights. The expert matrices themselves are NOT held
/// here — they are dequantised on demand (only the 8 routed experts per
/// token), since all 128 at fp32 would be ~3 GB per layer.
struct MoeWeights {
    /// Post-norm for the shared MLP branch.
    post_ffw_norm_1: Vec<f32>,
    /// Pre-norm for the routed-expert branch.
    pre_ffw_norm_2:  Vec<f32>,
    /// Post-norm for the routed-expert branch.
    post_ffw_norm_2: Vec<f32>,
    /// Router projection, F32 [hidden, n_expert].
    gate_inp:    Vec<f32>,
    /// Router input scale, F32 [hidden].
    gate_inp_s:  Vec<f32>,
    /// Per-expert scalar on the down-projection output, F32 [n_expert].
    down_exps_s: Vec<f32>,
}

/// Per-layer KV cache. Each `push` appends `kv_heads · head_dim` floats.
pub struct Gemma4State {
    k_cache: Vec<Vec<f32>>,
    v_cache: Vec<Vec<f32>>,
    pub pos: usize,
}

impl Gemma4State {
    pub fn new(n_layers: usize) -> Self {
        Self {
            k_cache: vec![Vec::new(); n_layers],
            v_cache: vec![Vec::new(); n_layers],
            pos: 0,
        }
    }
    pub fn reset(&mut self) {
        for c in &mut self.k_cache { c.clear(); }
        for c in &mut self.v_cache { c.clear(); }
        self.pos = 0;
    }
}

impl Gemma4CpuModel {
    pub fn load(gguf: GgufFile) -> Result<Self, String> {
        let model = Gemma4Model::load(&gguf).map_err(|e| e.to_string())?;
        let token_embd  = dequant(&gguf, "token_embd.weight")?;
        let output_norm = dequant(&gguf, "output_norm.weight")?;
        Ok(Self { model, gguf, token_embd, output_norm })
    }

    pub fn new_state(&self) -> Gemma4State {
        Gemma4State::new(self.model.config.block_count as usize)
    }

    fn layer_weights(&self, layer: u32, kind: AttnKind) -> Result<LayerWeights, String> {
        let p = format!("blk.{layer}.");
        let attn_v = if kind == AttnKind::Sliding {
            Some(dequant(&self.gguf, &format!("{p}attn_v.weight"))?)
        } else {
            None
        };
        let los = dequant(&self.gguf, &format!("{p}layer_output_scale.weight"))?;
        let moe = if self.model.config.is_moe() {
            Some(MoeWeights {
                post_ffw_norm_1: dequant(&self.gguf, &format!("{p}post_ffw_norm_1.weight"))?,
                pre_ffw_norm_2:  dequant(&self.gguf, &format!("{p}pre_ffw_norm_2.weight"))?,
                post_ffw_norm_2: dequant(&self.gguf, &format!("{p}post_ffw_norm_2.weight"))?,
                gate_inp:    dequant(&self.gguf, &format!("{p}ffn_gate_inp.weight"))?,
                gate_inp_s:  dequant(&self.gguf, &format!("{p}ffn_gate_inp.scale"))?,
                down_exps_s: dequant(&self.gguf, &format!("{p}ffn_down_exps.scale"))?,
            })
        } else {
            None
        };
        Ok(LayerWeights {
            attn_norm:    dequant(&self.gguf, &format!("{p}attn_norm.weight"))?,
            attn_q:       dequant(&self.gguf, &format!("{p}attn_q.weight"))?,
            attn_k:       dequant(&self.gguf, &format!("{p}attn_k.weight"))?,
            attn_v,
            attn_q_norm:  dequant(&self.gguf, &format!("{p}attn_q_norm.weight"))?,
            attn_k_norm:  dequant(&self.gguf, &format!("{p}attn_k_norm.weight"))?,
            attn_output:  dequant(&self.gguf, &format!("{p}attn_output.weight"))?,
            post_attention_norm: dequant(&self.gguf, &format!("{p}post_attention_norm.weight"))?,
            ffn_norm:     dequant(&self.gguf, &format!("{p}ffn_norm.weight"))?,
            ffn_gate:     dequant(&self.gguf, &format!("{p}ffn_gate.weight"))?,
            ffn_up:       dequant(&self.gguf, &format!("{p}ffn_up.weight"))?,
            ffn_down:     dequant(&self.gguf, &format!("{p}ffn_down.weight"))?,
            post_ffw_norm: dequant(&self.gguf, &format!("{p}post_ffw_norm.weight"))?,
            layer_output_scale: los[0],
            moe,
        })
    }

    /// One decode step. Returns vocab-length logits (soft-capped).
    pub fn forward_token(&self, token: u32, state: &mut Gemma4State) -> Result<Vec<f32>, String> {
        let cfg = &self.model.config;
        let h     = cfg.hidden_size as usize;
        let vocab = cfg.vocab_size as usize;
        let eps   = cfg.rms_norm_eps;
        let pos   = state.pos;

        // Embedding, scaled by √hidden (Gemma convention).
        let mut x = vec![0.0f32; h];
        let off = token as usize * h;
        let emb_scale = (h as f32).sqrt();
        for i in 0..h { x[i] = self.token_embd[off + i] * emb_scale; }

        for layer in 0..cfg.block_count {
            let kind = cfg.attn_kinds[layer as usize];
            let w = self.layer_weights(layer, kind)?;
            self.layer_forward(&mut x, &w, layer as usize, kind, state)?;
        }

        // Final norm + tied output projection.
        let mut normed = vec![0.0f32; h];
        ops::rmsnorm(&x, &self.output_norm, eps, &mut normed);
        let mut logits = vec![0.0f32; vocab];
        ops::matvec(&normed, &self.token_embd, h, vocab, &mut logits);

        // Final-logit soft-cap: cap · tanh(logits / cap).
        let cap = cfg.final_logit_softcapping;
        if cap > 0.0 {
            for v in logits.iter_mut() { *v = cap * (*v / cap).tanh(); }
        }

        state.pos = pos + 1;
        Ok(logits)
    }

    fn layer_forward(&self, x: &mut [f32], w: &LayerWeights, layer: usize,
                     kind: AttnKind, state: &mut Gemma4State) -> Result<(), String>
    {
        let cfg = &self.model.config;
        let h        = cfg.hidden_size as usize;
        let eps      = cfg.rms_norm_eps;
        let n_heads  = cfg.n_heads as usize;
        let head_dim = cfg.head_dim(layer) as usize;
        let n_kv     = cfg.kv_heads[layer] as usize;
        let q_dim    = n_heads * head_dim;
        let kv_dim   = n_kv * head_dim;
        let rotary   = match kind {
            AttnKind::Sliding => cfg.rope_dim_swa,
            AttnKind::Full    => cfg.rope_dim_full,
        } as usize;
        let rope = RopeCache::new(rotary, cfg.context_length.min(8192) as usize + 8,
                                  cfg.rope_base(layer));
        let pos = state.pos;

        // --- Attention sub-layer ---
        let mut normed = vec![0.0f32; h];
        ops::rmsnorm(x, &w.attn_norm, eps, &mut normed);

        // Q projection → per-head norm + RoPE.
        let mut q = vec![0.0f32; q_dim];
        ops::matvec(&normed, &w.attn_q, h, q_dim, &mut q);
        let mut tmp = vec![0.0f32; head_dim];
        for hd in 0..n_heads {
            let s = &mut q[hd * head_dim..(hd + 1) * head_dim];
            ops::rmsnorm(s, &w.attn_q_norm, eps, &mut tmp);
            s.copy_from_slice(&tmp);
            apply_rope(s, &rope, pos);
        }

        // K projection; V = attn_v projection, or the K projection itself.
        let mut k_proj = vec![0.0f32; kv_dim];
        ops::matvec(&normed, &w.attn_k, h, kv_dim, &mut k_proj);
        let mut v = match &w.attn_v {
            Some(wv) => {
                let mut vv = vec![0.0f32; kv_dim];
                ops::matvec(&normed, wv, h, kv_dim, &mut vv);
                vv
            }
            None => k_proj.clone(),  // full-attention layers: V is the K projection
        };

        // K: per-head weighted RMSNorm + RoPE.  V: per-head plain RMSNorm.
        let mut k = vec![0.0f32; kv_dim];
        for hd in 0..n_kv {
            let src = &k_proj[hd * head_dim..(hd + 1) * head_dim];
            let dst = &mut k[hd * head_dim..(hd + 1) * head_dim];
            ops::rmsnorm(src, &w.attn_k_norm, eps, dst);
            apply_rope(dst, &rope, pos);
        }
        for hd in 0..n_kv {
            let vs = &mut v[hd * head_dim..(hd + 1) * head_dim];
            rmsnorm_plain(vs, eps);
        }

        // Append (k, v) to this layer's cache.
        state.k_cache[layer].extend_from_slice(&k);
        state.v_cache[layer].extend_from_slice(&v);
        let cache_len = state.k_cache[layer].len() / kv_dim;

        // Sliding-window lower bound on attended positions.
        let lo = match kind {
            AttnKind::Sliding => {
                let win = cfg.sliding_window as usize;
                cache_len.saturating_sub(win)
            }
            AttnKind::Full => 0,
        };

        // Per-head attention. Gemma 4 uses softmax scale 1.0 (the q/k
        // norms carry the effective scaling).
        let groups = n_heads / n_kv;
        let mut attn = vec![0.0f32; q_dim];
        let mut scores = vec![0.0f32; cache_len];
        for hd in 0..n_heads {
            let kv_h = hd / groups;
            let q_h = &q[hd * head_dim..(hd + 1) * head_dim];
            let win_len = cache_len - lo;
            for t in lo..cache_len {
                let k_t = &state.k_cache[layer][(t * n_kv + kv_h) * head_dim..][..head_dim];
                let mut acc = 0.0f32;
                for d in 0..head_dim { acc += q_h[d] * k_t[d]; }
                scores[t - lo] = acc;
            }
            ops::softmax(&mut scores[..win_len]);
            let head_out = &mut attn[hd * head_dim..(hd + 1) * head_dim];
            head_out.fill(0.0);
            for t in lo..cache_len {
                let v_t = &state.v_cache[layer][(t * n_kv + kv_h) * head_dim..][..head_dim];
                let s = scores[t - lo];
                for d in 0..head_dim { head_out[d] += s * v_t[d]; }
            }
        }

        // Output projection, post-norm, residual.
        let mut attn_out = vec![0.0f32; h];
        ops::matvec(&attn, &w.attn_output, q_dim, h, &mut attn_out);
        let mut attn_normed = vec![0.0f32; h];
        ops::rmsnorm(&attn_out, &w.post_attention_norm, eps, &mut attn_normed);
        for i in 0..h { x[i] += attn_normed[i]; }

        // --- FFN sub-layer ---
        // Shared MLP (GeGLU) — runs on every layer. `x` still holds
        // attn_out; we don't mutate it until the final residual add.
        let f = cfg.ffn_size as usize;
        let mut ffn_in = vec![0.0f32; h];
        ops::rmsnorm(x, &w.ffn_norm, eps, &mut ffn_in);
        let mut gate = vec![0.0f32; f];
        let mut up   = vec![0.0f32; f];
        ops::matvec(&ffn_in, &w.ffn_gate, h, f, &mut gate);
        ops::matvec(&ffn_in, &w.ffn_up,   h, f, &mut up);
        for i in 0..f { gate[i] = ops::gelu(gate[i]) * up[i]; }
        let mut mlp = vec![0.0f32; h];
        ops::matvec(&gate, &w.ffn_down, f, h, &mut mlp);

        // Combined FFN result (pre the shared post_ffw_norm).
        let ffn_result = match &w.moe {
            None => mlp,   // dense layer — the MLP is the whole FFN
            Some(mw) => {
                // Shared-MLP branch gets its own post-norm.
                let mut cur_mlp = vec![0.0f32; h];
                ops::rmsnorm(&mlp, &mw.post_ffw_norm_1, eps, &mut cur_mlp);
                // Routed-expert branch.
                let moe = self.moe_branch(x, layer, mw)?;
                let mut cur_moe = vec![0.0f32; h];
                ops::rmsnorm(&moe, &mw.post_ffw_norm_2, eps, &mut cur_moe);
                for i in 0..h { cur_mlp[i] += cur_moe[i]; }
                cur_mlp
            }
        };

        let mut ffn_normed = vec![0.0f32; h];
        ops::rmsnorm(&ffn_result, &w.post_ffw_norm, eps, &mut ffn_normed);
        for i in 0..h { x[i] += ffn_normed[i]; }

        // Per-layer output scale.
        for v in x.iter_mut() { *v *= w.layer_output_scale; }
        Ok(())
    }

    /// Routed-expert branch of a MoE layer. `attn_out` is the post-
    /// attention residual stream; returns the expert mixture (h-dim,
    /// before `post_ffw_norm_2`). Only the 8 routed experts per token
    /// are dequantised.
    fn moe_branch(&self, attn_out: &[f32], layer: usize, mw: &MoeWeights)
        -> Result<Vec<f32>, String>
    {
        let cfg = &self.model.config;
        let h      = cfg.hidden_size as usize;
        let eps    = cfg.rms_norm_eps;
        let n_exp  = cfg.expert_count as usize;
        let n_used = cfg.expert_used_count as usize;
        let ff_exp = cfg.expert_ff_size as usize;

        // Router: plain RMSNorm of attn_out, scaled by gate_inp_s/√h,
        // then a small F32 projection to per-expert logits.
        let inv_sqrt_h = 1.0 / (h as f32).sqrt();
        let mut router_in = attn_out.to_vec();
        rmsnorm_plain(&mut router_in, eps);
        for i in 0..h { router_in[i] *= mw.gate_inp_s[i] * inv_sqrt_h; }
        let mut logits = vec![0.0f32; n_exp];
        ops::matvec(&router_in, &mw.gate_inp, h, n_exp, &mut logits);

        // softmax over all experts, then top-k selection.
        let mut probs = logits;
        ops::softmax(&mut probs);
        let mut order: Vec<usize> = (0..n_exp).collect();
        order.sort_unstable_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
        let sel = &order[..n_used];
        // Renormalise the selected weights to sum to 1 (clamped).
        let wsum: f32 = sel.iter().map(|&e| probs[e]).sum::<f32>().max(6.103515625e-5);

        // Expert FFN input: pre_ffw_norm_2 of attn_out.
        let mut moe_in = vec![0.0f32; h];
        ops::rmsnorm(attn_out, &mw.pre_ffw_norm_2, eps, &mut moe_in);

        let gate_up_name = format!("blk.{layer}.ffn_gate_up_exps.weight");
        let down_name    = format!("blk.{layer}.ffn_down_exps.weight");
        let mut moe = vec![0.0f32; h];
        for &e in sel {
            let weight = probs[e] / wsum;
            // gate_up: [h → 2·ff_exp]; gate = first half, up = second.
            let gate_up_w = dequant_expert(&self.gguf, &gate_up_name, e, h, 2 * ff_exp)?;
            let mut gu = vec![0.0f32; 2 * ff_exp];
            ops::matvec(&moe_in, &gate_up_w, h, 2 * ff_exp, &mut gu);
            let mut act = vec![0.0f32; ff_exp];
            for i in 0..ff_exp { act[i] = ops::gelu(gu[i]) * gu[ff_exp + i]; }
            // down: [ff_exp → h], scaled by the per-expert scalar.
            let down_w = dequant_expert(&self.gguf, &down_name, e, ff_exp, h)?;
            let mut down = vec![0.0f32; h];
            ops::matvec(&act, &down_w, ff_exp, h, &mut down);
            let s = weight * mw.down_exps_s[e];
            for i in 0..h { moe[i] += s * down[i]; }
        }
        Ok(moe)
    }
}

/// RMSNorm without a learnable weight: `x ← x · rsqrt(mean(x²) + eps)`.
/// Gemma applies this to V (there is no attn_v_norm tensor).
fn rmsnorm_plain(x: &mut [f32], eps: f32) {
    let n = x.len() as f32;
    let mut ss = 0.0f32;
    for &v in x.iter() { ss += v * v; }
    let rrms = (ss / n + eps).sqrt().recip();
    for v in x.iter_mut() { *v *= rrms; }
}

/// Dequantise one expert's slice of a 3D expert tensor
/// `[in_dim, out_dim, n_expert]` to fp32 `[in_dim·out_dim]`. Each expert
/// is a contiguous `[in_dim, out_dim]` quantized matrix.
fn dequant_expert(gguf: &GgufFile, name: &str, expert: usize,
                  in_dim: usize, out_dim: usize) -> Result<Vec<f32>, String> {
    use crate::gguf::GgmlType;
    let info = gguf.tensor(name).ok_or_else(|| format!("missing tensor {name}"))?;
    let bytes = gguf.tensor_data(name)
        .map_err(|e| format!("read {name}: {e}"))?
        .ok_or_else(|| format!("no data for {name}"))?;
    let (bs, bpb) = match info.ggml_type {
        GgmlType::Q6_K => (quant::q6_k::BLOCK_SIZE, quant::q6_k::BYTES_PER_BLOCK),
        GgmlType::Q8_0 => (quant::q8_0::BLOCK_SIZE, quant::q8_0::BYTES_PER_BLOCK),
        GgmlType::Q5_K => (quant::q5_k::BLOCK_SIZE, quant::q5_k::BYTES_PER_BLOCK),
        GgmlType::Q4_K => (quant::q4_k::BLOCK_SIZE, quant::q4_k::BYTES_PER_BLOCK),
        other => return Err(format!("dequant_expert {name}: unsupported type {other:?}")),
    };
    if in_dim % bs != 0 {
        return Err(format!("dequant_expert {name}: in_dim {in_dim} not a multiple of {bs}"));
    }
    let bytes_per_expert = (in_dim / bs) * bpb * out_dim;
    let start = expert * bytes_per_expert;
    let slice = bytes.get(start..start + bytes_per_expert)
        .ok_or_else(|| format!("dequant_expert {name}: expert {expert} out of range"))?;
    let mut out = vec![0.0f32; in_dim * out_dim];
    match info.ggml_type {
        GgmlType::Q6_K => quant::q6_k::dequantize_to_f32(slice, &mut out),
        GgmlType::Q8_0 => quant::q8_0::dequantize_to_f32(slice, &mut out),
        GgmlType::Q5_K => quant::q5_k::dequantize_to_f32(slice, &mut out),
        GgmlType::Q4_K => quant::q4_k::dequantize_to_f32(slice, &mut out),
        _ => unreachable!(),
    }
    Ok(out)
}

/// Dequantise a named tensor to fp32.
fn dequant(gguf: &GgufFile, name: &str) -> Result<Vec<f32>, String> {
    let info = gguf.tensor(name).ok_or_else(|| format!("missing tensor {name}"))?;
    let bytes = gguf.tensor_data(name)
        .map_err(|e| format!("read {name}: {e}"))?
        .ok_or_else(|| format!("no data for {name}"))?;
    quant::dequantize_tensor(info, bytes).map_err(|e| format!("dequant {name}: {e}"))
}
