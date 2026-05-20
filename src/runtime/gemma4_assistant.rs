//! Gemma 4 MTP drafter — the small 4-layer "assistant" model that
//! drafts K tokens for the larger target to verify in one pass.
//!
//! Architecture (verified against HF transformers @ 948990bd):
//!
//!   inputs at the start of a round (from the target):
//!     h_prev   : last-layer hidden state at the just-validated position,
//!                pre-`output_norm`. Width = n_embd_backbone (= target
//!                hidden_size; 5376 on the 31B).
//!     prev_tok : the token id the target just produced.
//!     KV(SWA)  : shared K/V from the target's last SWA-owning layer.
//!     KV(Full) : shared K/V from the target's last full-owning layer.
//!     pos      : the validated position. Constant for the whole round.
//!
//!   per drafter step (k = 0..K):
//!     e = target.embed_tokens[prev_tok]               # [5376], no √h scale
//!     x = concat(e, h_prev)                           # [10752]  (e first, h second)
//!     x = pre_projection(x)                           # [1024]
//!     for each of 4 blocks:
//!         standard gemma4 block, but Q-ONLY: no attn_k / attn_v / attn_k_norm,
//!         attention reads K=V from the target's KV view for this layer's kind.
//!     x = output_norm(x)                              # [1024]
//!     logits = token_embd @ x                         # [vocab]
//!     prev_tok = argmax(logits)
//!     h_prev   = post_projection(x)                   # [5376], for next step
//!
//! The drafter has no KV cache of its own (`shared_kv_layers == block_count`),
//! which is why a rejection just truncates the target's cache — there is
//! nothing to truncate on the drafter side.
//!
//! Reuses the target's launch helpers (rmsnorm / matvec / rope / attn_q8 /
//! geglu / scale / add) so kernels are loaded exactly once. The drafter
//! owns one new tiny kernel — `concat2_f32` — for the pre_projection
//! input combiner.

use std::ffi::c_void;

use crate::gguf::{GgufFile, GgmlType};
use crate::hip::{DeviceBuf, Module};
use crate::model::gemma4::AttnKind;
use crate::model::gemma4_assistant::{Gemma4AssistantConfig, Gemma4AssistantModel};
use crate::runtime::KernelCache;
use crate::runtime::gemma4::{GpuGemma4, Gemma4GpuState};
use crate::runtime::qwen35::GpuMatvecTensor;

const CONCAT2_SRC: &str = include_str!("../../kernels/concat2_f32.cpp");

/// Per-layer drafter weights — gemma4 block layout minus K/V (those
/// come from the target's KV cache).
struct DrafterBlock {
    attn_norm:      DeviceBuf<f32>,
    attn_q:         GpuMatvecTensor,
    attn_q_norm:    DeviceBuf<f32>,
    attn_output:    GpuMatvecTensor,
    post_attn_norm: DeviceBuf<f32>,
    ffn_norm:       DeviceBuf<f32>,
    ffn_gate:       GpuMatvecTensor,
    ffn_up:         GpuMatvecTensor,
    ffn_down:       GpuMatvecTensor,
    post_ffw_norm:  DeviceBuf<f32>,
    layer_output_scale: f32,
    kind:     AttnKind,
    head_dim: usize,
    n_kv:     usize,
}

impl DrafterBlock {
    fn from_gguf(gguf: &GgufFile, layer: u32,
                 kind: AttnKind, head_dim: usize, n_kv: usize) -> Result<Self, String> {
        let p = format!("blk.{layer}.");
        // Read layer_output_scale (one F32 element on host).
        let los_info = gguf.tensor(&format!("{p}layer_output_scale.weight"))
            .ok_or_else(|| format!("{p}layer_output_scale.weight missing"))?;
        let los_bytes = gguf.tensor_data(&format!("{p}layer_output_scale.weight"))
            .map_err(|e| e.to_string())?.ok_or("los no data")?;
        if los_info.ggml_type != GgmlType::F32 {
            return Err("layer_output_scale not F32".into());
        }
        let layer_output_scale = bytemuck::cast_slice::<u8, f32>(los_bytes)[0];

        Ok(Self {
            attn_norm:      load_fp32(gguf, &format!("{p}attn_norm.weight"))?,
            attn_q:         GpuMatvecTensor::from_gguf_matvec(gguf, &format!("{p}attn_q.weight"))?,
            attn_q_norm:    load_fp32(gguf, &format!("{p}attn_q_norm.weight"))?,
            attn_output:    GpuMatvecTensor::from_gguf_matvec(gguf, &format!("{p}attn_output.weight"))?,
            post_attn_norm: load_fp32(gguf, &format!("{p}post_attention_norm.weight"))?,
            ffn_norm:       load_fp32(gguf, &format!("{p}ffn_norm.weight"))?,
            ffn_gate:       GpuMatvecTensor::from_gguf_matvec(gguf, &format!("{p}ffn_gate.weight"))?,
            ffn_up:         GpuMatvecTensor::from_gguf_matvec(gguf, &format!("{p}ffn_up.weight"))?,
            ffn_down:       GpuMatvecTensor::from_gguf_matvec(gguf, &format!("{p}ffn_down.weight"))?,
            post_ffw_norm:  load_fp32(gguf, &format!("{p}post_ffw_norm.weight"))?,
            layer_output_scale, kind, head_dim, n_kv,
        })
    }
}

pub struct GpuGemma4Assistant {
    config: Gemma4AssistantConfig,

    // Top-level weights.
    token_embd:      GpuMatvecTensor,   // [hidden_d=1024, vocab]  output head
    output_norm:     DeviceBuf<f32>,    // [hidden_d]
    pre_projection:  GpuMatvecTensor,   // [2 * n_embd_backbone, hidden_d]
    post_projection: GpuMatvecTensor,   // [hidden_d, n_embd_backbone]

    blocks: Vec<DrafterBlock>,

    // Working buffers (1 per round; reused across drafter steps).
    embed_buf:   DeviceBuf<f32>,   // [n_embd_backbone]
    h_prev:      DeviceBuf<f32>,   // [n_embd_backbone]
    concat_buf:  DeviceBuf<f32>,   // [2 * n_embd_backbone]
    block_a:     DeviceBuf<f32>,   // [hidden_d] residual stream
    block_b:     DeviceBuf<f32>,   // [hidden_d] scratch
    normed:      DeviceBuf<f32>,   // [hidden_d]
    q_buf:       DeviceBuf<f32>,   // [n_heads * max(head_dim)]
    attn_out:    DeviceBuf<f32>,   // [n_heads * max(head_dim)]
    ffn_a:       DeviceBuf<f32>,   // [ffn]
    ffn_b:       DeviceBuf<f32>,   // [ffn]
    post_norm:   DeviceBuf<f32>,   // [hidden_d] after output_norm
    logits:      DeviceBuf<f32>,   // [vocab]

    // The one new kernel the drafter owns.
    m_concat: Module,
}

impl GpuGemma4Assistant {
    pub fn new(model: &Gemma4AssistantModel, gguf: &GgufFile,
               target: &GpuGemma4, cache: &KernelCache) -> Result<Self, String>
    {
        let cfg = &model.config;
        // Shape compatibility with the target.
        if cfg.n_embd_backbone as usize != target.hidden_size() {
            return Err(format!(
                "drafter n_embd_backbone ({}) != target hidden_size ({}) — \
                 wrong drafter for this target",
                cfg.n_embd_backbone, target.hidden_size()));
        }
        if cfg.requires_target_arch != "gemma4" {
            return Err(format!("drafter requires_target_arch = {:?}, expected \"gemma4\"",
                               cfg.requires_target_arch));
        }

        let hidden_d = cfg.hidden_size as usize;
        let backbone = cfg.n_embd_backbone as usize;
        let ffn = cfg.ffn_size as usize;
        let n_heads = cfg.n_heads as usize;
        let max_head_dim = cfg.head_dim_full.max(cfg.head_dim_swa) as usize;
        let qkv_max = n_heads * max_head_dim;
        let vocab = cfg.vocab_size as usize;

        let mut blocks = Vec::with_capacity(cfg.block_count as usize);
        for li in 0..cfg.block_count as usize {
            blocks.push(DrafterBlock::from_gguf(
                gguf, li as u32,
                cfg.attn_kinds[li],
                cfg.head_dim(li) as usize,
                cfg.kv_heads[li] as usize,
            )?);
        }

        let hsaco = cache.compile("concat2_f32", CONCAT2_SRC)?;
        let m_concat = Module::load(&hsaco)?;

        Ok(Self {
            token_embd:      GpuMatvecTensor::from_gguf_matvec(gguf, "token_embd.weight")?,
            output_norm:     load_fp32(gguf, "output_norm.weight")?,
            pre_projection:  GpuMatvecTensor::from_gguf_matvec(gguf, "mtp.pre_projection.weight")?,
            post_projection: GpuMatvecTensor::from_gguf_matvec(gguf, "mtp.post_projection.weight")?,

            blocks,

            embed_buf:  DeviceBuf::new(backbone)?,
            h_prev:     DeviceBuf::new(backbone)?,
            concat_buf: DeviceBuf::new(2 * backbone)?,
            block_a:    DeviceBuf::new(hidden_d)?,
            block_b:    DeviceBuf::new(hidden_d)?,
            normed:     DeviceBuf::new(hidden_d)?,
            q_buf:      DeviceBuf::new(qkv_max)?,
            attn_out:   DeviceBuf::new(qkv_max)?,
            ffn_a:      DeviceBuf::new(ffn)?,
            ffn_b:      DeviceBuf::new(ffn)?,
            post_norm:  DeviceBuf::new(hidden_d)?,
            logits:     DeviceBuf::new(vocab)?,

            m_concat,
            config: cfg.clone(),
        })
    }

    pub fn config(&self) -> &Gemma4AssistantConfig { &self.config }

    /// Seed `h_prev` from the target's last hidden state (k=0 of a
    /// round). For k≥1 the previous step's `post_projection` has
    /// already populated `h_prev` in place — no call needed.
    pub fn set_h_prev_from_target(&self, target: &GpuGemma4) -> Result<(), String> {
        let src = target.last_hidden_state();
        let n = self.config.n_embd_backbone as usize;
        if src.len() < n {
            return Err(format!("set_h_prev_from_target: target hidden buf has {} \
                               elems, drafter needs {n}", src.len()));
        }
        self.h_prev.copy_range_from_device(src, 0, 0, n)
    }

    fn launch_concat2(&self, target: &GpuGemma4, a: *mut c_void, b: *mut c_void,
                      out: *mut c_void, n: u32) -> Result<(), String> {
        let f = self.m_concat.function("concat2_f32")?;
        let block: u32 = 256;
        let grid = (2 * n + block - 1) / block;
        let mut aa = a; let mut bb = b; let mut oo = out; let mut nn = n;
        let mut args: [*mut c_void; 4] = [
            &mut aa as *mut _ as *mut c_void, &mut bb as *mut _ as *mut c_void,
            &mut oo as *mut _ as *mut c_void, &mut nn as *mut _ as *mut c_void];
        unsafe {
            f.launch((grid, 1, 1), (block, 1, 1), 0,
                     Some(target.stream()), &mut args)
        }
    }

    /// One drafter step. Reads `self.h_prev`, consumes `prev_tok`,
    /// attends the target's KV cache at `pos` (constant across a round),
    /// returns the host-readable vocab logits and updates `self.h_prev`
    /// in place with the post_projection output for the next step.
    pub fn forward_step(&self, target: &GpuGemma4, target_state: &Gemma4GpuState,
                        prev_tok: u32, pos: usize) -> Result<Vec<f32>, String>
    {
        let cfg = &self.config;
        let hidden_d = cfg.hidden_size as u32;
        let backbone = cfg.n_embd_backbone as u32;

        // Pin the device position so RoPE + attention see the constant
        // round position. (Shared-KV layers don't increment per step.)
        target.set_d_pos(pos)?;

        // 1) target_embed(prev_tok) → embed_buf [5376], scaled by √backbone.
        // The target model applies the same √hidden scale to its input
        // embeddings inside its own forward pass (standard Gemma scaling).
        // ik_llama.cpp's build_gemma4_mtp does this scale at the drafter's
        // pre_projection input — without it, the embed half of the concat
        // sits at norm ~1 while the trained drafter expects ~√backbone.
        target.embed_token_raw(prev_tok, self.embed_buf.raw_ptr())?;
        target.launch_scale(self.embed_buf.raw_ptr(), backbone, (backbone as f32).sqrt())?;

        // 2) concat(embed_buf, h_prev) → concat_buf [10752]
        self.launch_concat2(target, self.embed_buf.raw_ptr(), self.h_prev.raw_ptr(),
                            self.concat_buf.raw_ptr(), backbone)?;

        // 3) block_a = pre_projection @ concat_buf  [10752 → 1024]
        target.launch_matvec(&self.pre_projection,
                             self.concat_buf.raw_ptr(), self.block_a.raw_ptr())?;

        // 4) 4 drafter blocks — Q-only attention against the target's
        //    shared KV for this block's attention kind.
        for b in &self.blocks {
            let head_dim = b.head_dim as u32;
            let n_kv = b.n_kv as u32;

            // --- attention ---
            target.launch_rmsnorm(self.block_a.raw_ptr(), b.attn_norm.raw_ptr(),
                                   self.normed.raw_ptr(), hidden_d)?;
            target.launch_matvec(&b.attn_q,
                                 self.normed.raw_ptr(), self.q_buf.raw_ptr())?;
            target.launch_rmsnorm_mh(self.q_buf.raw_ptr(), b.attn_q_norm.raw_ptr(),
                                     self.q_buf.raw_ptr(),
                                     target.n_heads() as u32, head_dim)?;
            target.launch_rope(self.q_buf.raw_ptr(),
                               target.n_heads() as u32, head_dim, b.kind)?;

            // Shared KV: the target's last-of-this-kind KV-owning layer.
            let donor_layer = target.last_kv_owning_layer(b.kind)
                .ok_or_else(|| format!("drafter: target has no KV-owning layer of kind {:?}",
                                       b.kind))?;
            let kv = target_state.layer_kv_view(donor_layer);
            if kv.head_dim as u32 != head_dim || kv.n_kv as u32 != n_kv {
                return Err(format!(
                    "drafter block KV shape mismatch (kind {:?}): drafter expects \
                     n_kv={}, head_dim={}; target donor layer {} has n_kv={}, head_dim={}",
                    b.kind, n_kv, head_dim, donor_layer, kv.n_kv, kv.head_dim));
            }
            let window: u32 = match b.kind {
                AttnKind::Sliding => cfg.sliding_window,
                AttnKind::Full    => 0,
            };
            target.launch_attn_q8(self.q_buf.raw_ptr(),
                                   kv.k.raw_ptr(), kv.ks.raw_ptr(),
                                   kv.v.raw_ptr(), kv.vs.raw_ptr(),
                                   self.attn_out.raw_ptr(),
                                   n_kv, head_dim, window)?;

            target.launch_matvec(&b.attn_output,
                                 self.attn_out.raw_ptr(), self.block_b.raw_ptr())?;
            target.launch_rmsnorm(self.block_b.raw_ptr(), b.post_attn_norm.raw_ptr(),
                                   self.normed.raw_ptr(), hidden_d)?;
            target.launch_add(self.block_a.raw_ptr(),
                              self.normed.raw_ptr(), hidden_d)?;

            // --- FFN ---
            target.launch_rmsnorm(self.block_a.raw_ptr(), b.ffn_norm.raw_ptr(),
                                   self.normed.raw_ptr(), hidden_d)?;
            target.launch_matvec(&b.ffn_gate,
                                 self.normed.raw_ptr(), self.ffn_a.raw_ptr())?;
            target.launch_matvec(&b.ffn_up,
                                 self.normed.raw_ptr(), self.ffn_b.raw_ptr())?;
            target.launch_geglu(self.ffn_a.raw_ptr(), self.ffn_b.raw_ptr(),
                                self.ffn_a.raw_ptr(), cfg.ffn_size)?;
            target.launch_matvec(&b.ffn_down,
                                 self.ffn_a.raw_ptr(), self.block_b.raw_ptr())?;
            target.launch_rmsnorm(self.block_b.raw_ptr(), b.post_ffw_norm.raw_ptr(),
                                   self.normed.raw_ptr(), hidden_d)?;
            target.launch_add(self.block_a.raw_ptr(),
                              self.normed.raw_ptr(), hidden_d)?;
            target.launch_scale(self.block_a.raw_ptr(),
                                hidden_d, b.layer_output_scale)?;
        }

        // 5) post-block output norm.
        target.launch_rmsnorm(self.block_a.raw_ptr(), self.output_norm.raw_ptr(),
                               self.post_norm.raw_ptr(), hidden_d)?;

        // 6) tied vocab head.
        target.launch_matvec(&self.token_embd,
                             self.post_norm.raw_ptr(), self.logits.raw_ptr())?;

        // 7) post_projection: drafter hidden (post-norm) → backbone-dim
        //    h_prev for the NEXT step. Overwrites self.h_prev in place.
        target.launch_matvec(&self.post_projection,
                             self.post_norm.raw_ptr(), self.h_prev.raw_ptr())?;

        target.stream().synchronize()?;
        let mut out = vec![0.0f32; cfg.vocab_size as usize];
        self.logits.copy_to_host(&mut out)?;
        Ok(out)
    }
}

// ---- helpers ----

fn load_fp32(gguf: &GgufFile, name: &str) -> Result<DeviceBuf<f32>, String> {
    let info = gguf.tensor(name).ok_or_else(|| format!("tensor {name} not found"))?;
    if info.ggml_type != GgmlType::F32 {
        return Err(format!("tensor {name}: expected F32, got {:?}", info.ggml_type));
    }
    let bytes = gguf.tensor_data(name).map_err(|e| e.to_string())?
        .ok_or_else(|| format!("tensor {name} has no data"))?;
    let floats: &[f32] = bytemuck::cast_slice(bytes);
    DeviceBuf::from_slice(floats)
}

