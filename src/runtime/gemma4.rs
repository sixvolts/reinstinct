//! GPU forward path for Gemma 4 (dense variant — the 31B).
//!
//! Mirrors `cpu::gemma4` on the MI50: weights stay resident in their
//! on-disk quantized form, the forward chains HIP kernels on one
//! stream. Reuses the matvec / rmsnorm / rope / add kernels; the
//! Gemma-specific ones (geglu, logit soft-cap, scale, windowed
//! attention) were added alongside.

use std::ffi::c_void;

use crate::gguf::{GgmlType, GgufFile};
use crate::hip::{DeviceBuf, Event, Graph, GraphExec, Module, Stream};
use crate::hip::sys::HipStreamCaptureMode;
use crate::model::gemma4::{AttnKind, Gemma4Model};
use crate::runtime::KernelCache;
use crate::runtime::qwen35::GpuMatvecTensor;

// Reused kernel sources.
const RMSNORM_SRC:           &str = include_str!("../../kernels/rmsnorm.cpp");
const RMSNORM_MULTIHEAD_SRC: &str = include_str!("../../kernels/rmsnorm_multihead.cpp");
const ROPE_SRC:              &str = include_str!("../../kernels/rope_dpos.cpp");
const ADD_INPLACE_SRC:       &str = include_str!("../../kernels/add_inplace.cpp");
const MATVEC_F32_W_SRC:      &str = include_str!("../../kernels/matvec_f32_wave64.cpp");
const MATVEC_Q4K_W_SRC:      &str = include_str!("../../kernels/matvec_q4_k_rowblock.cpp");
const MATVEC_Q5K_W_SRC:      &str = include_str!("../../kernels/matvec_q5_k_rowblock.cpp");
const MATVEC_Q6K_W_SRC:      &str = include_str!("../../kernels/matvec_q6_k_rowblock.cpp");
const QUANTIZE_Q8_SRC:       &str = include_str!("../../kernels/quantize_q8.cpp");
const MATVEC_Q4K_DP4A_SRC:   &str = include_str!("../../kernels/matvec_q4_k_dp4a.cpp");
const MATVEC_Q5K_DP4A_SRC:   &str = include_str!("../../kernels/matvec_q5_k_dp4a.cpp");
const MATVEC_Q6K_DP4A_SRC:   &str = include_str!("../../kernels/matvec_q6_k_dp4a.cpp");
/// Output rows per wavefront in the row-blocked K-quant matvecs — must
/// match `ROWS` in matvec_q{4,5,6}_k_rowblock.cpp.
const Q4K_ROWBLOCK: u32 = 2;
const MATVEC_Q8_0_W_SRC:     &str = include_str!("../../kernels/matvec_q8_0_wave64.cpp");
const MATVEC_F16_W_SRC:      &str = include_str!("../../kernels/matvec_f16_wave64.cpp");
// Gemma-specific kernel sources.
const GEGLU_SRC:             &str = include_str!("../../kernels/geglu.cpp");
const LOGIT_SOFTCAP_SRC:     &str = include_str!("../../kernels/logit_softcap.cpp");
const SCALE_INPLACE_SRC:     &str = include_str!("../../kernels/scale_inplace.cpp");
const ATTN_WINDOW_SRC:       &str = include_str!("../../kernels/attn_step_q8.cpp");
const KV_WRITE_SRC:          &str = include_str!("../../kernels/kv_write_q8.cpp");
const EMBED_Q5K_SRC:         &str = include_str!("../../kernels/embed_lookup_q5_k.cpp");
const EMBED_Q6K_SRC:         &str = include_str!("../../kernels/embed_lookup_q6_k.cpp");
const EMBED_F32_SRC:         &str = include_str!("../../kernels/embed_lookup.cpp");
const EMBED_Q8_0_SRC:        &str = include_str!("../../kernels/embed_lookup_q8_0.cpp");
// MoE kernel sources.
const MATVEC_Q8_0_DP4A_SRC:  &str = include_str!("../../kernels/matvec_q8_0_dp4a.cpp");
// Prefill kernel sources.
const ROPE_PREFILL_SRC:      &str = include_str!("../../kernels/rope_prefill.cpp");
const ATTN_PREFILL_SRC:      &str = include_str!("../../kernels/attn_prefill.cpp");
const MOE_TOPK_SRC:          &str = include_str!("../../kernels/moe_topk.cpp");
const MOE_MATVEC_Q6K_SRC:    &str = include_str!("../../kernels/moe_matvec_q6k_dp4a.cpp");
const MOE_MATVEC_Q8_0_SRC:   &str = include_str!("../../kernels/moe_matvec_q8_0_dp4a.cpp");
const MOE_GEGLU_SRC:         &str = include_str!("../../kernels/moe_geglu.cpp");
const MOE_COMBINE_SRC:       &str = include_str!("../../kernels/moe_combine.cpp");

/// Offset an f32 device pointer by `elems` elements (prefill row indexing).
fn pf_off(p: *mut c_void, elems: usize) -> *mut c_void {
    unsafe { (p as *mut f32).add(elems) as *mut c_void }
}

/// Load an fp32 GGUF tensor straight to device.
fn load_fp32(gguf: &GgufFile, name: &str) -> Result<DeviceBuf<f32>, String> {
    let info = gguf.tensor(name).ok_or_else(|| format!("tensor {name} not found"))?;
    if info.ggml_type != GgmlType::F32 {
        return Err(format!("tensor {name}: expected F32, got {:?}", info.ggml_type));
    }
    let bytes = gguf.tensor_data(name).map_err(|e| format!("{name}: {e}"))?
        .ok_or_else(|| format!("{name}: no data"))?;
    let floats: &[f32] = bytemuck::cast_slice(bytes);
    DeviceBuf::from_slice(floats)
}

/// A 3D expert-weight tensor `[in_dim, out_dim, n_expert]` resident on
/// device in its on-disk quantized form. Each expert is a contiguous
/// `[in_dim, out_dim]` matrix `bytes_per_expert` apart; the moe_matvec
/// kernel offsets into the slab by the device-resident expert id.
pub struct ExpertTensor {
    data:  DeviceBuf<u8>,
    /// Quant type — Unsloth's UD recipe varies it per layer (the 26B's
    /// last-layer gate_up is Q8_0 while the rest are Q6_K), so the
    /// moe_matvec dispatch must be per-tensor, not hard-coded.
    dtype: GgmlType,
    bytes_per_expert: usize,
}

impl ExpertTensor {
    fn from_gguf(gguf: &GgufFile, name: &str) -> Result<Self, String> {
        let info = gguf.tensor(name).ok_or_else(|| format!("tensor {name} not found"))?;
        let bytes = gguf.tensor_data(name)
            .map_err(|e| format!("read {name}: {e}"))?
            .ok_or_else(|| format!("tensor {name} has no data"))?;
        let shape = info.shape();
        if shape.len() != 3 {
            return Err(format!("expert tensor {name}: expected 3D, got {shape:?}"));
        }
        let n_expert = shape[2] as usize;
        Ok(Self {
            bytes_per_expert: bytes.len() / n_expert,
            dtype: info.ggml_type,
            data: DeviceBuf::from_slice(bytes)?,
        })
    }

}

/// MoE-layer weights: the routed-expert branch that runs alongside the
/// shared MLP. Present only on MoE models (the 26B-A4B).
pub struct MoeBlock {
    post_ffw_norm_1: DeviceBuf<f32>,
    pre_ffw_norm_2:  DeviceBuf<f32>,
    post_ffw_norm_2: DeviceBuf<f32>,
    /// Router projection, F32 [hidden, n_expert].
    gate_inp:    GpuMatvecTensor,
    /// Router input scale, F32 [hidden].
    gate_inp_s:  DeviceBuf<f32>,
    /// Fused gate+up experts, [hidden, 2·expert_ff, n_expert].
    gate_up_exps: ExpertTensor,
    /// Down experts, [expert_ff, hidden, n_expert].
    down_exps:    ExpertTensor,
    /// Per-expert down-output scalar, F32 [n_expert] — device-resident
    /// so the combine kernel can index it by the device expert id.
    down_exps_s:  DeviceBuf<f32>,
}

/// All weights for one Gemma 4 transformer block on device.
pub struct GpuGemma4Block {
    attn_norm:      DeviceBuf<f32>,
    attn_q:         GpuMatvecTensor,
    attn_k:         GpuMatvecTensor,
    /// `None` on full-attention layers — V reuses the K projection.
    attn_v:         Option<GpuMatvecTensor>,
    attn_q_norm:    DeviceBuf<f32>,
    attn_k_norm:    DeviceBuf<f32>,
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
    /// `Some` on MoE layers — the routed-expert branch.
    moe:      Option<MoeBlock>,
}

impl GpuGemma4Block {
    fn from_gguf(gguf: &GgufFile, layer: u32, kind: AttnKind,
                 head_dim: usize, n_kv: usize, moe: bool) -> Result<Self, String> {
        let p = format!("blk.{layer}.");
        let moe_block = if moe {
            Some(MoeBlock {
                post_ffw_norm_1: load_fp32(gguf, &format!("{p}post_ffw_norm_1.weight"))?,
                pre_ffw_norm_2:  load_fp32(gguf, &format!("{p}pre_ffw_norm_2.weight"))?,
                post_ffw_norm_2: load_fp32(gguf, &format!("{p}post_ffw_norm_2.weight"))?,
                gate_inp:    GpuMatvecTensor::from_gguf(gguf, &format!("{p}ffn_gate_inp.weight"))?,
                gate_inp_s:  load_fp32(gguf, &format!("{p}ffn_gate_inp.scale"))?,
                gate_up_exps: ExpertTensor::from_gguf(gguf, &format!("{p}ffn_gate_up_exps.weight"))?,
                down_exps:    ExpertTensor::from_gguf(gguf, &format!("{p}ffn_down_exps.weight"))?,
                down_exps_s:  load_fp32(gguf, &format!("{p}ffn_down_exps.scale"))?,
            })
        } else { None };
        let attn_v = if kind == AttnKind::Sliding {
            Some(GpuMatvecTensor::from_gguf(gguf, &format!("{p}attn_v.weight"))?)
        } else { None };
        // layer_output_scale is a [1] f32 — read it to host.
        let los_info = gguf.tensor(&format!("{p}layer_output_scale.weight"))
            .ok_or_else(|| format!("{p}layer_output_scale.weight missing"))?;
        let los_bytes = gguf.tensor_data(&format!("{p}layer_output_scale.weight"))
            .map_err(|e| e.to_string())?.ok_or("los no data")?;
        let layer_output_scale = if los_info.ggml_type == GgmlType::F32 {
            bytemuck::cast_slice::<u8, f32>(los_bytes)[0]
        } else { return Err("layer_output_scale not F32".into()); };

        Ok(Self {
            attn_norm:      load_fp32(gguf, &format!("{p}attn_norm.weight"))?,
            attn_q:         GpuMatvecTensor::from_gguf(gguf, &format!("{p}attn_q.weight"))?,
            attn_k:         GpuMatvecTensor::from_gguf(gguf, &format!("{p}attn_k.weight"))?,
            attn_v,
            attn_q_norm:    load_fp32(gguf, &format!("{p}attn_q_norm.weight"))?,
            attn_k_norm:    load_fp32(gguf, &format!("{p}attn_k_norm.weight"))?,
            attn_output:    GpuMatvecTensor::from_gguf(gguf, &format!("{p}attn_output.weight"))?,
            post_attn_norm: load_fp32(gguf, &format!("{p}post_attention_norm.weight"))?,
            ffn_norm:       load_fp32(gguf, &format!("{p}ffn_norm.weight"))?,
            ffn_gate:       GpuMatvecTensor::from_gguf(gguf, &format!("{p}ffn_gate.weight"))?,
            ffn_up:         GpuMatvecTensor::from_gguf(gguf, &format!("{p}ffn_up.weight"))?,
            ffn_down:       GpuMatvecTensor::from_gguf(gguf, &format!("{p}ffn_down.weight"))?,
            post_ffw_norm:  load_fp32(gguf, &format!("{p}post_ffw_norm.weight"))?,
            layer_output_scale, kind, head_dim, n_kv,
            moe: moe_block,
        })
    }
}

/// Per-layer int8 KV cache. K and V are stored as symmetric int8 with
/// one f32 scale per (token, head) — 4× smaller than f32, and the
/// attention kernel dots K against a quantized Q via dp4a. Sliding and
/// full layers have different (n_kv, head_dim), so each sizes its own.
pub struct Gemma4KvCache {
    k:  DeviceBuf<i8>,    // [max_seq, n_kv, head_dim]
    v:  DeviceBuf<i8>,
    ks: DeviceBuf<f32>,   // [max_seq, n_kv]
    vs: DeviceBuf<f32>,
    max_seq: usize,
    len: usize,
}

impl Gemma4KvCache {
    fn new(n_kv: usize, head_dim: usize, max_seq: usize) -> Result<Self, String> {
        let kv_dim = n_kv * head_dim;
        Ok(Self {
            k:  DeviceBuf::new(max_seq * kv_dim)?,
            v:  DeviceBuf::new(max_seq * kv_dim)?,
            ks: DeviceBuf::new(max_seq * n_kv)?,
            vs: DeviceBuf::new(max_seq * n_kv)?,
            max_seq, len: 0,
        })
    }
}

/// Per-token mutable state: one KV cache per layer.
pub struct Gemma4GpuState {
    caches: Vec<Gemma4KvCache>,
    pub pos: usize,
}

impl Gemma4GpuState {
    pub fn new(model: &Gemma4Model, max_seq: usize) -> Result<Self, String> {
        let cfg = &model.config;
        let mut caches = Vec::with_capacity(cfg.block_count as usize);
        for layer in 0..cfg.block_count as usize {
            caches.push(Gemma4KvCache::new(
                cfg.kv_heads[layer] as usize,
                cfg.head_dim(layer) as usize,
                max_seq)?);
        }
        Ok(Self { caches, pos: 0 })
    }
    pub fn reset(&mut self) {
        for c in &mut self.caches { c.len = 0; }
        self.pos = 0;
    }
}

pub struct GpuGemma4 {
    token_embd:  GpuMatvecTensor,   // also the tied output projection
    output_norm: DeviceBuf<f32>,
    blocks:      Vec<GpuGemma4Block>,

    // RoPE tables — sliding (rotary 256, base 1e4) and full (512, 1e6).
    rope_cos_swa: DeviceBuf<f32>,
    rope_sin_swa: DeviceBuf<f32>,
    rope_cos_full: DeviceBuf<f32>,
    rope_sin_full: DeviceBuf<f32>,

    // Scratch (sized to the per-layer maxima).
    hidden_a:    DeviceBuf<f32>,
    hidden_b:    DeviceBuf<f32>,
    normed:      DeviceBuf<f32>,
    q_buf:       DeviceBuf<f32>,
    k_proj:      DeviceBuf<f32>,
    k_norm:      DeviceBuf<f32>,
    v_norm:      DeviceBuf<f32>,
    attn_concat: DeviceBuf<f32>,
    ffn_a:       DeviceBuf<f32>,
    ffn_b:       DeviceBuf<f32>,
    logits:      DeviceBuf<f32>,
    /// All-ones weight for the plain (unweighted) V RMSNorm.
    ones:        DeviceBuf<f32>,

    // MoE scratch (allocated for all models; tiny when unused).
    moe_logits:  DeviceBuf<f32>,   // [n_expert]
    moe_ids:     DeviceBuf<i32>,   // [n_expert_used]
    moe_weights: DeviceBuf<f32>,   // [n_expert_used]
    moe_in:      DeviceBuf<f32>,   // [hidden] — routed-expert input
    moe_acc:     DeviceBuf<f32>,   // [hidden] — expert mixture accumulator
    cur_mlp:     DeviceBuf<f32>,   // [hidden] — shared-MLP result, kept live
    expert_gu:   DeviceBuf<f32>,   // [n_used · 2·expert_ff] — fused gate_up
    expert_act:  DeviceBuf<f32>,   // [n_used · expert_ff]    — geglu output
    expert_outs: DeviceBuf<f32>,   // [n_used · hidden]       — per-expert down
    xq8_experts: DeviceBuf<u8>,    // batched int8 activation for the 8 experts

    // Kernel modules.
    m_rmsnorm:   Module,
    m_rmsnorm_mh: Module,
    m_rope:      Module,
    m_add:       Module,
    m_geglu:     Module,
    m_softcap:   Module,
    m_scale:     Module,
    m_attn_win:  Module,
    m_embed_q5k: Module,
    m_embed_q6k: Module,
    m_embed_q8_0: Module,
    m_embed_f32: Module,
    m_mv_f32:    Module,
    m_mv_q4k:    Module,
    m_mv_q5k:    Module,
    m_mv_q6k:    Module,
    m_mv_q8_0:   Module,
    m_mv_f16:    Module,
    m_quantize:  Module,
    m_mv_q4k_dp4a: Module,
    m_mv_q5k_dp4a: Module,
    m_mv_q6k_dp4a: Module,
    m_mv_q8_0_dp4a: Module,
    m_moe_topk:  Module,
    m_moe_mv_q6k:  Module,
    m_moe_mv_q8_0: Module,
    m_moe_geglu:   Module,
    m_moe_combine: Module,
    m_kv_write:    Module,
    /// Scratch for the int8-quantized activation feeding the dp4a matvec.
    xq8: DeviceBuf<u8>,

    /// Decode token + position, device-resident so the embed / rope /
    /// attention / KV-write kernels read them at execution time — which
    /// makes the whole forward capturable into one parametric HIP graph.
    d_token: DeviceBuf<u32>,
    d_pos:   DeviceBuf<u32>,
    max_seq: usize,

    stream: Stream,

    // Dimensions.
    hidden:     usize,
    ffn:        usize,
    vocab:      usize,
    n_heads:    usize,
    rms_eps:    f32,
    softcap:    f32,
    sliding_window: usize,
    rope_dim_swa:  usize,
    rope_dim_full: usize,
    // MoE dimensions (0 on dense models).
    n_expert:      usize,
    n_expert_used: usize,
    expert_ff:     usize,
}

impl GpuGemma4 {
    pub fn new(model: &Gemma4Model, gguf: &GgufFile, cache: &KernelCache, max_seq: usize)
        -> Result<Self, String>
    {
        let cfg = &model.config;
        let hidden = cfg.hidden_size as usize;
        let ffn    = cfg.ffn_size as usize;
        let vocab  = cfg.vocab_size as usize;
        let n_heads = cfg.n_heads as usize;
        let hd_max = cfg.head_dim_full.max(cfg.head_dim_swa) as usize;
        let q_max  = n_heads * hd_max;
        let kv_max = cfg.kv_heads.iter().copied().max().unwrap_or(0) as usize * hd_max;

        let token_embd  = GpuMatvecTensor::from_gguf(gguf, "token_embd.weight")?;
        let output_norm = load_fp32(gguf, "output_norm.weight")?;

        let moe = cfg.is_moe();
        // MoE scratch sizes — .max(1) keeps the buffers non-empty on the
        // dense 31B (which leaves the expert counts at 0).
        let n_used_a    = (cfg.expert_used_count as usize).max(1);
        let expert_ff_a = (cfg.expert_ff_size as usize).max(32);
        let mut blocks = Vec::with_capacity(cfg.block_count as usize);
        for layer in 0..cfg.block_count {
            let kind = cfg.attn_kinds[layer as usize];
            blocks.push(GpuGemma4Block::from_gguf(
                gguf, layer, kind,
                cfg.head_dim(layer as usize) as usize,
                cfg.kv_heads[layer as usize] as usize, moe)?);
        }

        // RoPE tables for both kinds.
        let build_rope = |rotary: usize, base: f32| -> Result<(DeviceBuf<f32>, DeviceBuf<f32>), String> {
            let rc = crate::cpu::rope::RopeCache::new(rotary, max_seq, base);
            let mut cos = vec![0.0f32; max_seq * rotary];
            let mut sin = vec![0.0f32; max_seq * rotary];
            for pos in 0..max_seq {
                let (c, s) = rc.get(pos);
                cos[pos*rotary..(pos+1)*rotary].copy_from_slice(c);
                sin[pos*rotary..(pos+1)*rotary].copy_from_slice(s);
            }
            Ok((DeviceBuf::from_slice(&cos)?, DeviceBuf::from_slice(&sin)?))
        };
        let (rope_cos_swa, rope_sin_swa) =
            build_rope(cfg.rope_dim_swa as usize, cfg.rope_freq_base_swa)?;
        let (rope_cos_full, rope_sin_full) =
            build_rope(cfg.rope_dim_full as usize, cfg.rope_freq_base)?;

        let ld = |name: &str, src: &str| -> Result<Module, String> {
            Module::load(&cache.compile(name, src)?)
        };

        let ones = DeviceBuf::from_slice(&vec![1.0f32; hd_max])?;

        // Scratch for the quantized activation: one BlockQ8 (40 bytes)
        // per 32 input elements, sized to the widest matvec.
        let max_in_dim = blocks.iter()
            .flat_map(|b| [b.attn_q.in_dim, b.attn_k.in_dim, b.attn_output.in_dim,
                           b.ffn_gate.in_dim, b.ffn_up.in_dim, b.ffn_down.in_dim])
            .chain(std::iter::once(token_embd.in_dim))
            .max().unwrap_or(0) as usize;
        let xq8 = DeviceBuf::<u8>::new((max_in_dim / 32) * 40)?;

        Ok(Self {
            token_embd, output_norm, blocks,
            rope_cos_swa, rope_sin_swa, rope_cos_full, rope_sin_full,
            hidden_a:    DeviceBuf::new(hidden)?,
            hidden_b:    DeviceBuf::new(hidden)?,
            normed:      DeviceBuf::new(hidden)?,
            q_buf:       DeviceBuf::new(q_max)?,
            k_proj:      DeviceBuf::new(kv_max)?,
            k_norm:      DeviceBuf::new(kv_max)?,
            v_norm:      DeviceBuf::new(kv_max)?,
            attn_concat: DeviceBuf::new(q_max)?,
            ffn_a:       DeviceBuf::new(ffn)?,
            ffn_b:       DeviceBuf::new(ffn)?,
            logits:      DeviceBuf::new(vocab)?,
            ones,
            m_rmsnorm:    ld("rmsnorm", RMSNORM_SRC)?,
            m_rmsnorm_mh: ld("rmsnorm_multihead", RMSNORM_MULTIHEAD_SRC)?,
            m_rope:       ld("rope", ROPE_SRC)?,
            m_add:        ld("add_inplace", ADD_INPLACE_SRC)?,
            m_geglu:      ld("geglu", GEGLU_SRC)?,
            m_softcap:    ld("logit_softcap", LOGIT_SOFTCAP_SRC)?,
            m_scale:      ld("scale_inplace", SCALE_INPLACE_SRC)?,
            m_attn_win:   ld("attn_step_window", ATTN_WINDOW_SRC)?,
            m_embed_q5k:  ld("embed_lookup_q5_k", EMBED_Q5K_SRC)?,
            m_embed_q6k:  ld("embed_lookup_q6_k", EMBED_Q6K_SRC)?,
            m_embed_q8_0: ld("embed_lookup_q8_0", EMBED_Q8_0_SRC)?,
            m_embed_f32:  ld("embed_lookup", EMBED_F32_SRC)?,
            m_mv_f32:     ld("matvec_f32_wave64", MATVEC_F32_W_SRC)?,
            m_mv_q4k:     ld("matvec_q4_k_rowblock", MATVEC_Q4K_W_SRC)?,
            m_mv_q5k:     ld("matvec_q5_k_rowblock", MATVEC_Q5K_W_SRC)?,
            m_mv_q6k:     ld("matvec_q6_k_rowblock", MATVEC_Q6K_W_SRC)?,
            m_mv_q8_0:    ld("matvec_q8_0_wave64", MATVEC_Q8_0_W_SRC)?,
            m_mv_f16:     ld("matvec_f16_wave64", MATVEC_F16_W_SRC)?,
            m_quantize:     ld("quantize_q8", QUANTIZE_Q8_SRC)?,
            m_mv_q4k_dp4a:  ld("matvec_q4_k_dp4a", MATVEC_Q4K_DP4A_SRC)?,
            m_mv_q5k_dp4a:  ld("matvec_q5_k_dp4a", MATVEC_Q5K_DP4A_SRC)?,
            m_mv_q6k_dp4a:  ld("matvec_q6_k_dp4a", MATVEC_Q6K_DP4A_SRC)?,
            m_mv_q8_0_dp4a: ld("matvec_q8_0_dp4a", MATVEC_Q8_0_DP4A_SRC)?,
            m_moe_topk:     ld("moe_topk", MOE_TOPK_SRC)?,
            m_moe_mv_q6k:   ld("moe_matvec_q6k_dp4a", MOE_MATVEC_Q6K_SRC)?,
            m_moe_mv_q8_0:  ld("moe_matvec_q8_0_dp4a", MOE_MATVEC_Q8_0_SRC)?,
            m_moe_geglu:    ld("moe_geglu", MOE_GEGLU_SRC)?,
            m_moe_combine:  ld("moe_combine", MOE_COMBINE_SRC)?,
            m_kv_write:     ld("kv_write", KV_WRITE_SRC)?,
            d_token: DeviceBuf::new(1)?,
            d_pos:   DeviceBuf::new(1)?,
            max_seq,
            moe_logits:  DeviceBuf::new((cfg.expert_count as usize).max(1))?,
            moe_ids:     DeviceBuf::new((cfg.expert_used_count as usize).max(1))?,
            moe_weights: DeviceBuf::new((cfg.expert_used_count as usize).max(1))?,
            moe_in:      DeviceBuf::new(hidden)?,
            moe_acc:     DeviceBuf::new(hidden)?,
            cur_mlp:     DeviceBuf::new(hidden)?,
            expert_gu:   DeviceBuf::new(n_used_a * 2 * expert_ff_a)?,
            expert_act:  DeviceBuf::new(n_used_a * expert_ff_a)?,
            expert_outs: DeviceBuf::new(n_used_a * hidden)?,
            xq8_experts: DeviceBuf::<u8>::new(n_used_a * (expert_ff_a / 32).max(1) * 40)?,
            xq8,
            stream: Stream::new()?,
            hidden, ffn, vocab, n_heads,
            rms_eps: cfg.rms_norm_eps,
            softcap: cfg.final_logit_softcapping,
            sliding_window: cfg.sliding_window as usize,
            rope_dim_swa:  cfg.rope_dim_swa as usize,
            rope_dim_full: cfg.rope_dim_full as usize,
            n_expert:      cfg.expert_count as usize,
            n_expert_used: cfg.expert_used_count as usize,
            expert_ff:     cfg.expert_ff_size as usize,
        })
    }

    // ---- launch helpers ----------------------------------------------------

    fn launch_rmsnorm(&self, x: *mut c_void, w: *mut c_void, y: *mut c_void, n: u32)
        -> Result<(), String>
    {
        let f = self.m_rmsnorm.function("rmsnorm_f32")?;
        let block: u32 = 256;
        let mut xa=x; let mut wa=w; let mut ya=y; let mut na=n; let mut ea=self.rms_eps;
        let mut args: [*mut c_void; 5] = [
            &mut xa as *mut _ as *mut c_void, &mut wa as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void, &mut na as *mut _ as *mut c_void,
            &mut ea as *mut _ as *mut c_void];
        let smem = block * 4;
        unsafe { f.launch((1,1,1),(block,1,1), smem, Some(&self.stream), &mut args) }
    }

    fn launch_rmsnorm_mh(&self, x: *mut c_void, w: *mut c_void, y: *mut c_void,
                         n_heads: u32, head_dim: u32) -> Result<(), String>
    {
        let f = self.m_rmsnorm_mh.function("rmsnorm_multihead_f32")?;
        let block: u32 = 256;
        let mut xa=x; let mut wa=w; let mut ya=y;
        let mut nh=n_heads; let mut hd=head_dim; let mut ea=self.rms_eps;
        let mut args: [*mut c_void; 6] = [
            &mut xa as *mut _ as *mut c_void, &mut wa as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void, &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void, &mut ea as *mut _ as *mut c_void];
        let smem = block * 4;
        unsafe { f.launch((n_heads,1,1),(block,1,1), smem, Some(&self.stream), &mut args) }
    }

    fn launch_rope(&self, x: *mut c_void, n_heads: u32, head_dim: u32, kind: AttnKind)
        -> Result<(), String>
    {
        let f = self.m_rope.function("rope_apply_f32")?;
        let (cos, sin, rd) = match kind {
            AttnKind::Sliding => (self.rope_cos_swa.raw_ptr(), self.rope_sin_swa.raw_ptr(),
                                  self.rope_dim_swa as u32),
            AttnKind::Full    => (self.rope_cos_full.raw_ptr(), self.rope_sin_full.raw_ptr(),
                                  self.rope_dim_full as u32),
        };
        let half = rd / 2;
        let block: u32 = 64;
        let grid_x = (half + block - 1) / block;
        let mut xa=x; let mut ca=cos; let mut sa=sin;
        let mut hd=head_dim; let mut rdv=rd; let mut nh=n_heads;
        let mut p=self.d_pos.raw_ptr();
        let mut args: [*mut c_void; 7] = [
            &mut xa as *mut _ as *mut c_void, &mut ca as *mut _ as *mut c_void,
            &mut sa as *mut _ as *mut c_void, &mut hd as *mut _ as *mut c_void,
            &mut rdv as *mut _ as *mut c_void, &mut nh as *mut _ as *mut c_void,
            &mut p as *mut _ as *mut c_void];
        unsafe { f.launch((grid_x, n_heads, 1),(block,1,1), 0, Some(&self.stream), &mut args) }
    }

    /// Quantize a normed K/V vector to int8 and append it to the cache
    /// at `d_pos` — one f32 scale per head. grid = n_kv heads.
    fn launch_kv_write_q8(&self, src: *mut c_void, dst_q: *mut c_void, dst_s: *mut c_void,
                          n_kv: u32, head_dim: u32) -> Result<(), String>
    {
        let f = self.m_kv_write.function("kv_write_q8_f32")?;
        let mut sa=src; let mut dq=dst_q; let mut ds=dst_s;
        let mut pa=self.d_pos.raw_ptr(); let mut nk=n_kv; let mut hd=head_dim;
        let mut args: [*mut c_void; 6] = [
            &mut sa as *mut _ as *mut c_void, &mut dq as *mut _ as *mut c_void,
            &mut ds as *mut _ as *mut c_void, &mut pa as *mut _ as *mut c_void,
            &mut nk as *mut _ as *mut c_void, &mut hd as *mut _ as *mut c_void];
        unsafe { f.launch((n_kv,1,1),(256,1,1), 0, Some(&self.stream), &mut args) }
    }

    fn launch_add(&self, x: *mut c_void, y: *mut c_void, n: u32) -> Result<(), String> {
        let f = self.m_add.function("add_inplace_f32")?;
        let block: u32 = 256;
        let grid = (n + block - 1) / block;
        let mut xa=x; let mut ya=y; let mut na=n;
        let mut args: [*mut c_void; 3] = [
            &mut xa as *mut _ as *mut c_void, &mut ya as *mut _ as *mut c_void,
            &mut na as *mut _ as *mut c_void];
        unsafe { f.launch((grid,1,1),(block,1,1), 0, Some(&self.stream), &mut args) }
    }

    fn launch_scale(&self, x: *mut c_void, n: u32, s: f32) -> Result<(), String> {
        let f = self.m_scale.function("scale_inplace_f32")?;
        let block: u32 = 256;
        let grid = (n + block - 1) / block;
        let mut xa=x; let mut na=n; let mut sa=s;
        let mut args: [*mut c_void; 3] = [
            &mut xa as *mut _ as *mut c_void, &mut na as *mut _ as *mut c_void,
            &mut sa as *mut _ as *mut c_void];
        unsafe { f.launch((grid,1,1),(block,1,1), 0, Some(&self.stream), &mut args) }
    }

    fn launch_geglu(&self, gate: *mut c_void, up: *mut c_void, out: *mut c_void, n: u32)
        -> Result<(), String>
    {
        let f = self.m_geglu.function("geglu_mul_f32")?;
        let block: u32 = 256;
        let grid = (n + block - 1) / block;
        let mut g=gate; let mut u=up; let mut o=out; let mut na=n;
        let mut args: [*mut c_void; 4] = [
            &mut g as *mut _ as *mut c_void, &mut u as *mut _ as *mut c_void,
            &mut o as *mut _ as *mut c_void, &mut na as *mut _ as *mut c_void];
        unsafe { f.launch((grid,1,1),(block,1,1), 0, Some(&self.stream), &mut args) }
    }

    fn launch_softcap(&self, y: *mut c_void, n: u32) -> Result<(), String> {
        let f = self.m_softcap.function("logit_softcap_f32")?;
        let block: u32 = 256;
        let grid = (n + block - 1) / block;
        let mut ya=y; let mut na=n; let mut c=self.softcap;
        let mut args: [*mut c_void; 3] = [
            &mut ya as *mut _ as *mut c_void, &mut na as *mut _ as *mut c_void,
            &mut c as *mut _ as *mut c_void];
        unsafe { f.launch((grid,1,1),(block,1,1), 0, Some(&self.stream), &mut args) }
    }

    /// Quantize `n_vec` contiguous f32 activations of `in_dim` elements
    /// each into int8 BlockQ8 blocks at `out`, for the dp4a matvec.
    fn launch_quantize_q8(&self, x: *mut c_void, out: *mut c_void,
                          in_dim: u32, n_vec: u32) -> Result<(), String> {
        let f = self.m_quantize.function("quantize_q8_f32")?;
        let mut xa = x;
        let mut oa = out;
        let mut ia = in_dim;
        let mut args: [*mut c_void; 3] = [
            &mut xa as *mut _ as *mut c_void, &mut oa as *mut _ as *mut c_void,
            &mut ia as *mut _ as *mut c_void];
        unsafe { f.launch((in_dim / 32, n_vec, 1), (32, 1, 1), 0, Some(&self.stream), &mut args) }
    }

    fn launch_matvec(&self, w: &GpuMatvecTensor, x: *mut c_void, y: *mut c_void)
        -> Result<(), String>
    {
        self.launch_matvec_raw(w.data.raw_ptr(), w.dtype, w.in_dim, w.out_dim, x, y)
    }

    /// Matvec from an explicit weight pointer — lets the MoE path point
    /// at one expert's slice of a 3D expert tensor.
    fn launch_matvec_raw(&self, w_ptr: *mut c_void, dtype: GgmlType,
                         in_dim: u32, out_dim: u32, x: *mut c_void, y: *mut c_void)
        -> Result<(), String>
    {
        let block: u32 = 64;

        // K-quants + Q8_0: int8 dp4a path — quantize the activation,
        // then matvec with v_dot4_i32_i8. Same stream, ordering implicit.
        // REINSTINCT_GEMMA_NO_DP4A forces the f32/wave64 path (A/B check).
        let dp4a = std::env::var_os("REINSTINCT_GEMMA_NO_DP4A").is_none()
            && match dtype {
                GgmlType::Q4_K => std::env::var_os("REINSTINCT_NO_DP4A_Q4").is_none(),
                GgmlType::Q5_K => std::env::var_os("REINSTINCT_NO_DP4A_Q5").is_none(),
                GgmlType::Q6_K => std::env::var_os("REINSTINCT_NO_DP4A_Q6").is_none(),
                GgmlType::Q8_0 => std::env::var_os("REINSTINCT_NO_DP4A_Q8").is_none(),
                _ => false,
            };
        if dp4a {
            self.launch_quantize_q8(x, self.xq8.raw_ptr(), in_dim, 1)?;
            let (module, kname) = match dtype {
                GgmlType::Q4_K => (&self.m_mv_q4k_dp4a,  "matvec_q4_k_dp4a_f32"),
                GgmlType::Q5_K => (&self.m_mv_q5k_dp4a,  "matvec_q5_k_dp4a_f32"),
                GgmlType::Q6_K => (&self.m_mv_q6k_dp4a,  "matvec_q6_k_dp4a_f32"),
                _              => (&self.m_mv_q8_0_dp4a, "matvec_q8_0_dp4a_f32"),
            };
            let f = module.function(kname)?;
            let grid = (out_dim + Q4K_ROWBLOCK - 1) / Q4K_ROWBLOCK;
            let mut wa = w_ptr;
            let mut xa = self.xq8.raw_ptr();
            let mut ya = y;
            let mut ia = in_dim; let mut oa = out_dim;
            let mut args: [*mut c_void; 5] = [
                &mut wa as *mut _ as *mut c_void, &mut xa as *mut _ as *mut c_void,
                &mut ya as *mut _ as *mut c_void, &mut ia as *mut _ as *mut c_void,
                &mut oa as *mut _ as *mut c_void];
            return unsafe {
                f.launch((grid, 1, 1), (block, 1, 1), 0, Some(&self.stream), &mut args)
            };
        }

        // Q4/5/6_K use the row-blocked kernel; the rest the wave64 ones.
        let (module, kname, grid) = match dtype {
            GgmlType::F32    => (&self.m_mv_f32,  "matvec_f32_wave64",      out_dim),
            GgmlType::Q4_K   => (&self.m_mv_q4k,  "matvec_q4_k_rowblock_f32",
                                 (out_dim + Q4K_ROWBLOCK - 1) / Q4K_ROWBLOCK),
            GgmlType::Q5_K   => (&self.m_mv_q5k,  "matvec_q5_k_rowblock_f32",
                                 (out_dim + Q4K_ROWBLOCK - 1) / Q4K_ROWBLOCK),
            GgmlType::Q6_K   => (&self.m_mv_q6k,  "matvec_q6_k_rowblock_f32",
                                 (out_dim + Q4K_ROWBLOCK - 1) / Q4K_ROWBLOCK),
            GgmlType::Q8_0   => (&self.m_mv_q8_0, "matvec_q8_0_wave64_f32", out_dim),
            GgmlType::F16    => (&self.m_mv_f16,  "matvec_f16_wave64_f32",  out_dim),
            other => return Err(format!("gemma4 matvec: no kernel for {other:?}")),
        };
        let f = module.function(kname)?;
        let mut wa=w_ptr; let mut xa=x; let mut ya=y;
        let mut ia=in_dim; let mut oa=out_dim;
        let mut args: [*mut c_void; 5] = [
            &mut wa as *mut _ as *mut c_void, &mut xa as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void, &mut ia as *mut _ as *mut c_void,
            &mut oa as *mut _ as *mut c_void];
        unsafe { f.launch((grid,1,1),(block,1,1), 0, Some(&self.stream), &mut args) }
    }

    /// Router: softmax + top-k over `n_expert` logits → expert ids and
    /// renormalised weights (device buffers `moe_ids` / `moe_weights`).
    fn launch_moe_topk(&self) -> Result<(), String> {
        let f = self.m_moe_topk.function("moe_topk_f32")?;
        let mut la = self.moe_logits.raw_ptr();
        let mut ne = self.n_expert as i32;
        let mut nu = self.n_expert_used as i32;
        let mut ida = self.moe_ids.raw_ptr();
        let mut wa  = self.moe_weights.raw_ptr();
        let mut args: [*mut c_void; 5] = [
            &mut la as *mut _ as *mut c_void, &mut ne as *mut _ as *mut c_void,
            &mut nu as *mut _ as *mut c_void, &mut ida as *mut _ as *mut c_void,
            &mut wa as *mut _ as *mut c_void];
        let block: u32 = 128;
        let smem = self.n_expert as u32 * 4;
        unsafe { f.launch((1,1,1),(block,1,1), smem, Some(&self.stream), &mut args) }
    }

    /// One launch covering all `n_expert_used` routed experts: grid.y is
    /// the expert slot, the expert id is read from `self.moe_ids` on
    /// device. `xq_stride` is the BlockQ8 count per slot (0 ⇒ all slots
    /// share one activation, the fused gate_up case).
    fn launch_moe_matvec(&self, dtype: GgmlType, slab: *mut c_void, xq: *mut c_void,
                         y: *mut c_void, in_dim: u32, out_dim: u32,
                         bytes_per_expert: u32, xq_stride: u32) -> Result<(), String>
    {
        let (module, kname) = match dtype {
            GgmlType::Q6_K => (&self.m_moe_mv_q6k,  "moe_matvec_q6k_dp4a_f32"),
            GgmlType::Q8_0 => (&self.m_moe_mv_q8_0, "moe_matvec_q8_0_dp4a_f32"),
            other => return Err(format!("moe matvec: no kernel for expert type {other:?}")),
        };
        let f = module.function(kname)?;
        let block: u32 = 64;
        let grid_x = (out_dim + Q4K_ROWBLOCK - 1) / Q4K_ROWBLOCK;
        let mut sa=slab; let mut ida=self.moe_ids.raw_ptr(); let mut xa=xq; let mut ya=y;
        let mut ia=in_dim; let mut oa=out_dim; let mut bpe=bytes_per_expert; let mut st=xq_stride;
        let mut args: [*mut c_void; 8] = [
            &mut sa as *mut _ as *mut c_void, &mut ida as *mut _ as *mut c_void,
            &mut xa as *mut _ as *mut c_void, &mut ya as *mut _ as *mut c_void,
            &mut ia as *mut _ as *mut c_void, &mut oa as *mut _ as *mut c_void,
            &mut bpe as *mut _ as *mut c_void, &mut st as *mut _ as *mut c_void];
        unsafe {
            f.launch((grid_x, self.n_expert_used as u32, 1), (block,1,1), 0,
                     Some(&self.stream), &mut args)
        }
    }

    /// Batched GeGLU over all routed experts: `gu` [n_used, 2·ff_exp] →
    /// `act` [n_used, ff_exp].
    fn launch_moe_geglu(&self, gu: *mut c_void, act: *mut c_void) -> Result<(), String> {
        let f = self.m_moe_geglu.function("moe_geglu_f32")?;
        let block: u32 = 256;
        let total = (self.n_expert_used * self.expert_ff) as u32;
        let grid = (total + block - 1) / block;
        let mut ga=gu; let mut aa=act;
        let mut ff=self.expert_ff as u32; let mut ns=self.n_expert_used as u32;
        let mut args: [*mut c_void; 4] = [
            &mut ga as *mut _ as *mut c_void, &mut aa as *mut _ as *mut c_void,
            &mut ff as *mut _ as *mut c_void, &mut ns as *mut _ as *mut c_void];
        unsafe { f.launch((grid,1,1),(block,1,1), 0, Some(&self.stream), &mut args) }
    }

    /// Weighted sum of the per-expert down outputs into `out` [hidden].
    fn launch_moe_combine(&self, experts: *mut c_void, down_exps_s: *mut c_void,
                          out: *mut c_void) -> Result<(), String>
    {
        let f = self.m_moe_combine.function("moe_combine_f32")?;
        let block: u32 = 256;
        let h = self.hidden as u32;
        let grid = (h + block - 1) / block;
        let mut ea=experts; let mut ida=self.moe_ids.raw_ptr();
        let mut wa=self.moe_weights.raw_ptr(); let mut sa=down_exps_s; let mut oa=out;
        let mut ha=h; let mut nu=self.n_expert_used as u32;
        let mut args: [*mut c_void; 7] = [
            &mut ea as *mut _ as *mut c_void, &mut ida as *mut _ as *mut c_void,
            &mut wa as *mut _ as *mut c_void, &mut sa as *mut _ as *mut c_void,
            &mut oa as *mut _ as *mut c_void, &mut ha as *mut _ as *mut c_void,
            &mut nu as *mut _ as *mut c_void];
        unsafe { f.launch((grid,1,1),(block,1,1), 0, Some(&self.stream), &mut args) }
    }

    /// Embedding lookup — the token row is read from `d_token` on device
    /// (capturable). gemma4's token_embd is Q5_K (31B) or Q8_0 (26B).
    fn launch_embed(&self, table: &GpuMatvecTensor, out: *mut c_void) -> Result<(), String> {
        let hidden = table.in_dim;   // [hidden, vocab]
        let (module, kname, threads, grid): (&Module, &str, u32, u32) = match table.dtype {
            GgmlType::Q5_K => (&self.m_embed_q5k, "embed_lookup_q5_k_f32", 256, hidden/256),
            GgmlType::Q8_0 => (&self.m_embed_q8_0, "embed_lookup_q8_0_f32", 256, (hidden + 255)/256),
            other => return Err(format!("gemma4 embed: no kernel for {other:?}")),
        };
        let f = module.function(kname)?;
        let mut t=table.data.raw_ptr(); let mut o=out;
        let mut row=self.d_token.raw_ptr(); let mut h=hidden;
        let mut args: [*mut c_void; 4] = [
            &mut t as *mut _ as *mut c_void, &mut o as *mut _ as *mut c_void,
            &mut row as *mut _ as *mut c_void, &mut h as *mut _ as *mut c_void];
        unsafe { f.launch((grid,1,1),(threads,1,1), 0, Some(&self.stream), &mut args) }
    }

    /// int8-KV decode attention: dp4a Q·Kᵀ, f32-accumulate P·V.
    fn launch_attn_q8(&self, q: *mut c_void, kq: *mut c_void, ks: *mut c_void,
                      vq: *mut c_void, vs: *mut c_void, out: *mut c_void,
                      n_kv: u32, head_dim: u32, window: u32) -> Result<(), String>
    {
        let f = self.m_attn_win.function("attn_step_q8_f32")?;
        let block: u32 = 256;
        // LDS: qi (head_dim int8) | scores (max_seq f32) | tmp (block f32).
        let smem = head_dim + (self.max_seq as u32 + block) * 4;
        let mut qa=q; let mut kqa=kq; let mut ksa=ks; let mut vqa=vq; let mut vsa=vs;
        let mut oa=out;
        let mut nh=self.n_heads as u32; let mut nkv=n_kv; let mut hd=head_dim;
        let mut tl=self.d_pos.raw_ptr(); let mut wn=window; let mut sc=1.0f32;
        let mut args: [*mut c_void; 12] = [
            &mut qa as *mut _ as *mut c_void, &mut kqa as *mut _ as *mut c_void,
            &mut ksa as *mut _ as *mut c_void, &mut vqa as *mut _ as *mut c_void,
            &mut vsa as *mut _ as *mut c_void, &mut oa as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void, &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void, &mut tl as *mut _ as *mut c_void,
            &mut wn as *mut _ as *mut c_void, &mut sc as *mut _ as *mut c_void];
        unsafe { f.launch((self.n_heads as u32,1,1),(block,1,1), smem,
                          Some(&self.stream), &mut args) }
    }

    /// One Gemma 4 decode step → vocab-length soft-capped logits.
    /// Stage the decode token + position into the device buffers the
    /// embed / rope / attention / KV-write kernels read.
    fn set_inputs(&self, token: u32, pos: usize) -> Result<(), String> {
        self.d_token.copy_from_host(&[token])?;
        self.d_pos.copy_from_host(&[pos as u32])?;
        Ok(())
    }

    /// Enqueue the full forward as a pure async kernel chain on
    /// `self.stream` — no host sync, no readback. Reads `d_token` /
    /// `d_pos`, so it is identical for every token/position and can be
    /// captured once into a HIP graph.
    fn enqueue_forward(&self, state: &Gemma4GpuState) -> Result<(), String> {
        let h = self.hidden as u32;
        self.launch_embed(&self.token_embd, self.hidden_a.raw_ptr())?;
        self.launch_scale(self.hidden_a.raw_ptr(), h, (self.hidden as f32).sqrt())?;
        for (li, block) in self.blocks.iter().enumerate() {
            self.block_forward(block, &state.caches[li])?;
        }
        self.launch_rmsnorm(self.hidden_a.raw_ptr(), self.output_norm.raw_ptr(),
                            self.hidden_b.raw_ptr(), h)?;
        self.launch_matvec(&self.token_embd, self.hidden_b.raw_ptr(), self.logits.raw_ptr())?;
        if self.softcap > 0.0 {
            self.launch_softcap(self.logits.raw_ptr(), self.vocab as u32)?;
        }
        Ok(())
    }

    /// One decode step → vocab-length soft-capped logits.
    pub fn forward_token(&self, token: u32, state: &mut Gemma4GpuState)
        -> Result<Vec<f32>, String>
    {
        self.set_inputs(token, state.pos)?;
        self.enqueue_forward(state)?;
        self.stream.synchronize()?;
        state.pos += 1;
        let mut out = vec![0.0f32; self.vocab];
        self.logits.copy_to_host(&mut out)?;
        Ok(out)
    }

    /// Capture the forward as a parametric HIP graph. The graph reads
    /// `d_token` / `d_pos`, so the single captured executable serves
    /// every decode step — `forward_via_graph` just stages those two
    /// device words and replays the graph with one submission, eliding
    /// the ~1300 individual kernel launches the MI50 is bound by.
    pub fn capture_forward_graph(&self, state: &Gemma4GpuState)
        -> Result<GraphExec, String>
    {
        Graph::begin_capture(&self.stream, HipStreamCaptureMode::Global)?;
        if let Err(e) = self.enqueue_forward(state) {
            let _ = Graph::end_capture(&self.stream);
            return Err(e);
        }
        let graph = Graph::end_capture(&self.stream)?;
        let exec = graph.instantiate()?;
        drop(graph);
        Ok(exec)
    }

    /// Replay a captured forward graph for `token` at `state.pos`.
    pub fn forward_via_graph(&self, exec: &GraphExec, token: u32, state: &mut Gemma4GpuState)
        -> Result<Vec<f32>, String>
    {
        self.set_inputs(token, state.pos)?;
        exec.launch(&self.stream)?;
        self.stream.synchronize()?;
        state.pos += 1;
        let mut out = vec![0.0f32; self.vocab];
        self.logits.copy_to_host(&mut out)?;
        Ok(out)
    }

    /// Timed forward: HIP events around embed / each block / output.
    /// Returns (logits, embed_ms, per_block_ms, output_ms).
    pub fn forward_token_timed(&self, token: u32, state: &mut Gemma4GpuState)
        -> Result<(Vec<f32>, f32, Vec<f32>, f32), String>
    {
        let h = self.hidden as u32;
        let n = self.blocks.len();
        self.set_inputs(token, state.pos)?;
        let ev: Vec<Event> = (0..n + 3).map(|_| Event::new()).collect::<Result<_, _>>()?;
        ev[0].record(&self.stream)?;
        self.launch_embed(&self.token_embd, self.hidden_a.raw_ptr())?;
        self.launch_scale(self.hidden_a.raw_ptr(), h, (self.hidden as f32).sqrt())?;
        ev[1].record(&self.stream)?;
        for (li, block) in self.blocks.iter().enumerate() {
            self.block_forward(block, &state.caches[li])?;
            ev[li + 2].record(&self.stream)?;
        }
        self.launch_rmsnorm(self.hidden_a.raw_ptr(), self.output_norm.raw_ptr(),
                            self.hidden_b.raw_ptr(), h)?;
        self.launch_matvec(&self.token_embd, self.hidden_b.raw_ptr(), self.logits.raw_ptr())?;
        if self.softcap > 0.0 {
            self.launch_softcap(self.logits.raw_ptr(), self.vocab as u32)?;
        }
        ev[n + 2].record(&self.stream)?;
        ev[n + 2].synchronize()?;
        state.pos += 1;

        let embed_ms = Event::elapsed_time(&ev[0], &ev[1])?;
        let mut block_ms = Vec::with_capacity(n);
        for i in 0..n { block_ms.push(Event::elapsed_time(&ev[i + 1], &ev[i + 2])?); }
        let output_ms = Event::elapsed_time(&ev[n + 1], &ev[n + 2])?;
        let mut out = vec![0.0f32; self.vocab];
        self.logits.copy_to_host(&mut out)?;
        Ok((out, embed_ms, block_ms, output_ms))
    }

    /// Batched prefill: process all `tokens` in one pass. Each weight is
    /// streamed once and reused across all P tokens via rocBLAS HGEMM,
    /// rather than the decode path's P sequential weight-streaming
    /// passes. Returns the last token's soft-capped logits.
    ///
    /// Dense path only (the 31B); the 26B MoE prefill is layered on top.
    pub fn prefill_forward(&self, cache: &KernelCache, tokens: &[u32])
        -> Result<Vec<f32>, String>
    {
        use crate::runtime::prefill::batched_matmul_resident;
        use crate::hip::rocblas::Handle;
        let p = tokens.len();
        let h = self.hidden;
        assert!(p > 0 && p <= self.max_seq, "prefill: bad token count");
        let handle = Handle::new()?;
        handle.set_stream(&self.stream)?;
        let m_rope = Module::load(&cache.compile("rope_prefill", ROPE_PREFILL_SRC)?)?;
        let m_attn = Module::load(&cache.compile("attn_prefill", ATTN_PREFILL_SRC)?)?;

        // --- embed P tokens → x [P, hidden] ---
        let x = DeviceBuf::<f32>::new(p * h)?;
        let es = (h as f32).sqrt();
        for (i, &tok) in tokens.iter().enumerate() {
            self.d_token.copy_from_host(&[tok])?;
            self.launch_embed(&self.token_embd, pf_off(x.raw_ptr(), i * h))?;
            self.launch_scale(pf_off(x.raw_ptr(), i * h), h as u32, es)?;
            self.stream.synchronize()?;          // d_token is reused next iter
        }

        let normed = DeviceBuf::<f32>::new(p * h)?;
        let hu = h as u32;
        let gemm = |w: &GpuMatvecTensor, xin: &DeviceBuf<f32>| -> Result<DeviceBuf<f32>, String> {
            self.stream.synchronize()?;          // input ready before the GEMM's cvt
            batched_matmul_resident(cache, &handle, &w.data, w.dtype,
                                    w.in_dim as usize, w.out_dim as usize, xin, p)
        };

        for b in &self.blocks {
            let hd = b.head_dim;
            let n_kv = b.n_kv;
            let q_dim = self.n_heads * hd;
            let kv_dim = n_kv * hd;

            // --- attention ---
            for i in 0..p {
                self.launch_rmsnorm(pf_off(x.raw_ptr(), i*h), b.attn_norm.raw_ptr(),
                                    pf_off(normed.raw_ptr(), i*h), hu)?;
            }
            let q = gemm(&b.attn_q, &normed)?;
            let k = gemm(&b.attn_k, &normed)?;
            let v_gemm;
            let v_ptr = match &b.attn_v {
                Some(wv) => { v_gemm = gemm(wv, &normed)?; v_gemm.raw_ptr() }
                None     => k.raw_ptr(),     // full layers: V is the K projection
            };
            // per-head Q/K norm; V plain norm; RoPE (batched over P).
            for i in 0..p {
                self.launch_rmsnorm_mh(pf_off(q.raw_ptr(), i*q_dim), b.attn_q_norm.raw_ptr(),
                                       pf_off(q.raw_ptr(), i*q_dim), self.n_heads as u32, hd as u32)?;
            }
            self.launch_rope_prefill(&m_rope, q.raw_ptr(), self.n_heads as u32, hd as u32, b.kind, p)?;
            let k_norm = DeviceBuf::<f32>::new(p * kv_dim)?;
            let v_norm = DeviceBuf::<f32>::new(p * kv_dim)?;
            for i in 0..p {
                self.launch_rmsnorm_mh(pf_off(k.raw_ptr(), i*kv_dim), b.attn_k_norm.raw_ptr(),
                                       pf_off(k_norm.raw_ptr(), i*kv_dim), n_kv as u32, hd as u32)?;
                self.launch_rmsnorm_mh(pf_off(v_ptr, i*kv_dim), self.ones.raw_ptr(),
                                       pf_off(v_norm.raw_ptr(), i*kv_dim), n_kv as u32, hd as u32)?;
            }
            self.launch_rope_prefill(&m_rope, k_norm.raw_ptr(), n_kv as u32, hd as u32, b.kind, p)?;
            let window = match b.kind {
                AttnKind::Sliding => self.sliding_window as u32,
                AttnKind::Full    => 0,
            };
            let attn = DeviceBuf::<f32>::new(p * q_dim)?;
            self.launch_attn_prefill(&m_attn, q.raw_ptr(), k_norm.raw_ptr(), v_norm.raw_ptr(),
                                     attn.raw_ptr(), n_kv as u32, hd as u32, window, p)?;
            let attn_out = gemm(&b.attn_output, &attn)?;
            for i in 0..p {
                self.launch_rmsnorm(pf_off(attn_out.raw_ptr(), i*h), b.post_attn_norm.raw_ptr(),
                                    pf_off(normed.raw_ptr(), i*h), hu)?;
                self.launch_add(pf_off(x.raw_ptr(), i*h), pf_off(normed.raw_ptr(), i*h), hu)?;
            }

            // --- dense FFN (GeGLU) ---
            for i in 0..p {
                self.launch_rmsnorm(pf_off(x.raw_ptr(), i*h), b.ffn_norm.raw_ptr(),
                                    pf_off(normed.raw_ptr(), i*h), hu)?;
            }
            let gate = gemm(&b.ffn_gate, &normed)?;
            let up   = gemm(&b.ffn_up,   &normed)?;
            let ff = self.ffn as u32;
            for i in 0..p {
                self.launch_geglu(pf_off(gate.raw_ptr(), i*self.ffn),
                                  pf_off(up.raw_ptr(), i*self.ffn),
                                  pf_off(gate.raw_ptr(), i*self.ffn), ff)?;
            }
            let ffn_out = gemm(&b.ffn_down, &gate)?;
            for i in 0..p {
                self.launch_rmsnorm(pf_off(ffn_out.raw_ptr(), i*h), b.post_ffw_norm.raw_ptr(),
                                    pf_off(normed.raw_ptr(), i*h), hu)?;
                self.launch_add(pf_off(x.raw_ptr(), i*h), pf_off(normed.raw_ptr(), i*h), hu)?;
                self.launch_scale(pf_off(x.raw_ptr(), i*h), hu, b.layer_output_scale)?;
            }
        }

        // --- output: last token only ---
        let last = pf_off(x.raw_ptr(), (p - 1) * h);
        self.launch_rmsnorm(last, self.output_norm.raw_ptr(), self.hidden_b.raw_ptr(), hu)?;
        self.launch_matvec(&self.token_embd, self.hidden_b.raw_ptr(), self.logits.raw_ptr())?;
        if self.softcap > 0.0 {
            self.launch_softcap(self.logits.raw_ptr(), self.vocab as u32)?;
        }
        self.stream.synchronize()?;
        let mut out = vec![0.0f32; self.vocab];
        self.logits.copy_to_host(&mut out)?;
        Ok(out)
    }

    fn launch_rope_prefill(&self, m: &Module, x: *mut c_void, n_heads: u32, head_dim: u32,
                           kind: AttnKind, p: usize) -> Result<(), String>
    {
        let f = m.function("rope_prefill_f32")?;
        let (cos, sin, rd) = match kind {
            AttnKind::Sliding => (self.rope_cos_swa.raw_ptr(), self.rope_sin_swa.raw_ptr(),
                                  self.rope_dim_swa as u32),
            AttnKind::Full    => (self.rope_cos_full.raw_ptr(), self.rope_sin_full.raw_ptr(),
                                  self.rope_dim_full as u32),
        };
        let block: u32 = 64;
        let grid_x = ((rd / 2) + block - 1) / block;
        let mut xa=x; let mut ca=cos; let mut sa=sin;
        let mut hd=head_dim; let mut rdv=rd; let mut nh=n_heads;
        let mut args: [*mut c_void; 6] = [
            &mut xa as *mut _ as *mut c_void, &mut ca as *mut _ as *mut c_void,
            &mut sa as *mut _ as *mut c_void, &mut hd as *mut _ as *mut c_void,
            &mut rdv as *mut _ as *mut c_void, &mut nh as *mut _ as *mut c_void];
        unsafe { f.launch((grid_x, n_heads, p as u32),(block,1,1), 0, Some(&self.stream), &mut args) }
    }

    fn launch_attn_prefill(&self, m: &Module, q: *mut c_void, k: *mut c_void, v: *mut c_void,
                           out: *mut c_void, n_kv: u32, head_dim: u32, window: u32, p: usize)
        -> Result<(), String>
    {
        let f = m.function("attn_prefill_f32")?;
        let block: u32 = 256;
        let smem = (head_dim + p as u32 + block) * 4;
        let mut qa=q; let mut ka=k; let mut va=v; let mut oa=out;
        let mut nh=self.n_heads as u32; let mut nkv=n_kv; let mut hd=head_dim;
        let mut wn=window; let mut sc=1.0f32;
        let mut args: [*mut c_void; 9] = [
            &mut qa as *mut _ as *mut c_void, &mut ka as *mut _ as *mut c_void,
            &mut va as *mut _ as *mut c_void, &mut oa as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void, &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void, &mut wn as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void];
        unsafe { f.launch((self.n_heads as u32, p as u32, 1),(block,1,1), smem,
                          Some(&self.stream), &mut args) }
    }

    /// One transformer block, in place on `hidden_a`. All position-
    /// dependent work (rope, KV write, attention) reads `d_pos`, so the
    /// chain is identical for every decode step.
    fn block_forward(&self, b: &GpuGemma4Block, kv: &Gemma4KvCache)
        -> Result<(), String>
    {
        let h = self.hidden as u32;
        let head_dim = b.head_dim;
        let n_kv = b.n_kv;
        let q_dim = (self.n_heads * head_dim) as u32;
        let kv_dim = (n_kv * head_dim) as u32;

        // --- Attention ---
        self.launch_rmsnorm(self.hidden_a.raw_ptr(), b.attn_norm.raw_ptr(),
                            self.normed.raw_ptr(), h)?;
        // Q: project, per-head norm, RoPE.
        self.launch_matvec(&b.attn_q, self.normed.raw_ptr(), self.q_buf.raw_ptr())?;
        self.launch_rmsnorm_mh(self.q_buf.raw_ptr(), b.attn_q_norm.raw_ptr(),
                               self.q_buf.raw_ptr(), self.n_heads as u32, head_dim as u32)?;
        self.launch_rope(self.q_buf.raw_ptr(), self.n_heads as u32, head_dim as u32,
                         b.kind)?;
        // K: project. V: project, or reuse the K projection.
        self.launch_matvec(&b.attn_k, self.normed.raw_ptr(), self.k_proj.raw_ptr())?;
        let v_src = match &b.attn_v {
            Some(wv) => {
                self.launch_matvec(wv, self.normed.raw_ptr(), self.v_norm.raw_ptr())?;
                self.v_norm.raw_ptr()  // temp holding the raw V projection
            }
            None => self.k_proj.raw_ptr(),  // full layers: V is the K projection
        };
        // K: per-head weighted norm + RoPE.
        self.launch_rmsnorm_mh(self.k_proj.raw_ptr(), b.attn_k_norm.raw_ptr(),
                               self.k_norm.raw_ptr(), n_kv as u32, head_dim as u32)?;
        self.launch_rope(self.k_norm.raw_ptr(), n_kv as u32, head_dim as u32,
                         b.kind)?;
        // V: per-head plain RMSNorm (ones weight). Reads v_src, writes v_norm.
        self.launch_rmsnorm_mh(v_src, self.ones.raw_ptr(), self.v_norm.raw_ptr(),
                               n_kv as u32, head_dim as u32)?;
        // Quantize (k, v) and append into the int8 cache at d_pos — a
        // kernel reading d_pos (a pos-offset memcpy can't be captured).
        self.launch_kv_write_q8(self.k_norm.raw_ptr(), kv.k.raw_ptr(), kv.ks.raw_ptr(),
                                n_kv as u32, head_dim as u32)?;
        self.launch_kv_write_q8(self.v_norm.raw_ptr(), kv.v.raw_ptr(), kv.vs.raw_ptr(),
                                n_kv as u32, head_dim as u32)?;
        let window = match b.kind {
            AttnKind::Sliding => self.sliding_window as u32,
            AttnKind::Full    => 0,
        };
        self.launch_attn_q8(self.q_buf.raw_ptr(), kv.k.raw_ptr(), kv.ks.raw_ptr(),
                            kv.v.raw_ptr(), kv.vs.raw_ptr(), self.attn_concat.raw_ptr(),
                            n_kv as u32, head_dim as u32, window)?;
        // Output projection, post-norm, residual.
        self.launch_matvec(&b.attn_output, self.attn_concat.raw_ptr(),
                           self.hidden_b.raw_ptr())?;
        self.launch_rmsnorm(self.hidden_b.raw_ptr(), b.post_attn_norm.raw_ptr(),
                            self.normed.raw_ptr(), h)?;
        self.launch_add(self.hidden_a.raw_ptr(), self.normed.raw_ptr(), h)?;
        let _ = (q_dim, kv_dim);

        // --- FFN --- (dense GeGLU, or the dual shared-MLP + MoE branch)
        match &b.moe {
            None => {
                self.launch_rmsnorm(self.hidden_a.raw_ptr(), b.ffn_norm.raw_ptr(),
                                    self.normed.raw_ptr(), h)?;
                self.launch_matvec(&b.ffn_gate, self.normed.raw_ptr(), self.ffn_a.raw_ptr())?;
                self.launch_matvec(&b.ffn_up,   self.normed.raw_ptr(), self.ffn_b.raw_ptr())?;
                self.launch_geglu(self.ffn_a.raw_ptr(), self.ffn_b.raw_ptr(),
                                  self.ffn_a.raw_ptr(), self.ffn as u32)?;
                self.launch_matvec(&b.ffn_down, self.ffn_a.raw_ptr(), self.hidden_b.raw_ptr())?;
                self.launch_rmsnorm(self.hidden_b.raw_ptr(), b.post_ffw_norm.raw_ptr(),
                                    self.normed.raw_ptr(), h)?;
                self.launch_add(self.hidden_a.raw_ptr(), self.normed.raw_ptr(), h)?;
            }
            Some(mw) => self.moe_ffn(b, mw)?,
        }

        // Per-layer output scale.
        self.launch_scale(self.hidden_a.raw_ptr(), h, b.layer_output_scale)?;
        Ok(())
    }

    /// Dual FFN for a MoE layer: a shared dense MLP plus a 128-expert
    /// top-8 routed branch, summed, then the shared post-norm + residual.
    /// `hidden_a` holds attn_out on entry and the post-FFN result on exit.
    fn moe_ffn(&self, b: &GpuGemma4Block, mw: &MoeBlock) -> Result<(), String> {
        let h = self.hidden as u32;
        let ff_exp = self.expert_ff as u32;

        // --- Shared MLP --- → cur_mlp (kept live across the MoE branch).
        self.launch_rmsnorm(self.hidden_a.raw_ptr(), b.ffn_norm.raw_ptr(),
                            self.normed.raw_ptr(), h)?;
        self.launch_matvec(&b.ffn_gate, self.normed.raw_ptr(), self.ffn_a.raw_ptr())?;
        self.launch_matvec(&b.ffn_up,   self.normed.raw_ptr(), self.ffn_b.raw_ptr())?;
        self.launch_geglu(self.ffn_a.raw_ptr(), self.ffn_b.raw_ptr(),
                          self.ffn_a.raw_ptr(), self.ffn as u32)?;
        self.launch_matvec(&b.ffn_down, self.ffn_a.raw_ptr(), self.hidden_b.raw_ptr())?;
        self.launch_rmsnorm(self.hidden_b.raw_ptr(), mw.post_ffw_norm_1.raw_ptr(),
                            self.cur_mlp.raw_ptr(), h)?;

        // --- Router --- on attn_out: plain RMSNorm scaled by gate_inp_s,
        // then by 1/√hidden, then the F32 projection to expert logits.
        self.launch_rmsnorm(self.hidden_a.raw_ptr(), mw.gate_inp_s.raw_ptr(),
                            self.normed.raw_ptr(), h)?;
        self.launch_scale(self.normed.raw_ptr(), h, 1.0 / (self.hidden as f32).sqrt())?;
        self.launch_matvec(&mw.gate_inp, self.normed.raw_ptr(), self.moe_logits.raw_ptr())?;
        self.launch_moe_topk()?;

        // --- Routed experts --- fully device-resident: the expert ids
        // from moe_topk stay on device, and one launch per stage covers
        // all n_expert_used experts (grid.y = expert slot). No host
        // round-trip → the whole forward is a pure kernel chain.
        self.launch_rmsnorm(self.hidden_a.raw_ptr(), mw.pre_ffw_norm_2.raw_ptr(),
                            self.moe_in.raw_ptr(), h)?;
        // gate_up: one shared activation, quantized once.
        self.launch_quantize_q8(self.moe_in.raw_ptr(), self.xq8.raw_ptr(), h, 1)?;
        self.launch_moe_matvec(mw.gate_up_exps.dtype, mw.gate_up_exps.data.raw_ptr(),
                               self.xq8.raw_ptr(), self.expert_gu.raw_ptr(), h, 2 * ff_exp,
                               mw.gate_up_exps.bytes_per_expert as u32, 0)?;
        self.launch_moe_geglu(self.expert_gu.raw_ptr(), self.expert_act.raw_ptr())?;
        // down: each expert has its own activation — quantize the batch.
        self.launch_quantize_q8(self.expert_act.raw_ptr(), self.xq8_experts.raw_ptr(),
                                ff_exp, self.n_expert_used as u32)?;
        self.launch_moe_matvec(mw.down_exps.dtype, mw.down_exps.data.raw_ptr(),
                               self.xq8_experts.raw_ptr(), self.expert_outs.raw_ptr(), ff_exp, h,
                               mw.down_exps.bytes_per_expert as u32, ff_exp / 32)?;
        self.launch_moe_combine(self.expert_outs.raw_ptr(), mw.down_exps_s.raw_ptr(),
                                self.moe_acc.raw_ptr())?;
        // cur_moe = rmsnorm(moe_acc, post_ffw_norm_2)
        self.launch_rmsnorm(self.moe_acc.raw_ptr(), mw.post_ffw_norm_2.raw_ptr(),
                            self.normed.raw_ptr(), h)?;
        // combined = cur_mlp + cur_moe → shared post_ffw_norm → residual.
        self.launch_add(self.cur_mlp.raw_ptr(), self.normed.raw_ptr(), h)?;
        self.launch_rmsnorm(self.cur_mlp.raw_ptr(), b.post_ffw_norm.raw_ptr(),
                            self.normed.raw_ptr(), h)?;
        self.launch_add(self.hidden_a.raw_ptr(), self.normed.raw_ptr(), h)?;
        Ok(())
    }
}

