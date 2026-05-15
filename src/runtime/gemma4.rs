//! GPU forward path for Gemma 4 (dense variant — the 31B).
//!
//! Mirrors `cpu::gemma4` on the MI50: weights stay resident in their
//! on-disk quantized form, the forward chains HIP kernels on one
//! stream. Reuses the matvec / rmsnorm / rope / add kernels; the
//! Gemma-specific ones (geglu, logit soft-cap, scale, windowed
//! attention) were added alongside.

use std::ffi::c_void;

use crate::gguf::{GgmlType, GgufFile};
use crate::hip::{DeviceBuf, Module, Stream};
use crate::model::gemma4::{AttnKind, Gemma4Model};
use crate::runtime::KernelCache;
use crate::runtime::qwen35::GpuMatvecTensor;

// Reused kernel sources.
const RMSNORM_SRC:           &str = include_str!("../../kernels/rmsnorm.cpp");
const RMSNORM_MULTIHEAD_SRC: &str = include_str!("../../kernels/rmsnorm_multihead.cpp");
const ROPE_SRC:              &str = include_str!("../../kernels/rope.cpp");
const ADD_INPLACE_SRC:       &str = include_str!("../../kernels/add_inplace.cpp");
const MATVEC_F32_W_SRC:      &str = include_str!("../../kernels/matvec_f32_wave64.cpp");
const MATVEC_Q4K_W_SRC:      &str = include_str!("../../kernels/matvec_q4_k_wave64.cpp");
const MATVEC_Q5K_W_SRC:      &str = include_str!("../../kernels/matvec_q5_k_wave64.cpp");
const MATVEC_Q6K_W_SRC:      &str = include_str!("../../kernels/matvec_q6_k_wave64.cpp");
const MATVEC_Q8_0_W_SRC:     &str = include_str!("../../kernels/matvec_q8_0_wave64.cpp");
const MATVEC_F16_W_SRC:      &str = include_str!("../../kernels/matvec_f16_wave64.cpp");
// Gemma-specific kernel sources.
const GEGLU_SRC:             &str = include_str!("../../kernels/geglu.cpp");
const LOGIT_SOFTCAP_SRC:     &str = include_str!("../../kernels/logit_softcap.cpp");
const SCALE_INPLACE_SRC:     &str = include_str!("../../kernels/scale_inplace.cpp");
const ATTN_WINDOW_SRC:       &str = include_str!("../../kernels/attn_step_window.cpp");
const EMBED_Q5K_SRC:         &str = include_str!("../../kernels/embed_lookup_q5_k.cpp");
const EMBED_Q6K_SRC:         &str = include_str!("../../kernels/embed_lookup_q6_k.cpp");
const EMBED_F32_SRC:         &str = include_str!("../../kernels/embed_lookup.cpp");

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
}

impl GpuGemma4Block {
    fn from_gguf(gguf: &GgufFile, layer: u32, kind: AttnKind,
                 head_dim: usize, n_kv: usize) -> Result<Self, String> {
        let p = format!("blk.{layer}.");
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
        })
    }
}

/// Per-layer KV cache. Sliding and full layers have different
/// (n_kv, head_dim), so each layer sizes its own.
pub struct Gemma4KvCache {
    k: DeviceBuf<f32>,
    v: DeviceBuf<f32>,
    kv_dim: usize,
    max_seq: usize,
    len: usize,
}

impl Gemma4KvCache {
    fn new(n_kv: usize, head_dim: usize, max_seq: usize) -> Result<Self, String> {
        let kv_dim = n_kv * head_dim;
        Ok(Self {
            k: DeviceBuf::new(max_seq * kv_dim)?,
            v: DeviceBuf::new(max_seq * kv_dim)?,
            kv_dim, max_seq, len: 0,
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
    m_embed_f32: Module,
    m_mv_f32:    Module,
    m_mv_q4k:    Module,
    m_mv_q5k:    Module,
    m_mv_q6k:    Module,
    m_mv_q8_0:   Module,
    m_mv_f16:    Module,

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

        let mut blocks = Vec::with_capacity(cfg.block_count as usize);
        for layer in 0..cfg.block_count {
            let kind = cfg.attn_kinds[layer as usize];
            blocks.push(GpuGemma4Block::from_gguf(
                gguf, layer, kind,
                cfg.head_dim(layer as usize) as usize,
                cfg.kv_heads[layer as usize] as usize)?);
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
            m_embed_f32:  ld("embed_lookup", EMBED_F32_SRC)?,
            m_mv_f32:     ld("matvec_f32_wave64", MATVEC_F32_W_SRC)?,
            m_mv_q4k:     ld("matvec_q4_k_wave64", MATVEC_Q4K_W_SRC)?,
            m_mv_q5k:     ld("matvec_q5_k_wave64", MATVEC_Q5K_W_SRC)?,
            m_mv_q6k:     ld("matvec_q6_k_wave64", MATVEC_Q6K_W_SRC)?,
            m_mv_q8_0:    ld("matvec_q8_0_wave64", MATVEC_Q8_0_W_SRC)?,
            m_mv_f16:     ld("matvec_f16_wave64", MATVEC_F16_W_SRC)?,
            stream: Stream::new()?,
            hidden, ffn, vocab, n_heads,
            rms_eps: cfg.rms_norm_eps,
            softcap: cfg.final_logit_softcapping,
            sliding_window: cfg.sliding_window as usize,
            rope_dim_swa:  cfg.rope_dim_swa as usize,
            rope_dim_full: cfg.rope_dim_full as usize,
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

    fn launch_rope(&self, x: *mut c_void, n_heads: u32, head_dim: u32,
                   kind: AttnKind, pos: u32) -> Result<(), String>
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
        let mut hd=head_dim; let mut rdv=rd; let mut nh=n_heads; let mut p=pos;
        let mut args: [*mut c_void; 7] = [
            &mut xa as *mut _ as *mut c_void, &mut ca as *mut _ as *mut c_void,
            &mut sa as *mut _ as *mut c_void, &mut hd as *mut _ as *mut c_void,
            &mut rdv as *mut _ as *mut c_void, &mut nh as *mut _ as *mut c_void,
            &mut p as *mut _ as *mut c_void];
        unsafe { f.launch((grid_x, n_heads, 1),(block,1,1), 0, Some(&self.stream), &mut args) }
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

    fn launch_matvec(&self, w: &GpuMatvecTensor, x: *mut c_void, y: *mut c_void)
        -> Result<(), String>
    {
        let (module, kname) = match w.dtype {
            GgmlType::F32    => (&self.m_mv_f32,  "matvec_f32_wave64"),
            GgmlType::Q4_K   => (&self.m_mv_q4k,  "matvec_q4_k_wave64_f32"),
            GgmlType::Q5_K   => (&self.m_mv_q5k,  "matvec_q5_k_wave64_f32"),
            GgmlType::Q6_K   => (&self.m_mv_q6k,  "matvec_q6_k_wave64_f32"),
            GgmlType::Q8_0   => (&self.m_mv_q8_0, "matvec_q8_0_wave64_f32"),
            GgmlType::F16    => (&self.m_mv_f16,  "matvec_f16_wave64_f32"),
            other => return Err(format!("gemma4 matvec: no kernel for {other:?}")),
        };
        let f = module.function(kname)?;
        let block: u32 = 64;
        let mut wa=w.data.raw_ptr(); let mut xa=x; let mut ya=y;
        let mut ia=w.in_dim; let mut oa=w.out_dim;
        let mut args: [*mut c_void; 5] = [
            &mut wa as *mut _ as *mut c_void, &mut xa as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void, &mut ia as *mut _ as *mut c_void,
            &mut oa as *mut _ as *mut c_void];
        unsafe { f.launch((w.out_dim,1,1),(block,1,1), 0, Some(&self.stream), &mut args) }
    }

    fn launch_embed(&self, table: &GpuMatvecTensor, out: *mut c_void, token: u32)
        -> Result<(), String>
    {
        let hidden = table.in_dim;   // [hidden, vocab]
        let (module, kname, threads, grid): (&Module, &str, u32, u32) = match table.dtype {
            GgmlType::F32  => (&self.m_embed_f32, "embed_lookup_f32",  256, (hidden + 255)/256),
            GgmlType::Q5_K => (&self.m_embed_q5k, "embed_lookup_q5_k_f32", 256, hidden/256),
            GgmlType::Q6_K => (&self.m_embed_q6k, "embed_lookup_q6_k_f32", 256, hidden/256),
            other => return Err(format!("gemma4 embed: no kernel for {other:?}")),
        };
        let f = module.function(kname)?;
        let mut t=table.data.raw_ptr(); let mut o=out; let mut row=token; let mut h=hidden;
        let mut args: [*mut c_void; 4] = [
            &mut t as *mut _ as *mut c_void, &mut o as *mut _ as *mut c_void,
            &mut row as *mut _ as *mut c_void, &mut h as *mut _ as *mut c_void];
        unsafe { f.launch((grid,1,1),(threads,1,1), 0, Some(&self.stream), &mut args) }
    }

    fn launch_attn(&self, q: *mut c_void, kc: *mut c_void, vc: *mut c_void, out: *mut c_void,
                   n_kv: u32, head_dim: u32, total_len: u32, window: u32)
        -> Result<(), String>
    {
        let f = self.m_attn_win.function("attn_step_window_f32")?;
        let block: u32 = 256;
        let win_len = if window > 0 && total_len > window { window } else { total_len };
        let smem = ((head_dim + win_len) + block) * 4;
        let mut qa=q; let mut ka=kc; let mut va=vc; let mut oa=out;
        let mut nh=self.n_heads as u32; let mut nkv=n_kv; let mut hd=head_dim;
        let mut tl=total_len; let mut wn=window; let mut sc=1.0f32;
        let mut args: [*mut c_void; 10] = [
            &mut qa as *mut _ as *mut c_void, &mut ka as *mut _ as *mut c_void,
            &mut va as *mut _ as *mut c_void, &mut oa as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void, &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void, &mut tl as *mut _ as *mut c_void,
            &mut wn as *mut _ as *mut c_void, &mut sc as *mut _ as *mut c_void];
        unsafe { f.launch((self.n_heads as u32,1,1),(block,1,1), smem,
                          Some(&self.stream), &mut args) }
    }

    /// One Gemma 4 decode step → vocab-length soft-capped logits.
    pub fn forward_token(&self, token: u32, state: &mut Gemma4GpuState)
        -> Result<Vec<f32>, String>
    {
        let h = self.hidden as u32;
        // Embed → scale by √hidden.
        self.launch_embed(&self.token_embd, self.hidden_a.raw_ptr(), token)?;
        self.launch_scale(self.hidden_a.raw_ptr(), h, (self.hidden as f32).sqrt())?;

        let pos = state.pos;
        for (li, block) in self.blocks.iter().enumerate() {
            let kv = &mut state.caches[li];
            self.block_forward(block, kv, pos)?;
        }

        // Final norm + tied projection + soft-cap.
        self.launch_rmsnorm(self.hidden_a.raw_ptr(), self.output_norm.raw_ptr(),
                            self.hidden_b.raw_ptr(), h)?;
        self.launch_matvec(&self.token_embd, self.hidden_b.raw_ptr(), self.logits.raw_ptr())?;
        if self.softcap > 0.0 {
            self.launch_softcap(self.logits.raw_ptr(), self.vocab as u32)?;
        }
        self.stream.synchronize()?;
        state.pos = pos + 1;
        let mut out = vec![0.0f32; self.vocab];
        self.logits.copy_to_host(&mut out)?;
        Ok(out)
    }

    /// One transformer block, in place on `hidden_a`.
    fn block_forward(&self, b: &GpuGemma4Block, kv: &mut Gemma4KvCache, pos: usize)
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
                         b.kind, pos as u32)?;
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
                         b.kind, pos as u32)?;
        // V: per-head plain RMSNorm (ones weight). Reads v_src, writes v_norm.
        self.launch_rmsnorm_mh(v_src, self.ones.raw_ptr(), self.v_norm.raw_ptr(),
                               n_kv as u32, head_dim as u32)?;
        // Push (k, v) into the cache.
        self.stream.synchronize()?;
        kv.k.copy_from_device_at(&self.k_norm, pos * kv.kv_dim)?;
        kv.v.copy_from_device_at(&self.v_norm, pos * kv.kv_dim)?;
        let total_len = (pos + 1) as u32;
        let window = match b.kind {
            AttnKind::Sliding => self.sliding_window as u32,
            AttnKind::Full    => 0,
        };
        self.launch_attn(self.q_buf.raw_ptr(), kv.k.raw_ptr(), kv.v.raw_ptr(),
                         self.attn_concat.raw_ptr(), n_kv as u32, head_dim as u32,
                         total_len, window)?;
        // Output projection, post-norm, residual.
        self.launch_matvec(&b.attn_output, self.attn_concat.raw_ptr(),
                           self.hidden_b.raw_ptr())?;
        self.launch_rmsnorm(self.hidden_b.raw_ptr(), b.post_attn_norm.raw_ptr(),
                            self.normed.raw_ptr(), h)?;
        self.launch_add(self.hidden_a.raw_ptr(), self.normed.raw_ptr(), h)?;
        let _ = (q_dim, kv_dim);

        // --- FFN (GeGLU) ---
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

        // Per-layer output scale.
        self.launch_scale(self.hidden_a.raw_ptr(), h, b.layer_output_scale)?;
        Ok(())
    }
}
