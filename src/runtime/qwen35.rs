//! Bridge between the CPU oracle (`cpu::qwen3_5::Qwen35F32Model`) and a
//! growing GPU forward path. This first cut owns *just* the weights and
//! kernels needed to validate the chaining/lifetime/integration story:
//!
//!   embed_lookup(token, token_embd)  →  hidden_a
//!   rmsnorm(hidden_a, output_norm)   →  hidden_b
//!   matvec(hidden_b, output_proj)    →  logits
//!
//! The middle step is not physically meaningful on its own (`output_norm`
//! belongs at the *end* of forward, not after the embedding) but the
//! chain still has a well-defined CPU oracle, so it's a tight integration
//! test for everything we need before we add real layers:
//!   - persistent device buffers for resident weights
//!   - persistent Module/Function handles
//!   - device pointers flowing between launches with no host round-trips
//!   - a final D2H copy to compare against CPU
//!
//! As we add kernels (attention, GDN, FFN, RoPE, ...), this struct grows
//! method-by-method and eventually becomes the real GPU forward path.

use std::ffi::c_void;

use crate::cpu::qwen3_5::Qwen35F32Model;
use crate::hip::{self, DeviceBuf, Module};
use super::KernelCache;

const EMBED_LOOKUP_SOURCE:      &str = include_str!("../../kernels/embed_lookup.cpp");
const RMSNORM_SOURCE:           &str = include_str!("../../kernels/rmsnorm.cpp");
const MATVEC_SOURCE:            &str = include_str!("../../kernels/matvec.cpp");
const SWIGLU_SOURCE:            &str = include_str!("../../kernels/swiglu.cpp");
const RMSNORM_MULTIHEAD_SOURCE: &str = include_str!("../../kernels/rmsnorm_multihead.cpp");
const SPLIT_Q_GATE_SOURCE:      &str = include_str!("../../kernels/split_q_gate.cpp");
const SIGMOID_MUL_SOURCE:       &str = include_str!("../../kernels/sigmoid_mul.cpp");
const ROPE_SOURCE:              &str = include_str!("../../kernels/rope.cpp");
const ATTN_STEP_SOURCE:         &str = include_str!("../../kernels/attn_step.cpp");
const ADD_INPLACE_SOURCE:       &str = include_str!("../../kernels/add_inplace.cpp");

/// FFN weights for a single transformer block, resident on device.
pub struct GpuFfnWeights {
    pub gate: DeviceBuf<f32>,   // [hidden, ffn]
    pub up:   DeviceBuf<f32>,   // [hidden, ffn]
    pub down: DeviceBuf<f32>,   // [ffn,    hidden]
}

impl GpuFfnWeights {
    pub fn from_cpu(gate: &[f32], up: &[f32], down: &[f32]) -> Result<Self, String> {
        Ok(Self {
            gate: DeviceBuf::from_slice(gate)?,
            up:   DeviceBuf::from_slice(up)?,
            down: DeviceBuf::from_slice(down)?,
        })
    }
}

/// All weights for one full-attention transformer block on the GPU.
/// Bundles the attention sub-layer, the post-attention norm, and the
/// FFN sub-layer in the same lifetime.
pub struct GpuFullAttnBlock {
    pub attn:       GpuFullAttnWeights,
    pub post_norm:  DeviceBuf<f32>,    // [hidden] — pre-FFN RMSNorm weight
    pub ffn:        GpuFfnWeights,
}

impl GpuFullAttnBlock {
    pub fn from_cpu(w: &crate::cpu::qwen3_5::FullAttnWeights) -> Result<Self, String> {
        Ok(Self {
            attn:      GpuFullAttnWeights::from_cpu(w)?,
            post_norm: DeviceBuf::from_slice(&w.post_attention_norm)?,
            ffn:       GpuFfnWeights::from_cpu(&w.ffn_gate, &w.ffn_up, &w.ffn_down)?,
        })
    }
}

/// Full-attention block weights for a single transformer block.
pub struct GpuFullAttnWeights {
    pub attn_norm:   DeviceBuf<f32>,   // [hidden]
    pub attn_q:      DeviceBuf<f32>,   // [hidden, 2 * q_dim]   (Q | gate concat)
    pub attn_k:      DeviceBuf<f32>,   // [hidden, kv_dim]
    pub attn_v:      DeviceBuf<f32>,   // [hidden, kv_dim]
    pub attn_q_norm: DeviceBuf<f32>,   // [head_dim]            (per-head)
    pub attn_k_norm: DeviceBuf<f32>,   // [head_dim]
    pub attn_output: DeviceBuf<f32>,   // [q_dim, hidden]
}

impl GpuFullAttnWeights {
    pub fn from_cpu(w: &crate::cpu::qwen3_5::FullAttnWeights) -> Result<Self, String> {
        Ok(Self {
            attn_norm:   DeviceBuf::from_slice(&w.attn_norm)?,
            attn_q:      DeviceBuf::from_slice(&w.attn_q)?,
            attn_k:      DeviceBuf::from_slice(&w.attn_k)?,
            attn_v:      DeviceBuf::from_slice(&w.attn_v)?,
            attn_q_norm: DeviceBuf::from_slice(&w.attn_q_norm)?,
            attn_k_norm: DeviceBuf::from_slice(&w.attn_k_norm)?,
            attn_output: DeviceBuf::from_slice(&w.attn_output)?,
        })
    }
}

/// Per-block KV cache resident on device.
pub struct GpuKvCache {
    pub k: DeviceBuf<f32>,     // [max_seq, n_kv_heads, head_dim]
    pub v: DeviceBuf<f32>,     // [max_seq, n_kv_heads, head_dim]
    pub max_seq: usize,
    pub kv_dim: usize,         // n_kv_heads * head_dim — bytes per slot
    pub len:    usize,         // populated positions [0, len)
}

impl GpuKvCache {
    pub fn new(max_seq: usize, n_kv_heads: usize, head_dim: usize) -> Result<Self, String> {
        let kv_dim = n_kv_heads * head_dim;
        Ok(Self {
            k: DeviceBuf::new(max_seq * kv_dim)?,
            v: DeviceBuf::new(max_seq * kv_dim)?,
            max_seq, kv_dim, len: 0,
        })
    }
    pub fn reset(&mut self) { self.len = 0; }
}

pub struct GpuQwen35 {
    // Resident weights.
    token_embd: DeviceBuf<f32>,           // [vocab, hidden]
    output_norm: DeviceBuf<f32>,          // [hidden]
    /// `None` when `tied_embeddings` — `output_proj` reuses `token_embd`.
    output_proj: Option<DeviceBuf<f32>>,  // [vocab, hidden]

    // Per-call activation scratch (persistent across calls; overwritten each call).
    hidden_a:    DeviceBuf<f32>,   // [hidden]
    hidden_b:    DeviceBuf<f32>,   // [hidden]
    ffn_a:       DeviceBuf<f32>,   // [ffn]
    ffn_b:       DeviceBuf<f32>,   // [ffn]
    q_raw:       DeviceBuf<f32>,   // [2 * q_dim]
    q_buf:       DeviceBuf<f32>,   // [q_dim]
    gate_buf:    DeviceBuf<f32>,   // [q_dim]
    k_raw:       DeviceBuf<f32>,   // [kv_dim]
    v_raw:       DeviceBuf<f32>,   // [kv_dim]
    k_norm:      DeviceBuf<f32>,   // [kv_dim]
    attn_concat: DeviceBuf<f32>,   // [q_dim]
    logits:      DeviceBuf<f32>,   // [vocab]

    // RoPE tables resident on device.
    rope_cos: DeviceBuf<f32>,      // [max_seq, rotary_dim]
    rope_sin: DeviceBuf<f32>,      // [max_seq, rotary_dim]

    // Compiled kernel modules — keep alive for the lifetime of self.
    embed_module:            Module,
    rmsnorm_module:          Module,
    matvec_module:           Module,
    swiglu_module:           Module,
    rmsnorm_multihead_module: Module,
    split_q_gate_module:     Module,
    sigmoid_mul_module:      Module,
    rope_module:             Module,
    attn_step_module:        Module,
    add_inplace_module:      Module,

    // Dimensions.
    hidden:     usize,
    ffn:        usize,
    vocab:      usize,
    n_heads:    usize,
    n_kv_heads: usize,
    head_dim:   usize,
    rotary_dim: usize,
    rms_eps:    f32,
    #[allow(dead_code)]
    max_seq:    usize,
}

impl GpuQwen35 {
    pub fn new(model: &Qwen35F32Model, cache: &KernelCache, max_seq: usize)
        -> Result<Self, String>
    {
        let cfg = &model.model.config;
        let hidden     = cfg.hidden_size      as usize;
        let ffn        = cfg.ffn_size         as usize;
        let vocab      = cfg.vocab_size       as usize;
        let n_heads    = cfg.attn_n_heads     as usize;
        let n_kv_heads = cfg.attn_n_kv_heads  as usize;
        let head_dim   = cfg.attn_head_dim    as usize;
        let rotary_dim = cfg.rope_dim_count   as usize;
        let q_dim  = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;

        let token_embd  = DeviceBuf::from_slice(&model.weights.token_embd)?;
        let output_norm = DeviceBuf::from_slice(&model.weights.output_norm)?;
        let output_proj = if cfg.tied_embeddings {
            None
        } else {
            let w = model.weights.output.as_ref()
                .ok_or("tied_embeddings=false but output.weight is missing")?;
            Some(DeviceBuf::from_slice(w)?)
        };

        let hidden_a    = DeviceBuf::new(hidden)?;
        let hidden_b    = DeviceBuf::new(hidden)?;
        let ffn_a       = DeviceBuf::new(ffn)?;
        let ffn_b       = DeviceBuf::new(ffn)?;
        let q_raw       = DeviceBuf::new(2 * q_dim)?;
        let q_buf       = DeviceBuf::new(q_dim)?;
        let gate_buf    = DeviceBuf::new(q_dim)?;
        let k_raw       = DeviceBuf::new(kv_dim)?;
        let v_raw       = DeviceBuf::new(kv_dim)?;
        let k_norm      = DeviceBuf::new(kv_dim)?;
        let attn_concat = DeviceBuf::new(q_dim)?;
        let logits      = DeviceBuf::new(vocab)?;

        // Build RoPE tables host-side once and upload.
        let rope = crate::cpu::rope::RopeCache::new(rotary_dim, max_seq, cfg.rope_freq_base);
        let mut cos = vec![0.0f32; max_seq * rotary_dim];
        let mut sin = vec![0.0f32; max_seq * rotary_dim];
        for pos in 0..max_seq {
            let (c, s) = rope.get(pos);
            cos[pos * rotary_dim..(pos + 1) * rotary_dim].copy_from_slice(c);
            sin[pos * rotary_dim..(pos + 1) * rotary_dim].copy_from_slice(s);
        }
        let rope_cos = DeviceBuf::from_slice(&cos)?;
        let rope_sin = DeviceBuf::from_slice(&sin)?;

        let embed_hsaco             = cache.compile("embed_lookup",      EMBED_LOOKUP_SOURCE)?;
        let rmsnorm_hsaco           = cache.compile("rmsnorm",           RMSNORM_SOURCE)?;
        let matvec_hsaco            = cache.compile("matvec",            MATVEC_SOURCE)?;
        let swiglu_hsaco            = cache.compile("swiglu",            SWIGLU_SOURCE)?;
        let rmsnorm_multihead_hsaco = cache.compile("rmsnorm_multihead", RMSNORM_MULTIHEAD_SOURCE)?;
        let split_q_gate_hsaco      = cache.compile("split_q_gate",      SPLIT_Q_GATE_SOURCE)?;
        let sigmoid_mul_hsaco       = cache.compile("sigmoid_mul",       SIGMOID_MUL_SOURCE)?;
        let rope_hsaco              = cache.compile("rope",              ROPE_SOURCE)?;
        let attn_step_hsaco         = cache.compile("attn_step",         ATTN_STEP_SOURCE)?;
        let add_inplace_hsaco       = cache.compile("add_inplace",       ADD_INPLACE_SOURCE)?;

        Ok(Self {
            token_embd, output_norm, output_proj,
            hidden_a, hidden_b, ffn_a, ffn_b,
            q_raw, q_buf, gate_buf, k_raw, v_raw, k_norm, attn_concat, logits,
            rope_cos, rope_sin,
            embed_module:             Module::load(&embed_hsaco)?,
            rmsnorm_module:           Module::load(&rmsnorm_hsaco)?,
            matvec_module:            Module::load(&matvec_hsaco)?,
            swiglu_module:            Module::load(&swiglu_hsaco)?,
            rmsnorm_multihead_module: Module::load(&rmsnorm_multihead_hsaco)?,
            split_q_gate_module:      Module::load(&split_q_gate_hsaco)?,
            sigmoid_mul_module:       Module::load(&sigmoid_mul_hsaco)?,
            rope_module:              Module::load(&rope_hsaco)?,
            attn_step_module:         Module::load(&attn_step_hsaco)?,
            add_inplace_module:       Module::load(&add_inplace_hsaco)?,
            hidden, ffn, vocab, n_heads, n_kv_heads, head_dim, rotary_dim,
            rms_eps: cfg.rms_norm_eps,
            max_seq,
        })
    }

    /// q_dim = n_heads * head_dim
    pub fn q_dim(&self) -> usize { self.n_heads * self.head_dim }
    /// kv_dim = n_kv_heads * head_dim
    pub fn kv_dim(&self) -> usize { self.n_kv_heads * self.head_dim }

    fn output_proj_ptr(&self) -> *mut c_void {
        match &self.output_proj {
            Some(buf) => buf.raw_ptr(),
            None      => self.token_embd.raw_ptr(),
        }
    }

    // ---- Per-op launchers ---------------------------------------------------
    //
    // These take raw device pointers so callers can chain them without each
    // op needing to know about DeviceBuf<T>. They never allocate or sync —
    // sync is the caller's responsibility before reading results back.

    fn launch_embed_lookup(&self, table: *mut c_void, out: *mut c_void,
                           token: u32, n: u32) -> Result<(), String>
    {
        let f = self.embed_module.function("embed_lookup_f32")?;
        let block: u32 = 256;
        let grid = (n + block - 1) / block;
        let mut t = table; let mut o = out; let mut row = token; let mut nn = n;
        let mut args: [*mut c_void; 4] = [
            &mut t   as *mut _ as *mut c_void,
            &mut o   as *mut _ as *mut c_void,
            &mut row as *mut _ as *mut c_void,
            &mut nn  as *mut _ as *mut c_void,
        ];
        unsafe { f.launch((grid, 1, 1), (block, 1, 1), 0, None, &mut args) }
    }

    fn launch_rmsnorm(&self, x: *mut c_void, w: *mut c_void, y: *mut c_void,
                      n: u32, eps: f32) -> Result<(), String>
    {
        let f = self.rmsnorm_module.function("rmsnorm_f32")?;
        let block: u32 = 256;
        let mut xa = x; let mut wa = w; let mut ya = y;
        let mut na = n; let mut ea = eps;
        let mut args: [*mut c_void; 5] = [
            &mut xa as *mut _ as *mut c_void,
            &mut wa as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void,
            &mut na as *mut _ as *mut c_void,
            &mut ea as *mut _ as *mut c_void,
        ];
        let smem = block * std::mem::size_of::<f32>() as u32;
        unsafe { f.launch((1, 1, 1), (block, 1, 1), smem, None, &mut args) }
    }

    fn launch_matvec(&self, w: *mut c_void, x: *mut c_void, y: *mut c_void,
                     in_dim: u32, out_dim: u32) -> Result<(), String>
    {
        let f = self.matvec_module.function("matvec_f32")?;
        let block: u32 = 256;
        let mut wa = w; let mut xa = x; let mut ya = y;
        let mut ia = in_dim; let mut oa = out_dim;
        let mut args: [*mut c_void; 5] = [
            &mut wa as *mut _ as *mut c_void,
            &mut xa as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void,
            &mut ia as *mut _ as *mut c_void,
            &mut oa as *mut _ as *mut c_void,
        ];
        let smem = block * std::mem::size_of::<f32>() as u32;
        unsafe { f.launch((out_dim, 1, 1), (block, 1, 1), smem, None, &mut args) }
    }

    fn launch_swiglu(&self, gate: *mut c_void, up: *mut c_void, out: *mut c_void,
                     n: u32) -> Result<(), String>
    {
        let f = self.swiglu_module.function("swiglu_mul_f32")?;
        let block: u32 = 256;
        let grid = (n + block - 1) / block;
        let mut ga = gate; let mut ua = up; let mut oa = out; let mut na = n;
        let mut args: [*mut c_void; 4] = [
            &mut ga as *mut _ as *mut c_void,
            &mut ua as *mut _ as *mut c_void,
            &mut oa as *mut _ as *mut c_void,
            &mut na as *mut _ as *mut c_void,
        ];
        unsafe { f.launch((grid, 1, 1), (block, 1, 1), 0, None, &mut args) }
    }

    fn launch_rmsnorm_multihead(&self, x: *mut c_void, w: *mut c_void, y: *mut c_void,
                                n_heads: u32, head_dim: u32, eps: f32) -> Result<(), String>
    {
        let f = self.rmsnorm_multihead_module.function("rmsnorm_multihead_f32")?;
        let block: u32 = 256;
        let mut xa = x; let mut wa = w; let mut ya = y;
        let mut nh = n_heads; let mut hd = head_dim; let mut ea = eps;
        let mut args: [*mut c_void; 6] = [
            &mut xa as *mut _ as *mut c_void,
            &mut wa as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut ea as *mut _ as *mut c_void,
        ];
        let smem = block * std::mem::size_of::<f32>() as u32;
        unsafe { f.launch((n_heads, 1, 1), (block, 1, 1), smem, None, &mut args) }
    }

    fn launch_split_q_gate(&self, q_raw: *mut c_void, q: *mut c_void, gate: *mut c_void,
                           n_heads: u32, head_dim: u32) -> Result<(), String>
    {
        let f = self.split_q_gate_module.function("split_q_gate_f32")?;
        let block: u32 = 256;
        let total = n_heads * head_dim;
        let grid = (total + block - 1) / block;
        let mut qra = q_raw; let mut qa = q; let mut ga = gate;
        let mut nh = n_heads; let mut hd = head_dim;
        let mut args: [*mut c_void; 5] = [
            &mut qra as *mut _ as *mut c_void,
            &mut qa  as *mut _ as *mut c_void,
            &mut ga  as *mut _ as *mut c_void,
            &mut nh  as *mut _ as *mut c_void,
            &mut hd  as *mut _ as *mut c_void,
        ];
        unsafe { f.launch((grid, 1, 1), (block, 1, 1), 0, None, &mut args) }
    }

    fn launch_sigmoid_mul(&self, x: *mut c_void, gate: *mut c_void, n: u32)
        -> Result<(), String>
    {
        let f = self.sigmoid_mul_module.function("sigmoid_mul_inplace_f32")?;
        let block: u32 = 256;
        let grid = (n + block - 1) / block;
        let mut xa = x; let mut ga = gate; let mut na = n;
        let mut args: [*mut c_void; 3] = [
            &mut xa as *mut _ as *mut c_void,
            &mut ga as *mut _ as *mut c_void,
            &mut na as *mut _ as *mut c_void,
        ];
        unsafe { f.launch((grid, 1, 1), (block, 1, 1), 0, None, &mut args) }
    }

    fn launch_rope(&self, x: *mut c_void, n_heads: u32, pos: u32) -> Result<(), String> {
        let f = self.rope_module.function("rope_apply_f32")?;
        let half = (self.rotary_dim / 2) as u32;
        let block: u32 = 64;
        let grid_x = (half + block - 1) / block;
        let mut xa = x;
        let mut ca = self.rope_cos.raw_ptr();
        let mut sa = self.rope_sin.raw_ptr();
        let mut hd = self.head_dim   as u32;
        let mut rd = self.rotary_dim as u32;
        let mut nh = n_heads;
        let mut p  = pos;
        let mut args: [*mut c_void; 7] = [
            &mut xa as *mut _ as *mut c_void,
            &mut ca as *mut _ as *mut c_void,
            &mut sa as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut rd as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut p  as *mut _ as *mut c_void,
        ];
        unsafe { f.launch((grid_x, n_heads, 1), (block, 1, 1), 0, None, &mut args) }
    }

    fn launch_add_inplace(&self, x: *mut c_void, y: *mut c_void, n: u32) -> Result<(), String> {
        let f = self.add_inplace_module.function("add_inplace_f32")?;
        let block: u32 = 256;
        let grid = (n + block - 1) / block;
        let mut xa = x; let mut ya = y; let mut na = n;
        let mut args: [*mut c_void; 3] = [
            &mut xa as *mut _ as *mut c_void,
            &mut ya as *mut _ as *mut c_void,
            &mut na as *mut _ as *mut c_void,
        ];
        unsafe { f.launch((grid, 1, 1), (block, 1, 1), 0, None, &mut args) }
    }

    fn launch_attn_step(&self, q: *mut c_void, k_cache: *mut c_void, v_cache: *mut c_void,
                        out: *mut c_void, total_len: u32, scaling: f32) -> Result<(), String>
    {
        let f = self.attn_step_module.function("attn_step_f32")?;
        let block: u32 = 256;
        let grid: u32 = self.n_heads as u32;
        let head_dim = self.head_dim as u32;
        let smem = ((head_dim + total_len) + block) * std::mem::size_of::<f32>() as u32;
        let mut qa = q; let mut ka = k_cache; let mut va = v_cache; let mut oa = out;
        let mut nh = self.n_heads as u32;
        let mut nkv = self.n_kv_heads as u32;
        let mut hd = head_dim;
        let mut tl = total_len;
        let mut sc = scaling;
        let mut args: [*mut c_void; 9] = [
            &mut qa as *mut _ as *mut c_void,
            &mut ka as *mut _ as *mut c_void,
            &mut va as *mut _ as *mut c_void,
            &mut oa as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut nkv as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut tl as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
        ];
        unsafe { f.launch((grid, 1, 1), (block, 1, 1), smem, None, &mut args) }
    }

    /// embed → output_norm → output_proj. Returns vocab-length logits.
    /// Composition is artificial (norm doesn't belong here in real
    /// forward), but every kernel and every device pointer in the
    /// pipeline is exercised.
    pub fn embed_norm_proj(&self, token: u32) -> Result<Vec<f32>, String> {
        self.launch_embed_lookup(self.token_embd.raw_ptr(), self.hidden_a.raw_ptr(),
                                 token, self.hidden as u32)?;
        self.launch_rmsnorm(self.hidden_a.raw_ptr(), self.output_norm.raw_ptr(),
                            self.hidden_b.raw_ptr(), self.hidden as u32, self.rms_eps)?;
        self.launch_matvec(self.output_proj_ptr(), self.hidden_b.raw_ptr(),
                           self.logits.raw_ptr(),
                           self.hidden as u32, self.vocab as u32)?;
        hip::Device(0).synchronize()?;
        let mut out = vec![0.0f32; self.vocab];
        self.logits.copy_to_host(&mut out)?;
        Ok(out)
    }

    /// Device-pointer attention step. Reads `input_ptr`, writes the
    /// attention sub-layer output (post-projection, pre-residual) to
    /// `output_ptr`. `input_ptr` is read-only here — must NOT alias
    /// `output_ptr`. No H2D/D2H/sync.
    fn step_full_attention(&self,
        input_ptr: *mut c_void, output_ptr: *mut c_void,
        weights: &GpuFullAttnWeights, kv_cache: &mut GpuKvCache,
    ) -> Result<(), String>
    {
        assert!(kv_cache.len < kv_cache.max_seq, "KV cache full");
        let h_dim  = self.hidden as u32;
        let q_dim  = self.q_dim()  as u32;
        let kv_dim = self.kv_dim() as u32;
        let pos = kv_cache.len;
        let scaling = (self.head_dim as f32).powf(-0.5);

        // normed → output_ptr (output_ptr serves dual duty: normed first,
        //                      then final attn output overwrites it)
        self.launch_rmsnorm(input_ptr, weights.attn_norm.raw_ptr(),
                            output_ptr, h_dim, self.rms_eps)?;
        self.launch_matvec(weights.attn_q.raw_ptr(), output_ptr,
                           self.q_raw.raw_ptr(), h_dim, 2 * q_dim)?;
        self.launch_matvec(weights.attn_k.raw_ptr(), output_ptr,
                           self.k_raw.raw_ptr(), h_dim, kv_dim)?;
        self.launch_matvec(weights.attn_v.raw_ptr(), output_ptr,
                           self.v_raw.raw_ptr(), h_dim, kv_dim)?;
        self.launch_split_q_gate(self.q_raw.raw_ptr(), self.q_buf.raw_ptr(),
                                 self.gate_buf.raw_ptr(),
                                 self.n_heads as u32, self.head_dim as u32)?;
        self.launch_rmsnorm_multihead(self.q_buf.raw_ptr(), weights.attn_q_norm.raw_ptr(),
                                      self.q_buf.raw_ptr(),
                                      self.n_heads as u32, self.head_dim as u32, self.rms_eps)?;
        self.launch_rope(self.q_buf.raw_ptr(), self.n_heads as u32, pos as u32)?;
        self.launch_rmsnorm_multihead(self.k_raw.raw_ptr(), weights.attn_k_norm.raw_ptr(),
                                      self.k_norm.raw_ptr(),
                                      self.n_kv_heads as u32, self.head_dim as u32, self.rms_eps)?;
        self.launch_rope(self.k_norm.raw_ptr(), self.n_kv_heads as u32, pos as u32)?;
        // KV cache push needs the matvec/rope writes to be visible on host
        // before we issue the D2D memcpy (host call).
        hip::Device(0).synchronize()?;
        kv_cache.k.copy_from_device_at(&self.k_norm, pos * kv_cache.kv_dim)?;
        kv_cache.v.copy_from_device_at(&self.v_raw,  pos * kv_cache.kv_dim)?;
        let total_len = (pos + 1) as u32;
        self.launch_attn_step(self.q_buf.raw_ptr(),
                              kv_cache.k.raw_ptr(), kv_cache.v.raw_ptr(),
                              self.attn_concat.raw_ptr(), total_len, scaling)?;
        self.launch_sigmoid_mul(self.attn_concat.raw_ptr(), self.gate_buf.raw_ptr(), q_dim)?;
        self.launch_matvec(weights.attn_output.raw_ptr(), self.attn_concat.raw_ptr(),
                           output_ptr, q_dim, h_dim)?;
        kv_cache.len += 1;
        Ok(())
    }

    /// Device-pointer FFN step. `input_ptr == output_ptr` is allowed
    /// (gate/up matvecs run before down writes back). No H2D/D2H/sync.
    fn step_swiglu_ffn(&self,
        input_ptr: *mut c_void, output_ptr: *mut c_void,
        weights: &GpuFfnWeights,
    ) -> Result<(), String>
    {
        let h = self.hidden as u32;
        let f = self.ffn as u32;
        self.launch_matvec(weights.gate.raw_ptr(), input_ptr,
                           self.ffn_a.raw_ptr(), h, f)?;
        self.launch_matvec(weights.up.raw_ptr(), input_ptr,
                           self.ffn_b.raw_ptr(), h, f)?;
        self.launch_swiglu(self.ffn_a.raw_ptr(), self.ffn_b.raw_ptr(),
                           self.ffn_a.raw_ptr(), f)?;
        self.launch_matvec(weights.down.raw_ptr(), self.ffn_a.raw_ptr(),
                           output_ptr, f, h)?;
        Ok(())
    }

    /// One full transformer block (full-attention variant): pre-norm +
    /// attention + residual + pre-norm + FFN + residual. Mirrors
    /// `cpu::qwen3_5::full_attention_block`.
    ///
    /// Internal buffers used as scratch:
    ///   hidden_a — running hidden state (in/out)
    ///   hidden_b — first attn output, then post-norm output, then ffn output
    pub fn apply_full_attention_block(&self,
        input: &[f32],
        weights: &GpuFullAttnBlock,
        kv_cache: &mut GpuKvCache,
    ) -> Result<Vec<f32>, String>
    {
        assert_eq!(input.len(), self.hidden);
        let h_dim = self.hidden as u32;

        // H2D the input.
        self.hidden_a.copy_from_host(input)?;

        // Sub-layer 1: attention with pre-norm + residual.
        self.step_full_attention(self.hidden_a.raw_ptr(), self.hidden_b.raw_ptr(),
                                 &weights.attn, kv_cache)?;
        self.launch_add_inplace(self.hidden_a.raw_ptr(), self.hidden_b.raw_ptr(), h_dim)?;

        // Sub-layer 2: FFN with pre-norm + residual.
        // post-norm rewrites hidden_b (now serving as `normed`).
        self.launch_rmsnorm(self.hidden_a.raw_ptr(), weights.post_norm.raw_ptr(),
                            self.hidden_b.raw_ptr(), h_dim, self.rms_eps)?;
        // FFN reads hidden_b, writes hidden_b (alias OK — gate/up read
        // happens before down writes within the stream).
        self.step_swiglu_ffn(self.hidden_b.raw_ptr(), self.hidden_b.raw_ptr(),
                             &weights.ffn)?;
        self.launch_add_inplace(self.hidden_a.raw_ptr(), self.hidden_b.raw_ptr(), h_dim)?;

        hip::Device(0).synchronize()?;
        let mut out = vec![0.0f32; self.hidden];
        self.hidden_a.copy_to_host(&mut out)?;
        Ok(out)
    }

    /// Run one decode step of the full-attention block (matches
    /// `cpu::qwen3_5::full_attention_step`).
    ///
    ///   normed     = rmsnorm(input, attn_norm)
    ///   q_raw,k,v  = matvec(normed, {attn_q (2× width), attn_k, attn_v})
    ///   q, gate    = split per-head q_raw into Q + gate
    ///   q          = rmsnorm_per_head(q,      attn_q_norm); rope(q, pos)
    ///   k          = rmsnorm_per_head(k_raw,  attn_k_norm); rope(k, pos)
    ///   kv_cache.push(k, v) at position `cache_len`
    ///   attn       = attn_step(q, K_cache[0..len+1], V_cache[0..len+1])
    ///   attn      *= sigmoid(gate)
    ///   out        = matvec(attn, attn_output)
    ///
    /// Returns the hidden-sized output as a Vec<f32>. Increments
    /// `kv_cache.len` by 1.
    pub fn apply_full_attention(&self,
        input: &[f32],
        weights: &GpuFullAttnWeights,
        kv_cache: &mut GpuKvCache,
    ) -> Result<Vec<f32>, String>
    {
        assert_eq!(input.len(), self.hidden, "input must be hidden-sized");
        self.hidden_a.copy_from_host(input)?;
        self.step_full_attention(self.hidden_a.raw_ptr(), self.hidden_b.raw_ptr(),
                                 weights, kv_cache)?;
        hip::Device(0).synchronize()?;
        let mut out = vec![0.0f32; self.hidden];
        self.hidden_b.copy_to_host(&mut out)?;
        Ok(out)
    }

    /// Run a SwiGLU FFN block on `input` with the given block's weights.
    ///
    ///   gate = matvec(input, gate_w, hidden→ffn)
    ///   up   = matvec(input, up_w,   hidden→ffn)
    ///   mid  = silu(gate) * up                  (in-place into ffn_a)
    ///   out  = matvec(mid,  down_w, ffn→hidden)
    ///
    /// Returns the hidden-sized FFN output as a host Vec<f32>. `input`
    /// is copied H2D into the internal hidden_a scratch.
    pub fn apply_swiglu_ffn(&self, input: &[f32], weights: &GpuFfnWeights)
        -> Result<Vec<f32>, String>
    {
        assert_eq!(input.len(), self.hidden, "input must be hidden-sized");
        self.hidden_a.copy_from_host(input)?;
        self.step_swiglu_ffn(self.hidden_a.raw_ptr(), self.hidden_b.raw_ptr(), weights)?;
        hip::Device(0).synchronize()?;
        let mut out = vec![0.0f32; self.hidden];
        self.hidden_b.copy_to_host(&mut out)?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    use crate::gguf::GgufFile;

    fn fixture_path() -> Option<PathBuf> {
        if let Ok(p) = std::env::var("REINSTINCT_GGUF_FIXTURE") {
            return Some(PathBuf::from(p));
        }
        let home = std::env::var_os("HOME")?;
        let p = PathBuf::from(home).join("models/qwen-3.5-0.8B/Qwen3.5-0.8B-UD-Q4_K_XL.gguf");
        p.exists().then_some(p)
    }

    #[test]
    fn embed_norm_proj_matches_cpu_chain() {
        if hip::device_count().ok().unwrap_or(0) < 1 { eprintln!("skip: no HIP device"); return; }
        let _dev = hip::Device::set(0).unwrap();
        let Some(path) = fixture_path() else { eprintln!("skip: no GGUF fixture"); return };
        let cache = match KernelCache::new() {
            Ok(c) => c,
            Err(e) => { eprintln!("skip: kernel cache: {e}"); return }
        };

        let g = GgufFile::open(&path).expect("open gguf");
        let m = Qwen35F32Model::load(&g).expect("load model");
        let cfg = &m.model.config;
        let hidden = cfg.hidden_size as usize;
        let vocab  = cfg.vocab_size as usize;

        let gpu = GpuQwen35::new(&m, &cache, 32).expect("new GpuQwen35");

        // Test on a couple of tokens including EOS and a mid-vocab.
        for &token in &[cfg.eos_token_id, 100u32, 50_000u32] {
            // CPU oracle: embed → output_norm → output_proj.
            let off = token as usize * hidden;
            let embed = &m.weights.token_embd[off..off + hidden];
            let mut normed = vec![0.0f32; hidden];
            crate::cpu::ops::rmsnorm(embed, &m.weights.output_norm, cfg.rms_norm_eps, &mut normed);
            let proj_w = if cfg.tied_embeddings {
                m.weights.token_embd.as_slice()
            } else {
                m.weights.output.as_ref().unwrap().as_slice()
            };
            let mut cpu_logits = vec![0.0f32; vocab];
            crate::cpu::ops::matvec(&normed, proj_w, hidden, vocab, &mut cpu_logits);

            let gpu_logits = gpu.embed_norm_proj(token).expect("gpu chain");

            // Reduction order differs in rmsnorm + matvec, so bit-equality
            // isn't expected. Use combined abs+rel tolerance: a value
            // passes if it's within EITHER bound. Small logits get the
            // abs bound (avoiding catastrophic cancellation in the rel
            // metric); large logits get the rel bound.
            const ABS_TOL: f32 = 5.0e-4;
            const REL_TOL: f32 = 1.0e-3;
            let mut max_abs = 0.0f32;
            let mut worst_violation = 0.0f32;
            let mut worst_at = 0usize;
            for i in 0..vocab {
                let d = (gpu_logits[i] - cpu_logits[i]).abs();
                if d > max_abs { max_abs = d; }
                let allowed = ABS_TOL.max(REL_TOL * cpu_logits[i].abs());
                let v = d - allowed;
                if v > worst_violation { worst_violation = v; worst_at = i; }
            }
            eprintln!("token {token}: max_abs={max_abs:.3e}, worst violation {:.3e} at idx {}",
                worst_violation, worst_at);
            assert!(worst_violation <= 0.0,
                "token {token}: idx {worst_at} gpu={} cpu={} diff exceeds abs={ABS_TOL:.1e} or rel={REL_TOL:.1e}",
                gpu_logits[worst_at], cpu_logits[worst_at]);
        }
    }

    #[test]
    fn swiglu_ffn_matches_cpu_for_real_block() {
        if hip::device_count().ok().unwrap_or(0) < 1 { eprintln!("skip: no HIP device"); return; }
        let _dev = hip::Device::set(0).unwrap();
        let Some(path) = fixture_path() else { eprintln!("skip: no GGUF fixture"); return };
        let cache = match KernelCache::new() {
            Ok(c) => c,
            Err(e) => { eprintln!("skip: kernel cache: {e}"); return }
        };

        let g = GgufFile::open(&path).unwrap();
        let m = Qwen35F32Model::load(&g).unwrap();
        let cfg = &m.model.config;
        let h = cfg.hidden_size as usize;
        let f = cfg.ffn_size as usize;

        // Pull FFN weights from a couple of real blocks: one linear-attn (e.g. block 0)
        // and one full-attn (e.g. block 3 — pattern is L,L,L,F,L,L,L,F,...).
        // The FFN layout is identical between block kinds, so this also confirms
        // we're reading the same tensors per block.
        use crate::cpu::qwen3_5::BlockWeights;
        let block_indices = [0usize, 3];
        for &block_idx in &block_indices {
            let (gate_w, up_w, down_w) = match &m.weights.blocks[block_idx] {
                BlockWeights::LinearAttention(w) => (&w.ffn_gate, &w.ffn_up, &w.ffn_down),
                BlockWeights::FullAttention(w)   => (&w.ffn_gate, &w.ffn_up, &w.ffn_down),
            };

            // Build a deterministic, realistic-magnitude pre-norm input
            // (RMSNorm output has rms ≈ 1 by construction, so values O(1) are fair).
            let mut s: u64 = 0xDEADBEEF ^ block_idx as u64;
            let mut rng = || { s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                               ((s >> 33) as u32 as f32 / u32::MAX as f32) - 0.5 };
            let input: Vec<f32> = (0..h).map(|_| rng() * 2.0).collect();

            // CPU oracle.
            let mut cpu_out = vec![0.0f32; h];
            crate::cpu::qwen3_5::swiglu_ffn(&input, gate_w, up_w, down_w, h, f, &mut cpu_out);

            // GPU.
            let weights = GpuFfnWeights::from_cpu(gate_w, up_w, down_w).expect("alloc ffn weights");
            let gpu = GpuQwen35::new(&m, &cache, 32).expect("new GpuQwen35");
            let gpu_out = gpu.apply_swiglu_ffn(&input, &weights).expect("gpu ffn");

            // Compare with combined abs+rel tolerance.
            const ABS_TOL: f32 = 1.0e-3;
            const REL_TOL: f32 = 5.0e-3;
            let mut max_abs = 0.0f32;
            let mut worst_violation = 0.0f32;
            let mut worst_at = 0usize;
            for i in 0..h {
                let d = (gpu_out[i] - cpu_out[i]).abs();
                if d > max_abs { max_abs = d; }
                let allowed = ABS_TOL.max(REL_TOL * cpu_out[i].abs());
                let v = d - allowed;
                if v > worst_violation { worst_violation = v; worst_at = i; }
            }
            eprintln!("block {block_idx} ffn: max_abs={max_abs:.3e}, worst_violation={:.3e} at {worst_at}",
                worst_violation);
            assert!(worst_violation <= 0.0,
                "block {block_idx} ffn[{worst_at}]: gpu={} cpu={} diff exceeds abs={ABS_TOL:.1e} or rel={REL_TOL:.1e}",
                gpu_out[worst_at], cpu_out[worst_at]);
        }
    }

    #[test]
    fn full_attention_block_matches_cpu_for_real_block() {
        if hip::device_count().ok().unwrap_or(0) < 1 { eprintln!("skip: no HIP device"); return; }
        let _dev = hip::Device::set(0).unwrap();
        let Some(path) = fixture_path() else { eprintln!("skip: no GGUF fixture"); return };
        let cache = match KernelCache::new() {
            Ok(c) => c,
            Err(e) => { eprintln!("skip: kernel cache: {e}"); return }
        };

        use crate::cpu::qwen3_5::{BlockWeights, LayerKvCache, full_attention_block};
        use crate::cpu::rope::RopeCache;

        let g = GgufFile::open(&path).unwrap();
        let m = Qwen35F32Model::load(&g).unwrap();
        let cfg = &m.model.config;
        let h = cfg.hidden_size as usize;

        let block_idx = m.model.block_kinds.iter()
            .position(|k| matches!(k, crate::model::qwen3_5::BlockKind::FullAttention))
            .expect("model has at least one FullAttention block");
        let weights = match &m.weights.blocks[block_idx] {
            BlockWeights::FullAttention(w) => w,
            _ => unreachable!(),
        };
        eprintln!("validating full block {block_idx} (FullAttention + FFN)");

        let max_seq = 16usize;
        let mut layer_kv = LayerKvCache::new(
            max_seq,
            cfg.attn_n_kv_heads as usize,
            cfg.attn_head_dim   as usize,
        );
        let rope = RopeCache::new(cfg.rope_dim_count as usize, max_seq, cfg.rope_freq_base);

        let gpu = GpuQwen35::new(&m, &cache, max_seq).expect("new GpuQwen35");
        let gpu_block = GpuFullAttnBlock::from_cpu(weights).expect("upload block");
        let mut gpu_kv = GpuKvCache::new(
            max_seq,
            cfg.attn_n_kv_heads as usize,
            cfg.attn_head_dim   as usize,
        ).expect("alloc gpu kv");

        let mut s: u64 = 0xB10C_C0DE;
        let mut rng = || { s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                           ((s >> 33) as u32 as f32 / u32::MAX as f32) - 0.5 };

        let n_steps = 4usize;
        for step in 0..n_steps {
            let input: Vec<f32> = (0..h).map(|_| rng() * 2.0).collect();

            // CPU oracle: in-place block.
            let mut cpu_state = input.clone();
            full_attention_block(&mut cpu_state, weights, cfg, &mut layer_kv, &rope, step);

            let gpu_state = gpu.apply_full_attention_block(&input, &gpu_block, &mut gpu_kv)
                .expect("gpu block");

            const ABS_TOL: f32 = 5.0e-3;
            const REL_TOL: f32 = 5.0e-3;
            let mut max_abs = 0.0f32;
            let mut worst_violation = 0.0f32;
            let mut worst_at = 0usize;
            for i in 0..h {
                let d = (gpu_state[i] - cpu_state[i]).abs();
                if d > max_abs { max_abs = d; }
                let allowed = ABS_TOL.max(REL_TOL * cpu_state[i].abs());
                let v = d - allowed;
                if v > worst_violation { worst_violation = v; worst_at = i; }
            }
            eprintln!("block step {step}: max_abs={max_abs:.3e}, worst_violation={:.3e}",
                worst_violation);
            assert!(worst_violation <= 0.0,
                "block step {step} idx {worst_at}: gpu={} cpu={} exceeds tol",
                gpu_state[worst_at], cpu_state[worst_at]);
        }
    }

    #[test]
    fn full_attention_step_matches_cpu_for_real_block() {
        if hip::device_count().ok().unwrap_or(0) < 1 { eprintln!("skip: no HIP device"); return; }
        let _dev = hip::Device::set(0).unwrap();
        let Some(path) = fixture_path() else { eprintln!("skip: no GGUF fixture"); return };
        let cache = match KernelCache::new() {
            Ok(c) => c,
            Err(e) => { eprintln!("skip: kernel cache: {e}"); return }
        };

        use crate::cpu::qwen3_5::{BlockWeights, LayerKvCache, full_attention_step};
        use crate::cpu::rope::RopeCache;

        let g = GgufFile::open(&path).unwrap();
        let m = Qwen35F32Model::load(&g).unwrap();
        let cfg = &m.model.config;
        let h = cfg.hidden_size as usize;

        // Find the first FullAttention block (Qwen 3.5 pattern: L,L,L,F,...).
        let block_idx = m.model.block_kinds.iter()
            .position(|k| matches!(k, crate::model::qwen3_5::BlockKind::FullAttention))
            .expect("model has at least one FullAttention block");
        let weights = match &m.weights.blocks[block_idx] {
            BlockWeights::FullAttention(w) => w,
            _ => unreachable!(),
        };
        eprintln!("validating block {block_idx} (FullAttention)");

        let max_seq = 16usize;

        // CPU side.
        let mut layer_kv = LayerKvCache::new(
            max_seq,
            cfg.attn_n_kv_heads as usize,
            cfg.attn_head_dim   as usize,
        );
        let rope = RopeCache::new(cfg.rope_dim_count as usize, max_seq, cfg.rope_freq_base);

        // GPU side.
        let gpu = GpuQwen35::new(&m, &cache, max_seq).expect("new GpuQwen35");
        let gpu_w = GpuFullAttnWeights::from_cpu(weights).expect("upload attn weights");
        let mut gpu_kv = GpuKvCache::new(
            max_seq,
            cfg.attn_n_kv_heads as usize,
            cfg.attn_head_dim   as usize,
        ).expect("alloc gpu kv");

        // Drive both sides with the same sequence of inputs to verify KV
        // accumulates correctly. Realistic magnitudes: rmsnorm output is
        // O(1), so we feed inputs that look like post-residual hidden states.
        let mut s: u64 = 0xA77E_FACE_CAFE;
        let mut rng = || { s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                           ((s >> 33) as u32 as f32 / u32::MAX as f32) - 0.5 };

        let n_steps = 4usize;
        for step in 0..n_steps {
            let input: Vec<f32> = (0..h).map(|_| rng() * 2.0).collect();

            // CPU oracle.
            let mut cpu_out = vec![0.0f32; h];
            full_attention_step(&input, weights, cfg, &mut layer_kv, &rope, step, &mut cpu_out);

            // GPU.
            let gpu_out = gpu.apply_full_attention(&input, &gpu_w, &mut gpu_kv)
                .expect("gpu apply_full_attention");

            // Compare.
            const ABS_TOL: f32 = 1.0e-3;
            const REL_TOL: f32 = 5.0e-3;
            let mut max_abs = 0.0f32;
            let mut worst_violation = 0.0f32;
            let mut worst_at = 0usize;
            for i in 0..h {
                let d = (gpu_out[i] - cpu_out[i]).abs();
                if d > max_abs { max_abs = d; }
                let allowed = ABS_TOL.max(REL_TOL * cpu_out[i].abs());
                let v = d - allowed;
                if v > worst_violation { worst_violation = v; worst_at = i; }
            }
            eprintln!("step {step} (cache_len before push = {step}): max_abs={max_abs:.3e}, worst_violation={:.3e}",
                worst_violation);
            assert!(worst_violation <= 0.0,
                "step {step} idx {worst_at}: gpu={} cpu={} exceeds tol",
                gpu_out[worst_at], cpu_out[worst_at]);
            assert_eq!(gpu_kv.len, layer_kv.len(),
                "step {step}: GPU kv len {} doesn't match CPU {}", gpu_kv.len, layer_kv.len());
        }
    }

    #[test]
    fn embed_norm_proj_is_deterministic() {
        if hip::device_count().ok().unwrap_or(0) < 1 { eprintln!("skip: no HIP device"); return; }
        let _dev = hip::Device::set(0).unwrap();
        let Some(path) = fixture_path() else { eprintln!("skip: no GGUF fixture"); return };
        let cache = match KernelCache::new() {
            Ok(c) => c,
            Err(e) => { eprintln!("skip: kernel cache: {e}"); return }
        };

        let g = GgufFile::open(&path).unwrap();
        let m = Qwen35F32Model::load(&g).unwrap();
        let gpu = GpuQwen35::new(&m, &cache, 32).unwrap();
        let token = m.model.config.eos_token_id;

        let a = gpu.embed_norm_proj(token).unwrap();
        let b = gpu.embed_norm_proj(token).unwrap();
        assert_eq!(a.len(), b.len());
        for i in 0..a.len() {
            assert_eq!(a[i].to_bits(), b[i].to_bits(),
                "non-deterministic at index {i}: {} vs {}", a[i], b[i]);
        }
    }
}
