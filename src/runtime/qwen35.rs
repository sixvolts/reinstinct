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

const EMBED_LOOKUP_SOURCE: &str = include_str!("../../kernels/embed_lookup.cpp");
const RMSNORM_SOURCE:      &str = include_str!("../../kernels/rmsnorm.cpp");
const MATVEC_SOURCE:       &str = include_str!("../../kernels/matvec.cpp");
const SWIGLU_SOURCE:       &str = include_str!("../../kernels/swiglu.cpp");

/// FFN weights for a single transformer block, resident on device.
/// One of these per block in the eventual full GPU model.
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

pub struct GpuQwen35 {
    // Resident weights.
    token_embd: DeviceBuf<f32>,           // [vocab, hidden]
    output_norm: DeviceBuf<f32>,          // [hidden]
    /// `None` when `tied_embeddings` — `output_proj` reuses `token_embd`.
    output_proj: Option<DeviceBuf<f32>>,  // [vocab, hidden]

    // Per-call activation scratch (persistent across calls; overwritten each call).
    hidden_a: DeviceBuf<f32>,
    hidden_b: DeviceBuf<f32>,
    ffn_a:    DeviceBuf<f32>,   // [ffn] gate proj output / swiglu output
    ffn_b:    DeviceBuf<f32>,   // [ffn] up proj output
    logits:   DeviceBuf<f32>,

    // Compiled kernel modules — keep alive for the lifetime of self.
    embed_module:   Module,
    rmsnorm_module: Module,
    matvec_module:  Module,
    swiglu_module:  Module,

    // Dimensions.
    hidden:  usize,
    ffn:     usize,
    vocab:   usize,
    rms_eps: f32,
}

impl GpuQwen35 {
    pub fn new(model: &Qwen35F32Model, cache: &KernelCache) -> Result<Self, String> {
        let cfg = &model.model.config;
        let hidden = cfg.hidden_size as usize;
        let ffn    = cfg.ffn_size   as usize;
        let vocab  = cfg.vocab_size as usize;

        let token_embd  = DeviceBuf::from_slice(&model.weights.token_embd)?;
        let output_norm = DeviceBuf::from_slice(&model.weights.output_norm)?;
        let output_proj = if cfg.tied_embeddings {
            None
        } else {
            let w = model.weights.output.as_ref()
                .ok_or("tied_embeddings=false but output.weight is missing")?;
            Some(DeviceBuf::from_slice(w)?)
        };

        let hidden_a = DeviceBuf::new(hidden)?;
        let hidden_b = DeviceBuf::new(hidden)?;
        let ffn_a    = DeviceBuf::new(ffn)?;
        let ffn_b    = DeviceBuf::new(ffn)?;
        let logits   = DeviceBuf::new(vocab)?;

        let embed_hsaco   = cache.compile("embed_lookup", EMBED_LOOKUP_SOURCE)?;
        let rmsnorm_hsaco = cache.compile("rmsnorm",      RMSNORM_SOURCE)?;
        let matvec_hsaco  = cache.compile("matvec",       MATVEC_SOURCE)?;
        let swiglu_hsaco  = cache.compile("swiglu",       SWIGLU_SOURCE)?;

        Ok(Self {
            token_embd, output_norm, output_proj,
            hidden_a, hidden_b, ffn_a, ffn_b, logits,
            embed_module:   Module::load(&embed_hsaco)?,
            rmsnorm_module: Module::load(&rmsnorm_hsaco)?,
            matvec_module:  Module::load(&matvec_hsaco)?,
            swiglu_module:  Module::load(&swiglu_hsaco)?,
            hidden, ffn, vocab, rms_eps: cfg.rms_norm_eps,
        })
    }

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
        // gate_w · input → ffn_a   (in_dim=hidden, out_dim=ffn)
        self.launch_matvec(weights.gate.raw_ptr(), self.hidden_a.raw_ptr(),
                           self.ffn_a.raw_ptr(), self.hidden as u32, self.ffn as u32)?;
        // up_w · input → ffn_b
        self.launch_matvec(weights.up.raw_ptr(), self.hidden_a.raw_ptr(),
                           self.ffn_b.raw_ptr(), self.hidden as u32, self.ffn as u32)?;
        // silu(ffn_a) * ffn_b → ffn_a (in place, mirrors CPU swiglu_mul)
        self.launch_swiglu(self.ffn_a.raw_ptr(), self.ffn_b.raw_ptr(),
                           self.ffn_a.raw_ptr(), self.ffn as u32)?;
        // down_w · ffn_a → hidden_b   (in_dim=ffn, out_dim=hidden)
        self.launch_matvec(weights.down.raw_ptr(), self.ffn_a.raw_ptr(),
                           self.hidden_b.raw_ptr(), self.ffn as u32, self.hidden as u32)?;
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

        let gpu = GpuQwen35::new(&m, &cache).expect("new GpuQwen35");

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
            let gpu = GpuQwen35::new(&m, &cache).expect("new GpuQwen35");
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
        let gpu = GpuQwen35::new(&m, &cache).unwrap();
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
