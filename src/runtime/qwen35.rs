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

pub struct GpuQwen35 {
    // Resident weights.
    token_embd: DeviceBuf<f32>,           // [vocab, hidden]
    output_norm: DeviceBuf<f32>,          // [hidden]
    /// `None` when `tied_embeddings` — `output_proj` reuses `token_embd`.
    output_proj: Option<DeviceBuf<f32>>,  // [vocab, hidden]

    // Per-call activation scratch (persistent across calls; overwritten each call).
    hidden_a: DeviceBuf<f32>,
    hidden_b: DeviceBuf<f32>,
    logits:   DeviceBuf<f32>,

    // Compiled kernel modules — keep alive for the lifetime of self.
    embed_module:   Module,
    rmsnorm_module: Module,
    matvec_module:  Module,

    // Dimensions.
    hidden:  usize,
    vocab:   usize,
    rms_eps: f32,
}

impl GpuQwen35 {
    pub fn new(model: &Qwen35F32Model, cache: &KernelCache) -> Result<Self, String> {
        let cfg = &model.model.config;
        let hidden = cfg.hidden_size as usize;
        let vocab  = cfg.vocab_size  as usize;

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
        let logits   = DeviceBuf::new(vocab)?;

        let embed_hsaco   = cache.compile("embed_lookup", EMBED_LOOKUP_SOURCE)?;
        let rmsnorm_hsaco = cache.compile("rmsnorm",      RMSNORM_SOURCE)?;
        let matvec_hsaco  = cache.compile("matvec",       MATVEC_SOURCE)?;

        Ok(Self {
            token_embd, output_norm, output_proj,
            hidden_a, hidden_b, logits,
            embed_module:   Module::load(&embed_hsaco)?,
            rmsnorm_module: Module::load(&rmsnorm_hsaco)?,
            matvec_module:  Module::load(&matvec_hsaco)?,
            hidden, vocab, rms_eps: cfg.rms_norm_eps,
        })
    }

    fn output_proj_ptr(&self) -> *mut c_void {
        match &self.output_proj {
            Some(buf) => buf.raw_ptr(),
            None      => self.token_embd.raw_ptr(),
        }
    }

    /// embed → output_norm → output_proj. Returns vocab-length logits.
    /// Composition is artificial (norm doesn't belong here in real
    /// forward), but every kernel and every device pointer in the
    /// pipeline is exercised.
    pub fn embed_norm_proj(&self, token: u32) -> Result<Vec<f32>, String> {
        let block: u32 = 256;

        // 1) embed_lookup(token_embd, hidden_a, row=token, n=hidden)
        {
            let f = self.embed_module.function("embed_lookup_f32")?;
            let mut t = self.token_embd.raw_ptr();
            let mut o = self.hidden_a.raw_ptr();
            let mut row = token;
            let mut n = self.hidden as u32;
            let mut args: [*mut c_void; 4] = [
                &mut t   as *mut _ as *mut c_void,
                &mut o   as *mut _ as *mut c_void,
                &mut row as *mut _ as *mut c_void,
                &mut n   as *mut _ as *mut c_void,
            ];
            let grid = (self.hidden as u32 + block - 1) / block;
            unsafe { f.launch((grid, 1, 1), (block, 1, 1), 0, None, &mut args)?; }
        }

        // 2) rmsnorm(hidden_a, output_norm, hidden_b, n=hidden, eps)
        {
            let f = self.rmsnorm_module.function("rmsnorm_f32")?;
            let mut x = self.hidden_a.raw_ptr();
            let mut w = self.output_norm.raw_ptr();
            let mut y = self.hidden_b.raw_ptr();
            let mut n = self.hidden as u32;
            let mut e = self.rms_eps;
            let mut args: [*mut c_void; 5] = [
                &mut x as *mut _ as *mut c_void,
                &mut w as *mut _ as *mut c_void,
                &mut y as *mut _ as *mut c_void,
                &mut n as *mut _ as *mut c_void,
                &mut e as *mut _ as *mut c_void,
            ];
            let smem = block * std::mem::size_of::<f32>() as u32;
            unsafe { f.launch((1, 1, 1), (block, 1, 1), smem, None, &mut args)?; }
        }

        // 3) matvec(output_proj, hidden_b, logits, in=hidden, out=vocab)
        {
            let f = self.matvec_module.function("matvec_f32")?;
            let mut w = self.output_proj_ptr();
            let mut x = self.hidden_b.raw_ptr();
            let mut y = self.logits.raw_ptr();
            let mut in_dim  = self.hidden as u32;
            let mut out_dim = self.vocab  as u32;
            let mut args: [*mut c_void; 5] = [
                &mut w       as *mut _ as *mut c_void,
                &mut x       as *mut _ as *mut c_void,
                &mut y       as *mut _ as *mut c_void,
                &mut in_dim  as *mut _ as *mut c_void,
                &mut out_dim as *mut _ as *mut c_void,
            ];
            let smem = block * std::mem::size_of::<f32>() as u32;
            let grid = self.vocab as u32;
            unsafe { f.launch((grid, 1, 1), (block, 1, 1), smem, None, &mut args)?; }
        }

        hip::Device(0).synchronize()?;
        let mut out = vec![0.0f32; self.vocab];
        self.logits.copy_to_host(&mut out)?;
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
