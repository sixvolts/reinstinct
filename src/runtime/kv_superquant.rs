//! SuperQuant — 3-tier KV cache (fp16 Hot / int8 Warm / turbo3 Cold).
//!
//! Opt-in alternative to the per-layer int8 KV cache used by Gemma 4
//! and the per-layer fp32 cache used by Qwen 3.5. Trades modest decode
//! cost for substantially more context capacity at the same VRAM
//! budget, with precision matched to where attention actually puts its
//! softmax mass:
//!
//! ```text
//! Position  Tier    Format         Capacity   SNR (per-value)
//! 0..       Cold    turbo3         3.5 bpv    ~14.6 dB   (long tail)
//! cold_end..warm_end  Warm   int8 + scale  8 bpv      ~48 dB   (recent few turns)
//! warm_end..pos       Hot    fp16          16 bpv     exact    (current turn)
//! ```
//!
//! Tier sizing is set at construction time:
//! - `hot_cap` — number of fp16 positions (default 2048 for chat)
//! - `warm_cap` — int8 positions (default 8192)
//! - `cold_cap` — turbo3 positions (default = max_seq - hot - warm)
//!
//! ## Write path
//!
//! Every `write_step(src_k, src_v)` writes to the Hot tier. When Hot
//! is full, the oldest entry slides to Warm via the fp16→q8 demotion
//! kernel. When Warm is full, the oldest entry slides to Cold via the
//! q8→turbo3 demotion kernel. Cold overflow is currently treated as
//! "context exhausted" — caller is responsible for not exceeding the
//! sum of capacities.
//!
//! ## Read path (Phase #3)
//!
//! The 3-tier attention kernel reads from all three tiers in a single
//! launch, dequantizing on the fly. Each thread/wavefront handles a
//! position range mapped to one tier; the per-position softmax merges
//! scores across tiers naturally.
//!
//! ## Turn-boundary API
//!
//! `mark_turn_boundary()` is a no-op in the current implementation —
//! the position-sliding-window policy is used uniformly. Future
//! refinement: track per-turn position cuts so chat workloads can
//! demote on turn boundaries instead of single-position increments.

use crate::hip::{DeviceBuf, Module};
use crate::quant::turbo3::{CacheKind, ROT_GROUP};
use super::KernelCache;
use super::kv_turbo3::{slot_bytes, KV_WRITE_TURBO3_SOURCE,
                       launch_promote_fp16_to_q8, launch_promote_q8_to_turbo3};
use std::ffi::c_void;

pub const KV_WRITE_FP16_SOURCE: &str =
    include_str!("../../kernels/kv_write_fp16.cpp");
pub const ATTN_PARTIAL_SUPERQUANT_SOURCE: &str =
    include_str!("../../kernels/attn_partial_superquant.cpp");

/// Launch the SuperQuant 3-tier decode attention kernel.
/// Returns partial (m, l, o) buffers; caller passes these to attn_merge
/// to produce the final per-head output.
///
/// `q`: [n_heads × head_dim] fp32 on device.
/// `head_dim` must be a multiple of ROT_GROUP (128).
#[allow(clippy::too_many_arguments)]
pub fn launch_attn_partial_superquant(
    cache: &KernelCache,
    kv: &SuperQuantKvCache,
    q: *mut c_void,
    o_partial: *mut c_void,
    m_partial: *mut c_void,
    l_partial: *mut c_void,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    scaling: f32,
    n_splits: u32,
) -> Result<(), String>
{
    let hsaco = cache.compile("attn_partial_superquant", ATTN_PARTIAL_SUPERQUANT_SOURCE)?;
    let module = Module::load(&hsaco)?;
    let f = module.function("attn_partial_superquant_f32")?;

    let block: u32 = 256;
    let total_len = (kv.cold_count + kv.warm_count + kv.hot_count) as u32;
    let chunk = (total_len + n_splits - 1) / n_splits;

    // LDS: qf32 + scores + tmp(bs) + dqbuf(head_dim) + fwhtw(ROT_GROUP).
    let smem_floats = head_dim as usize + chunk as usize + block as usize
                    + head_dim as usize + ROT_GROUP;
    let smem_bytes = (smem_floats * std::mem::size_of::<f32>()) as u32;

    let mut q_p   = q;
    let mut hk_p  = kv.hot_k.raw_ptr();
    let mut hv_p  = kv.hot_v.raw_ptr();
    let mut wk_p  = kv.warm_k.raw_ptr();
    let mut wks_p = kv.warm_ks.raw_ptr();
    let mut wv_p  = kv.warm_v.raw_ptr();
    let mut wvs_p = kv.warm_vs.raw_ptr();
    let mut ck_p  = kv.cold_k.raw_ptr();
    let mut cv_p  = kv.cold_v.raw_ptr();
    let mut s1k_p = kv.signs1_k.raw_ptr();
    let mut s2k_p = kv.signs2_k.raw_ptr();
    let mut s1v_p = kv.signs1_v.raw_ptr();
    let mut s2v_p = kv.signs2_v.raw_ptr();
    let mut op_p  = o_partial;
    let mut mp_p  = m_partial;
    let mut lp_p  = l_partial;
    let mut nh    = n_heads;
    let mut nkv   = n_kv_heads;
    let mut hd    = head_dim;
    let mut cc    = kv.cold_count as u32;
    let mut wc    = kv.warm_count as u32;
    let mut hc    = kv.hot_count  as u32;
    let mut sc    = scaling;
    let mut ns    = n_splits;

    let mut args: [*mut c_void; 24] = [
        &mut q_p   as *mut _ as *mut c_void,
        &mut hk_p  as *mut _ as *mut c_void,
        &mut hv_p  as *mut _ as *mut c_void,
        &mut wk_p  as *mut _ as *mut c_void,
        &mut wks_p as *mut _ as *mut c_void,
        &mut wv_p  as *mut _ as *mut c_void,
        &mut wvs_p as *mut _ as *mut c_void,
        &mut ck_p  as *mut _ as *mut c_void,
        &mut cv_p  as *mut _ as *mut c_void,
        &mut s1k_p as *mut _ as *mut c_void,
        &mut s2k_p as *mut _ as *mut c_void,
        &mut s1v_p as *mut _ as *mut c_void,
        &mut s2v_p as *mut _ as *mut c_void,
        &mut op_p  as *mut _ as *mut c_void,
        &mut mp_p  as *mut _ as *mut c_void,
        &mut lp_p  as *mut _ as *mut c_void,
        &mut nh    as *mut _ as *mut c_void,
        &mut nkv   as *mut _ as *mut c_void,
        &mut hd    as *mut _ as *mut c_void,
        &mut cc    as *mut _ as *mut c_void,
        &mut wc    as *mut _ as *mut c_void,
        &mut hc    as *mut _ as *mut c_void,
        &mut sc    as *mut _ as *mut c_void,
        &mut ns    as *mut _ as *mut c_void,
    ];
    let grid = (n_heads, n_splits, 1);
    unsafe { f.launch(grid, (block, 1, 1), smem_bytes, None, &mut args)?; }
    Ok(())
}

/// Per-tier capacity configuration. All in token positions; the actual
/// memory cost scales as n_kv × head_dim × bpv × cap.
#[derive(Clone, Copy, Debug)]
pub struct SuperQuantConfig {
    pub hot_cap:  usize,    // fp16 positions
    pub warm_cap: usize,    // int8 positions
    pub cold_cap: usize,    // turbo3 positions
}

impl SuperQuantConfig {
    /// Defaults for chat-style workloads — Hot covers current turn,
    /// Warm covers prior turns up to ~8K tokens, Cold covers the rest.
    pub fn chat_defaults(max_seq: usize) -> Self {
        let hot  = 2048.min(max_seq);
        let warm = 8192.min(max_seq.saturating_sub(hot));
        let cold = max_seq.saturating_sub(hot + warm);
        Self { hot_cap: hot, warm_cap: warm, cold_cap: cold }
    }

    pub fn total(&self) -> usize { self.hot_cap + self.warm_cap + self.cold_cap }
}

/// Per-layer 3-tier KV cache.
pub struct SuperQuantKvCache {
    // Hot tier (fp16)
    pub hot_k:  DeviceBuf<u16>,   // [hot_cap, n_kv * head_dim] fp16 bits
    pub hot_v:  DeviceBuf<u16>,
    // Warm tier (int8 + per-(slot,head) scale)
    pub warm_k:  DeviceBuf<i8>,
    pub warm_v:  DeviceBuf<i8>,
    pub warm_ks: DeviceBuf<f32>,
    pub warm_vs: DeviceBuf<f32>,
    // Cold tier (turbo3 packed)
    pub cold_k: DeviceBuf<u8>,
    pub cold_v: DeviceBuf<u8>,
    // Sign masks (resident)
    pub signs1_k: DeviceBuf<i8>,
    pub signs2_k: DeviceBuf<i8>,
    pub signs1_v: DeviceBuf<i8>,
    pub signs2_v: DeviceBuf<i8>,

    // Scratch buffers for slide-after-demote (avoids in-place memcpy
    // with overlapping regions, which is undefined behavior).
    scratch_hot:    DeviceBuf<u16>,
    scratch_warm_q: DeviceBuf<i8>,
    scratch_warm_s: DeviceBuf<f32>,

    pub n_kv: usize,
    pub head_dim: usize,
    pub config: SuperQuantConfig,

    // Tier counts — number of populated positions in each tier.
    pub hot_count:  usize,
    pub warm_count: usize,
    pub cold_count: usize,

    // Cached HIP modules (lazy compile).
    write_hot_module:  Option<Module>,
    write_turbo3_module: Option<Module>,
}

impl SuperQuantKvCache {
    pub fn new(n_kv: usize, head_dim: usize, config: SuperQuantConfig)
        -> Result<Self, String>
    {
        assert_eq!(head_dim % ROT_GROUP, 0,
            "SuperQuant head_dim {head_dim} must be a multiple of {ROT_GROUP}");
        let sb = slot_bytes(head_dim);
        Ok(Self {
            hot_k:  DeviceBuf::new(config.hot_cap * n_kv * head_dim)?,
            hot_v:  DeviceBuf::new(config.hot_cap * n_kv * head_dim)?,
            warm_k: DeviceBuf::new(config.warm_cap * n_kv * head_dim)?,
            warm_v: DeviceBuf::new(config.warm_cap * n_kv * head_dim)?,
            warm_ks: DeviceBuf::new(config.warm_cap * n_kv)?,
            warm_vs: DeviceBuf::new(config.warm_cap * n_kv)?,
            cold_k: DeviceBuf::new(config.cold_cap * n_kv * sb)?,
            cold_v: DeviceBuf::new(config.cold_cap * n_kv * sb)?,
            scratch_hot:    DeviceBuf::new(config.hot_cap * n_kv * head_dim)?,
            scratch_warm_q: DeviceBuf::new(config.warm_cap * n_kv * head_dim)?,
            scratch_warm_s: DeviceBuf::new(config.warm_cap * n_kv)?,
            signs1_k: DeviceBuf::from_slice(CacheKind::K.signs1())?,
            signs2_k: DeviceBuf::from_slice(CacheKind::K.signs2())?,
            signs1_v: DeviceBuf::from_slice(CacheKind::V.signs1())?,
            signs2_v: DeviceBuf::from_slice(CacheKind::V.signs2())?,
            n_kv, head_dim, config,
            hot_count: 0, warm_count: 0, cold_count: 0,
            write_hot_module: None,
            write_turbo3_module: None,
        })
    }

    pub fn reset(&mut self) {
        self.hot_count = 0;
        self.warm_count = 0;
        self.cold_count = 0;
    }

    /// Total populated positions across all three tiers.
    pub fn len(&self) -> usize {
        self.cold_count + self.warm_count + self.hot_count
    }

    /// Capacity in positions.
    pub fn max_seq(&self) -> usize { self.config.total() }

    /// Lazy module load.
    fn ensure_write_hot(&mut self, cache: &KernelCache) -> Result<(), String> {
        if self.write_hot_module.is_none() {
            let hsaco = cache.compile("kv_write_fp16", KV_WRITE_FP16_SOURCE)?;
            self.write_hot_module = Some(Module::load(&hsaco)?);
        }
        Ok(())
    }
    fn ensure_write_turbo3(&mut self, cache: &KernelCache) -> Result<(), String> {
        if self.write_turbo3_module.is_none() {
            let hsaco = cache.compile("kv_write_turbo3", KV_WRITE_TURBO3_SOURCE)?;
            self.write_turbo3_module = Some(Module::load(&hsaco)?);
        }
        Ok(())
    }

    /// Write one decode-step's K + V projections (fp32 device buffers,
    /// [n_kv * head_dim] each) into the Hot tier. Demotes oldest Hot →
    /// Warm if Hot is full, and Warm → Cold if Warm is full.
    pub fn write_step(&mut self, cache: &KernelCache,
                      src_k: *mut c_void, src_v: *mut c_void)
        -> Result<(), String>
    {
        // If hot is full, demote the oldest hot entry to warm.
        if self.hot_count >= self.config.hot_cap {
            self.demote_hot_to_warm(cache, 1)?;
        }
        // Hot has room — write fp16 to slot `hot_count`.
        self.ensure_write_hot(cache)?;
        let module = self.write_hot_module.as_ref().unwrap();
        let f = module.function("kv_write_fp16_step_f32")?;
        let pos = self.hot_count as u32;
        for (src, dst_buf) in [(src_k, self.hot_k.raw_ptr()),
                                (src_v, self.hot_v.raw_ptr())] {
            let mut src_p = src;
            let mut dst_p = dst_buf;
            let mut nkv = self.n_kv as u32;
            let mut hd  = self.head_dim as u32;
            let mut p   = pos;
            let mut ms  = self.config.hot_cap as u32;
            let mut args: [*mut c_void; 6] = [
                &mut src_p as *mut _ as *mut c_void,
                &mut dst_p as *mut _ as *mut c_void,
                &mut nkv   as *mut _ as *mut c_void,
                &mut hd    as *mut _ as *mut c_void,
                &mut p     as *mut _ as *mut c_void,
                &mut ms    as *mut _ as *mut c_void,
            ];
            let block = (self.head_dim as u32).min(1024);
            unsafe { f.launch((self.n_kv as u32, 1, 1), (block, 1, 1), 0, None, &mut args)?; }
        }
        self.hot_count += 1;
        Ok(())
    }

    /// Move `n` oldest Hot entries to the Warm tier. If Warm is full,
    /// first cascades the oldest Warm → Cold. `n` is typically 1
    /// (single-step decode); larger batches arise on turn-boundary
    /// demotion in the chat path.
    pub fn demote_hot_to_warm(&mut self, cache: &KernelCache, n: usize)
        -> Result<(), String>
    {
        assert!(n <= self.hot_count, "demote_hot_to_warm: n={n} > hot_count={}", self.hot_count);
        // Cascade if warm overflows.
        let overflow = (self.warm_count + n).saturating_sub(self.config.warm_cap);
        if overflow > 0 {
            self.demote_warm_to_cold(cache, overflow)?;
        }

        // Compute byte offsets:
        //   hot[0..n] (fp16, n × n_kv × head_dim u16s) → warm[warm_count..warm_count+n]
        let row_elems = self.n_kv * self.head_dim;
        let src_off_u16 = 0usize;
        let dst_off_i8  = self.warm_count * row_elems;
        let dst_off_f32 = self.warm_count * self.n_kv;

        let src_k_ptr = unsafe { self.hot_k.raw_ptr().add(src_off_u16 * 2) };  // u16=2B
        let src_v_ptr = unsafe { self.hot_v.raw_ptr().add(src_off_u16 * 2) };
        let dst_k_ptr = unsafe { self.warm_k.raw_ptr().add(dst_off_i8) };
        let dst_v_ptr = unsafe { self.warm_v.raw_ptr().add(dst_off_i8) };
        let dst_ks_ptr = unsafe { self.warm_ks.raw_ptr().add(dst_off_f32 * 4) }; // f32=4B
        let dst_vs_ptr = unsafe { self.warm_vs.raw_ptr().add(dst_off_f32 * 4) };

        launch_promote_fp16_to_q8(cache, src_k_ptr, dst_k_ptr, dst_ks_ptr,
            n as u32, self.n_kv as u32, self.head_dim as u32)?;
        launch_promote_fp16_to_q8(cache, src_v_ptr, dst_v_ptr, dst_vs_ptr,
            n as u32, self.n_kv as u32, self.head_dim as u32)?;
        crate::hip::Device(0).synchronize()?;

        // Slide hot data forward by n positions: copy [n..hot_count] →
        // scratch → [0..hot_count-n]. Two D2D memcpys avoid the
        // undefined behavior of self-overlapping single-memcpy.
        if self.hot_count - n > 0 {
            let remaining = (self.hot_count - n) * row_elems;
            self.scratch_hot.copy_range_from_device(&self.hot_k, n * row_elems, 0, remaining)?;
            self.hot_k.copy_range_from_device(&self.scratch_hot, 0, 0, remaining)?;
            self.scratch_hot.copy_range_from_device(&self.hot_v, n * row_elems, 0, remaining)?;
            self.hot_v.copy_range_from_device(&self.scratch_hot, 0, 0, remaining)?;
        }
        self.hot_count  -= n;
        self.warm_count += n;
        Ok(())
    }

    /// Move `n` oldest Warm entries to the Cold tier.
    pub fn demote_warm_to_cold(&mut self, cache: &KernelCache, n: usize)
        -> Result<(), String>
    {
        assert!(n <= self.warm_count, "demote_warm_to_cold: n={n} > warm_count={}", self.warm_count);
        assert!(self.cold_count + n <= self.config.cold_cap,
            "cold tier overflow: cold_count={} + n={n} > cold_cap={}",
            self.cold_count, self.config.cold_cap);

        let sb = slot_bytes(self.head_dim);
        let row_elems = self.n_kv * self.head_dim;

        // Source warm[0..n], dest cold[cold_count..cold_count+n].
        let src_k_ptr = self.warm_k.raw_ptr();
        let src_v_ptr = self.warm_v.raw_ptr();
        let src_ks_ptr = self.warm_ks.raw_ptr();
        let src_vs_ptr = self.warm_vs.raw_ptr();
        let dst_off_bytes = self.cold_count * self.n_kv * sb;
        let dst_k_ptr = unsafe { self.cold_k.raw_ptr().add(dst_off_bytes) };
        let dst_v_ptr = unsafe { self.cold_v.raw_ptr().add(dst_off_bytes) };

        launch_promote_q8_to_turbo3(cache, src_k_ptr, src_ks_ptr,
            self.signs1_k.raw_ptr(), self.signs2_k.raw_ptr(),
            dst_k_ptr, n as u32, self.n_kv as u32, self.head_dim as u32)?;
        launch_promote_q8_to_turbo3(cache, src_v_ptr, src_vs_ptr,
            self.signs1_v.raw_ptr(), self.signs2_v.raw_ptr(),
            dst_v_ptr, n as u32, self.n_kv as u32, self.head_dim as u32)?;
        crate::hip::Device(0).synchronize()?;

        // Slide warm forward by n (data + scales).
        if self.warm_count - n > 0 {
            let remaining = (self.warm_count - n) * row_elems;
            let remaining_scales = (self.warm_count - n) * self.n_kv;
            // K + ks
            self.scratch_warm_q.copy_range_from_device(&self.warm_k, n * row_elems, 0, remaining)?;
            self.warm_k.copy_range_from_device(&self.scratch_warm_q, 0, 0, remaining)?;
            self.scratch_warm_s.copy_range_from_device(&self.warm_ks, n * self.n_kv, 0, remaining_scales)?;
            self.warm_ks.copy_range_from_device(&self.scratch_warm_s, 0, 0, remaining_scales)?;
            // V + vs
            self.scratch_warm_q.copy_range_from_device(&self.warm_v, n * row_elems, 0, remaining)?;
            self.warm_v.copy_range_from_device(&self.scratch_warm_q, 0, 0, remaining)?;
            self.scratch_warm_s.copy_range_from_device(&self.warm_vs, n * self.n_kv, 0, remaining_scales)?;
            self.warm_vs.copy_range_from_device(&self.scratch_warm_s, 0, 0, remaining_scales)?;
        }
        self.warm_count -= n;
        self.cold_count += n;
        Ok(())
    }

    /// No-op in current implementation. Reserved for future
    /// per-turn-boundary demotion policy.
    pub fn mark_turn_boundary(&mut self, _cache: &KernelCache) -> Result<(), String> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hip;

    fn skip_if_no_gpu() -> Option<KernelCache> {
        if hip::device_count().ok().unwrap_or(0) < 1 {
            eprintln!("skip: no HIP device"); return None;
        }
        let _ = hip::Device::set(0).ok()?;
        KernelCache::new().ok()
    }

    fn rng(seed: u64) -> impl FnMut() -> f32 {
        let mut s = seed;
        move || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let bits = ((s >> 33) as u32 & 0x007F_FFFF) | 0x3f80_0000;
            (f32::from_bits(bits) - 1.5) * 0.4
        }
    }

    #[test]
    fn config_defaults_sane() {
        let c = SuperQuantConfig::chat_defaults(32768);
        assert_eq!(c.hot_cap, 2048);
        assert_eq!(c.warm_cap, 8192);
        assert_eq!(c.cold_cap, 32768 - 2048 - 8192);
        assert_eq!(c.total(), 32768);
    }

    #[test]
    fn config_handles_small_max_seq() {
        let c = SuperQuantConfig::chat_defaults(1024);
        assert_eq!(c.hot_cap, 1024);
        assert_eq!(c.warm_cap, 0);
        assert_eq!(c.cold_cap, 0);
    }

    #[test]
    fn allocates_and_resets() {
        let Some(_cache) = skip_if_no_gpu() else { return };
        let cfg = SuperQuantConfig { hot_cap: 32, warm_cap: 64, cold_cap: 128 };
        let mut kv = SuperQuantKvCache::new(2, 256, cfg).expect("alloc");
        assert_eq!(kv.len(), 0);
        assert_eq!(kv.max_seq(), 32 + 64 + 128);
        kv.hot_count = 5; kv.warm_count = 10; kv.cold_count = 50;
        assert_eq!(kv.len(), 65);
        kv.reset();
        assert_eq!(kv.len(), 0);
    }

    /// SuperQuant 3-tier decode attention vs a pure fp32 reference
    /// computed on the original (pre-quantization) K/V tensors. Output
    /// is not bit-identical due to tier quantization noise, but
    /// relative L2 error per head should land within a few %.
    #[test]
    fn three_tier_attention_matches_fp32_reference_within_quant_noise() {
        let Some(cache) = skip_if_no_gpu() else { return };

        // Small cache. Hot=2, Warm=2, Cold=4 — total 8 positions.
        let cfg = SuperQuantConfig { hot_cap: 2, warm_cap: 2, cold_cap: 4 };
        let n_kv = 1;          // single KV head for simplicity
        let n_heads = 1;       // single Q head
        let head_dim = 128;
        let mut kv = SuperQuantKvCache::new(n_kv, head_dim, cfg).expect("alloc");

        // Generate 8 K/V tensors + the Q to attend with.
        let row_elems = n_kv * head_dim;
        let mut r = rng(0xABBA);
        let mut ref_k: Vec<Vec<f32>> = Vec::with_capacity(8);
        let mut ref_v: Vec<Vec<f32>> = Vec::with_capacity(8);
        for _ in 0..8 {
            let k: Vec<f32> = (0..row_elems).map(|_| r()).collect();
            let v: Vec<f32> = (0..row_elems).map(|_| r()).collect();
            let dk = DeviceBuf::<f32>::from_slice(&k).unwrap();
            let dv = DeviceBuf::<f32>::from_slice(&v).unwrap();
            kv.write_step(&cache, dk.raw_ptr(), dv.raw_ptr()).expect("write_step");
            ref_k.push(k); ref_v.push(v);
        }
        // After 8 writes with caps (2,2,4): 4 cold, 2 warm, 2 hot.
        assert_eq!(kv.cold_count, 4);
        assert_eq!(kv.warm_count, 2);
        assert_eq!(kv.hot_count,  2);

        // Query: random fp32.
        let q: Vec<f32> = (0..n_heads * head_dim).map(|_| r()).collect();
        let dq = DeviceBuf::<f32>::from_slice(&q).unwrap();

        // GPU attention.
        let n_splits = 1u32;
        let do_part = DeviceBuf::<f32>::new(n_heads * (n_splits as usize) * head_dim).unwrap();
        let dm_part = DeviceBuf::<f32>::new(n_heads * (n_splits as usize)).unwrap();
        let dl_part = DeviceBuf::<f32>::new(n_heads * (n_splits as usize)).unwrap();
        let scaling = 1.0 / (head_dim as f32).sqrt();
        launch_attn_partial_superquant(&cache, &kv, dq.raw_ptr(),
            do_part.raw_ptr(), dm_part.raw_ptr(), dl_part.raw_ptr(),
            n_heads as u32, n_kv as u32, head_dim as u32, scaling, n_splits)
            .expect("attn launch");
        hip::Device(0).synchronize().unwrap();

        let mut o_part = vec![0.0f32; n_heads * (n_splits as usize) * head_dim];
        let mut m_part = vec![0.0f32; n_heads * (n_splits as usize)];
        let mut l_part = vec![0.0f32; n_heads * (n_splits as usize)];
        do_part.copy_to_host(&mut o_part).unwrap();
        dm_part.copy_to_host(&mut m_part).unwrap();
        dl_part.copy_to_host(&mut l_part).unwrap();
        // With n_splits=1 the partial IS the output, just needs / l.
        let mut gpu_out = vec![0.0f32; n_heads * head_dim];
        for d in 0..head_dim { gpu_out[d] = o_part[d] / l_part[0]; }

        // CPU reference: standard scaled-dot-product attention on
        // the original (un-quantized) K/V.
        let mut scores = vec![0.0f32; 8];
        for i in 0..8 {
            let mut s = 0.0f32;
            for d in 0..head_dim { s += q[d] * ref_k[i][d]; }
            scores[i] = s * scaling;
        }
        let m = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mut sum = 0.0f32;
        let mut e = vec![0.0f32; 8];
        for i in 0..8 { e[i] = (scores[i] - m).exp(); sum += e[i]; }
        for i in 0..8 { e[i] /= sum; }
        let mut cpu_out = vec![0.0f32; head_dim];
        for d in 0..head_dim {
            let mut a = 0.0f32;
            for i in 0..8 { a += e[i] * ref_v[i][d]; }
            cpu_out[d] = a;
        }

        // Per-head L2 relative error.
        let mut s_sig = 0.0f64;
        let mut s_err = 0.0f64;
        for d in 0..head_dim {
            s_sig += (cpu_out[d] as f64).powi(2);
            s_err += ((cpu_out[d] - gpu_out[d]) as f64).powi(2);
        }
        let rel_l2 = (s_err.sqrt() / s_sig.sqrt()) as f32;
        eprintln!("SuperQuant attention rel_l2 vs fp32 reference: {rel_l2:.4}");
        // Tier mix: 4 cold (14 dB), 2 warm (48 dB), 2 hot (exact).
        // Worst-case dominates → cold tier's 14 dB SNR contributes
        // roughly ±0.2 relative error per cold position, dampened by
        // softmax weight on each. Empirically ~5-15% L2 error.
        assert!(rel_l2 < 0.30,
                "SuperQuant attention diverges too far: rel_l2={rel_l2:.4}");
    }

    /// End-to-end: fill hot beyond capacity, verify demotion fires,
    /// verify warm fills then cascades to cold.
    #[test]
    fn write_step_cascades_through_tiers() {
        let Some(cache) = skip_if_no_gpu() else { return };
        let cfg = SuperQuantConfig { hot_cap: 4, warm_cap: 4, cold_cap: 8 };
        let n_kv = 2;
        let head_dim = 128;
        let mut kv = SuperQuantKvCache::new(n_kv, head_dim, cfg).expect("alloc");

        // 12 writes — should land 4 in hot, 4 in warm, 4 in cold.
        let mut r = rng(0x5555_AAAA);
        let row_elems = n_kv * head_dim;
        for _i in 0..12 {
            let k: Vec<f32> = (0..row_elems).map(|_| r()).collect();
            let v: Vec<f32> = (0..row_elems).map(|_| r()).collect();
            let dk = DeviceBuf::<f32>::from_slice(&k).unwrap();
            let dv = DeviceBuf::<f32>::from_slice(&v).unwrap();
            kv.write_step(&cache, dk.raw_ptr(), dv.raw_ptr()).expect("write_step");
        }

        assert_eq!(kv.hot_count,  4);
        assert_eq!(kv.warm_count, 4);
        assert_eq!(kv.cold_count, 4);
        assert_eq!(kv.len(), 12);

        // Write one more — should overflow hot → warm, which overflows
        // warm → cold (cold has 4 free slots).
        let k: Vec<f32> = (0..row_elems).map(|_| r()).collect();
        let v: Vec<f32> = (0..row_elems).map(|_| r()).collect();
        let dk = DeviceBuf::<f32>::from_slice(&k).unwrap();
        let dv = DeviceBuf::<f32>::from_slice(&v).unwrap();
        kv.write_step(&cache, dk.raw_ptr(), dv.raw_ptr()).expect("write_step #13");
        assert_eq!(kv.hot_count,  4);
        assert_eq!(kv.warm_count, 4);
        assert_eq!(kv.cold_count, 5);
        assert_eq!(kv.len(), 13);
    }
}
