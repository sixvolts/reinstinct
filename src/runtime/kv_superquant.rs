//! SuperQuant — 2-tier KV cache (int8 Warm / turbo3 Cold).
//!
//! Opt-in alternative to the standard per-layer int8 KV cache. Writes
//! land in the Warm tier (int8, ~48 dB SNR per-value); when Warm is
//! full, the oldest entries demote to Cold (turbo3, ~14.6 dB SNR per-
//! value, 2× more compressed). For long-context workloads this gives
//! ~2× more context capacity than pure int8 with the precision drop
//! confined to the oldest portion of the cache.
//!
//! Layout:
//!
//! ```text
//! Position    Tier   Format         Capacity   SNR
//! 0..cold_count       Cold  turbo3        3.5 bpv    ~14.6 dB
//! cold_count..len     Warm  int8 + scale  8 bpv      ~48 dB
//! ```
//!
//! Tier sizing:
//! - `warm_cap` — int8 positions (default 8192 — covers typical
//!   short/mid-context decode at full precision).
//! - `cold_cap` — turbo3 positions (default max_seq - warm_cap).
//!
//! ## Design history
//!
//! The original 3-tier design (fp16 Hot / int8 Warm / turbo3 Cold) was
//! simplified to 2-tier per user feedback (2026-05-25): int8's 48 dB
//! is already plenty for any attention tier; the fp16 Hot tier added
//! complexity (separate write kernel + dequant path) for invisible
//! quality gain. Kept the Warm/Cold names and the existing turbo3 +
//! cold-demote pipeline; removed the Hot tier and its dedicated kernels.

use crate::hip::{DeviceBuf, Module};
use crate::quant::turbo3::{CacheKind, ROT_GROUP};
use super::KernelCache;
use super::kv_turbo3::{slot_bytes, launch_promote_q8_to_turbo3};
use std::ffi::c_void;

pub const KV_WRITE_Q8_STEP_SOURCE: &str =
    include_str!("../../kernels/kv_write_q8_step.cpp");
pub const ATTN_PARTIAL_SUPERQUANT_SOURCE: &str =
    include_str!("../../kernels/attn_partial_superquant.cpp");

/// Per-tier capacity (token positions).
#[derive(Clone, Copy, Debug)]
pub struct SuperQuantConfig {
    pub warm_cap: usize,    // int8 positions
    pub cold_cap: usize,    // turbo3 positions
}

impl SuperQuantConfig {
    /// Defaults for chat-style workloads — 8K int8 Warm + remainder Cold.
    pub fn chat_defaults(max_seq: usize) -> Self {
        let warm = 8192.min(max_seq);
        let cold = max_seq.saturating_sub(warm);
        Self { warm_cap: warm, cold_cap: cold }
    }

    pub fn total(&self) -> usize { self.warm_cap + self.cold_cap }
}

/// Per-layer 2-tier KV cache.
pub struct SuperQuantKvCache {
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
    // Scratch for slide-after-demote (avoids overlapping memcpy UB).
    scratch_warm_q: DeviceBuf<i8>,
    scratch_warm_s: DeviceBuf<f32>,

    pub n_kv: usize,
    pub head_dim: usize,
    pub config: SuperQuantConfig,

    // Tier counts — populated positions in each tier.
    pub warm_count: usize,
    pub cold_count: usize,

    write_q8_module: Option<Module>,
}

impl SuperQuantKvCache {
    pub fn new(n_kv: usize, head_dim: usize, config: SuperQuantConfig)
        -> Result<Self, String>
    {
        assert_eq!(head_dim % ROT_GROUP, 0,
            "SuperQuant head_dim {head_dim} must be a multiple of {ROT_GROUP}");
        let sb = slot_bytes(head_dim);
        Ok(Self {
            warm_k: DeviceBuf::new(config.warm_cap * n_kv * head_dim)?,
            warm_v: DeviceBuf::new(config.warm_cap * n_kv * head_dim)?,
            warm_ks: DeviceBuf::new(config.warm_cap * n_kv)?,
            warm_vs: DeviceBuf::new(config.warm_cap * n_kv)?,
            cold_k: DeviceBuf::new(config.cold_cap * n_kv * sb)?,
            cold_v: DeviceBuf::new(config.cold_cap * n_kv * sb)?,
            signs1_k: DeviceBuf::from_slice(CacheKind::K.signs1())?,
            signs2_k: DeviceBuf::from_slice(CacheKind::K.signs2())?,
            signs1_v: DeviceBuf::from_slice(CacheKind::V.signs1())?,
            signs2_v: DeviceBuf::from_slice(CacheKind::V.signs2())?,
            scratch_warm_q: DeviceBuf::new(config.warm_cap * n_kv * head_dim)?,
            scratch_warm_s: DeviceBuf::new(config.warm_cap * n_kv)?,
            n_kv, head_dim, config,
            warm_count: 0, cold_count: 0,
            write_q8_module: None,
        })
    }

    pub fn reset(&mut self) {
        self.warm_count = 0;
        self.cold_count = 0;
    }

    pub fn len(&self) -> usize { self.cold_count + self.warm_count }
    pub fn max_seq(&self) -> usize { self.config.total() }

    fn ensure_write_q8(&mut self, cache: &KernelCache) -> Result<(), String> {
        if self.write_q8_module.is_none() {
            let hsaco = cache.compile("kv_write_q8_step", KV_WRITE_Q8_STEP_SOURCE)?;
            self.write_q8_module = Some(Module::load(&hsaco)?);
        }
        Ok(())
    }

    /// Write one decode-step's K + V projections (fp32 device buffers,
    /// [n_kv * head_dim] each) into the Warm tier. Cascades the oldest
    /// Warm entries to Cold if Warm is full.
    pub fn write_step(&mut self, cache: &KernelCache,
                      src_k: *mut c_void, src_v: *mut c_void)
        -> Result<(), String>
    {
        if self.warm_count >= self.config.warm_cap {
            self.demote_warm_to_cold(cache, 1)?;
        }
        self.ensure_write_q8(cache)?;
        let nkv = self.n_kv as u32;
        let hd  = self.head_dim as u32;
        let pos = self.warm_count as u32;
        let ms  = self.config.warm_cap as u32;
        let module = self.write_q8_module.as_ref().unwrap();
        let f = module.function("kv_write_q8_step_f32")?;

        for (src, dst_q, dst_s) in [
            (src_k, self.warm_k.raw_ptr(), self.warm_ks.raw_ptr()),
            (src_v, self.warm_v.raw_ptr(), self.warm_vs.raw_ptr()),
        ] {
            let mut src_p = src;
            let mut dq_p  = dst_q;
            let mut ds_p  = dst_s;
            let mut nkv_a = nkv;
            let mut hd_a  = hd;
            let mut p     = pos;
            let mut ms_a  = ms;
            let mut args: [*mut c_void; 7] = [
                &mut src_p as *mut _ as *mut c_void,
                &mut dq_p  as *mut _ as *mut c_void,
                &mut ds_p  as *mut _ as *mut c_void,
                &mut nkv_a as *mut _ as *mut c_void,
                &mut hd_a  as *mut _ as *mut c_void,
                &mut p     as *mut _ as *mut c_void,
                &mut ms_a  as *mut _ as *mut c_void,
            ];
            unsafe { f.launch((nkv, 1, 1), (256, 1, 1), 0, None, &mut args)?; }
        }
        self.warm_count += 1;
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

        let src_k_ptr  = self.warm_k.raw_ptr();
        let src_v_ptr  = self.warm_v.raw_ptr();
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

        // Slide warm forward by n (data + scales). Two D2D memcpys via
        // per-cache scratch to avoid overlapping-source UB.
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

    /// No-op in current implementation. Reserved for future per-turn
    /// demotion policy. Kept as part of the API so chat-template code
    /// can call it unconditionally.
    pub fn mark_turn_boundary(&mut self, _cache: &KernelCache) -> Result<(), String> {
        Ok(())
    }
}

/// Launch the SuperQuant 2-tier decode attention kernel. Returns
/// partial (m, l, o) buffers; caller merges via standard attn_merge
/// stable-softmax combine.
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
    let total_len = (kv.cold_count + kv.warm_count) as u32;
    let chunk = (total_len + n_splits - 1) / n_splits.max(1);

    // LDS: qf32 + scores + tmp(bs) + dqbuf(head_dim) + acc_v(head_dim)
    //    + dq_group(ROT_GROUP) + fwhtw(ROT_GROUP).
    let smem_floats = head_dim as usize + chunk as usize + block as usize
                    + head_dim as usize     // dqbuf (K dequant)
                    + head_dim as usize     // acc_v
                    + ROT_GROUP             // dq_group (V per-group dequant)
                    + ROT_GROUP;            // fwhtw
    let smem_bytes = (smem_floats * std::mem::size_of::<f32>()) as u32;

    let mut q_p   = q;
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
    let mut sc    = scaling;
    let mut ns    = n_splits;

    let mut args: [*mut c_void; 20] = [
        &mut q_p   as *mut _ as *mut c_void,
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
        &mut sc    as *mut _ as *mut c_void,
    ];
    let grid = (n_heads, n_splits, 1);
    unsafe { f.launch(grid, (block, 1, 1), smem_bytes, None, &mut args)?; }
    Ok(())
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
        assert_eq!(c.warm_cap, 8192);
        assert_eq!(c.cold_cap, 32768 - 8192);
        assert_eq!(c.total(), 32768);
    }

    #[test]
    fn allocates_and_resets() {
        let Some(_cache) = skip_if_no_gpu() else { return };
        let cfg = SuperQuantConfig { warm_cap: 64, cold_cap: 128 };
        let mut kv = SuperQuantKvCache::new(2, 256, cfg).expect("alloc");
        assert_eq!(kv.len(), 0);
        kv.warm_count = 10; kv.cold_count = 50;
        assert_eq!(kv.len(), 60);
        kv.reset();
        assert_eq!(kv.len(), 0);
    }

    #[test]
    fn write_step_cascades_warm_to_cold() {
        let Some(cache) = skip_if_no_gpu() else { return };
        let cfg = SuperQuantConfig { warm_cap: 4, cold_cap: 8 };
        let n_kv = 2;
        let head_dim = 128;
        let mut kv = SuperQuantKvCache::new(n_kv, head_dim, cfg).expect("alloc");

        let row_elems = n_kv * head_dim;
        let mut r = rng(0x9876);
        for _ in 0..6 {
            let k: Vec<f32> = (0..row_elems).map(|_| r()).collect();
            let v: Vec<f32> = (0..row_elems).map(|_| r()).collect();
            let dk = DeviceBuf::<f32>::from_slice(&k).unwrap();
            let dv = DeviceBuf::<f32>::from_slice(&v).unwrap();
            kv.write_step(&cache, dk.raw_ptr(), dv.raw_ptr()).expect("write_step");
        }
        // After 6 writes with warm_cap=4: warm=4, cold=2.
        assert_eq!(kv.warm_count, 4);
        assert_eq!(kv.cold_count, 2);
        assert_eq!(kv.len(), 6);
    }

    /// 2-tier attention vs pure fp32 reference. Should land tighter
    /// than the 3-tier version (no Hot vs fp32 mismatch — Warm int8
    /// is much closer to fp32 than turbo3).
    #[test]
    fn two_tier_attention_matches_fp32_reference_within_quant_noise() {
        let Some(cache) = skip_if_no_gpu() else { return };
        let cfg = SuperQuantConfig { warm_cap: 4, cold_cap: 4 };
        let n_kv = 1;
        let n_heads = 1;
        let head_dim = 128;
        let mut kv = SuperQuantKvCache::new(n_kv, head_dim, cfg).expect("alloc");

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
        assert_eq!(kv.cold_count, 4);
        assert_eq!(kv.warm_count, 4);

        let q: Vec<f32> = (0..n_heads * head_dim).map(|_| r()).collect();
        let dq = DeviceBuf::<f32>::from_slice(&q).unwrap();

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
        let mut gpu_out = vec![0.0f32; n_heads * head_dim];
        for d in 0..head_dim { gpu_out[d] = o_part[d] / l_part[0]; }

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

        let mut s_sig = 0.0f64;
        let mut s_err = 0.0f64;
        for d in 0..head_dim {
            s_sig += (cpu_out[d] as f64).powi(2);
            s_err += ((cpu_out[d] - gpu_out[d]) as f64).powi(2);
        }
        let rel_l2 = (s_err.sqrt() / s_sig.sqrt()) as f32;
        eprintln!("SuperQuant 2-tier attention rel_l2: {rel_l2:.4}");
        assert!(rel_l2 < 0.30,
                "SuperQuant attention diverges too far: rel_l2={rel_l2:.4}");
    }
}
