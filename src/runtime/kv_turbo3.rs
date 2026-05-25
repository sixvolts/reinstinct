//! Turbo3 KV cache — 3.5 bits/value packed K + V storage.
//!
//! Parallel to `Gemma4KvCache` (int8, 8 bpv) and the qwen35 `GpuKvCache`
//! (f32, 32 bpv). Trade-off: ~4.6× capacity vs fp16 + ~2× vs int8, with
//! a 30 dB SNR drop per-value (see `crate::quant::turbo3` for the full
//! analysis and the L2-preserving correction that keeps attention-score
//! magnitudes intact on average).
//!
//! **Phase 2a status:** storage + decode-step write kernel + GPU
//! round-trip oracle are wired. The matching attention-decode kernel
//! that consumes this cache is Phase 2b (TODO — see MANUAL.md or
//! `project_turbo3_kv` memory note).
//!
//! ## Layout (per cache buffer, K or V)
//!
//! ```text
//! [max_seq, n_kv, head_dim/32, 16]  bytes
//! └ slot ─┘ └head┘ └─ rotation groups ─┘ └─ block ─┘
//! ```
//!
//! For head_dim=256, n_kv=2, max_seq=8192:
//!   8192 × 2 × 8 × 16 = 2.0 MB per layer
//!
//! For comparison, int8 KV for the same shape:
//!   8192 × 2 × 256 = 4.0 MB (qs) + 8192 × 2 × 4 = 64 KB (scales) = 4.06 MB
//!
//! → 2.0× compression vs int8, 4.6× vs fp16.

use crate::quant::turbo3::{BLOCKS_PER_GROUP, BYTES_PER_BLOCK, CacheKind, ROT_GROUP};
use crate::hip::{DeviceBuf, Module};
#[cfg(test)] use crate::hip;
use super::KernelCache;
use std::ffi::c_void;

pub const KV_WRITE_TURBO3_SOURCE: &str =
    include_str!("../../kernels/kv_write_turbo3.cpp");
pub const KV_PROMOTE_FP16_TO_Q8_SOURCE: &str =
    include_str!("../../kernels/kv_promote_fp16_to_q8.cpp");
pub const KV_PROMOTE_Q8_TO_TURBO3_SOURCE: &str =
    include_str!("../../kernels/kv_promote_q8_to_turbo3.cpp");

/// Demote a contiguous range of fp16 K (or V) slots to int8 + per-(slot,
/// head) scale. Caller supplies the source fp16 device buffer (the Hot
/// tier's K or V slot range) and destination int8 + scale buffers (the
/// Warm tier's K or V append region).
///
/// `n_demote` is the number of (slot) positions to promote, `n_kv` is
/// the head count, `head_dim` the per-head dimension.
pub fn launch_promote_fp16_to_q8(cache: &KernelCache,
                                  src_fp16: *mut c_void,
                                  dst_q: *mut c_void,
                                  dst_s: *mut c_void,
                                  n_demote: u32,
                                  n_kv: u32,
                                  head_dim: u32)
    -> Result<(), String>
{
    let hsaco = cache.compile("kv_promote_fp16_to_q8", KV_PROMOTE_FP16_TO_Q8_SOURCE)?;
    let module = Module::load(&hsaco)?;
    let f = module.function("kv_promote_fp16_to_q8_f32")?;

    let mut s_p  = src_fp16;
    let mut dq_p = dst_q;
    let mut ds_p = dst_s;
    let mut nkv  = n_kv;
    let mut hd   = head_dim;
    let mut args: [*mut c_void; 5] = [
        &mut s_p  as *mut _ as *mut c_void,
        &mut dq_p as *mut _ as *mut c_void,
        &mut ds_p as *mut _ as *mut c_void,
        &mut nkv  as *mut _ as *mut c_void,
        &mut hd   as *mut _ as *mut c_void,
    ];
    let grid = (n_demote, n_kv, 1);
    unsafe { f.launch(grid, (256, 1, 1), 0, None, &mut args)?; }
    Ok(())
}

/// Demote int8 + scale slots into turbo3 cold-tier blocks. `kind`
/// selects K vs V RHT sign masks (the cache owns both sets).
pub fn launch_promote_q8_to_turbo3(cache: &KernelCache,
                                    src_q: *mut c_void,
                                    src_s: *mut c_void,
                                    signs1: *mut c_void,
                                    signs2: *mut c_void,
                                    dst: *mut c_void,
                                    n_demote: u32,
                                    n_kv: u32,
                                    head_dim: u32)
    -> Result<(), String>
{
    assert_eq!(head_dim % ROT_GROUP as u32, 0);
    let hsaco = cache.compile("kv_promote_q8_to_turbo3", KV_PROMOTE_Q8_TO_TURBO3_SOURCE)?;
    let module = Module::load(&hsaco)?;
    let f = module.function("kv_promote_q8_to_turbo3_f32")?;

    let mut sq_p = src_q;
    let mut ss_p = src_s;
    let mut s1_p = signs1;
    let mut s2_p = signs2;
    let mut d_p  = dst;
    let mut nd   = n_demote;
    let mut nkv  = n_kv;
    let mut hd   = head_dim;
    let mut args: [*mut c_void; 8] = [
        &mut sq_p as *mut _ as *mut c_void,
        &mut ss_p as *mut _ as *mut c_void,
        &mut s1_p as *mut _ as *mut c_void,
        &mut s2_p as *mut _ as *mut c_void,
        &mut d_p  as *mut _ as *mut c_void,
        &mut nd   as *mut _ as *mut c_void,
        &mut nkv  as *mut _ as *mut c_void,
        &mut hd   as *mut _ as *mut c_void,
    ];
    let groups_per_head = head_dim / ROT_GROUP as u32;
    let grid = (n_demote * n_kv, groups_per_head, 1);
    unsafe { f.launch(grid, (128, 1, 1), 0, None, &mut args)?; }
    Ok(())
}

/// Bytes per (token, head) slot for `head_dim`. Asserts head_dim is a
/// multiple of the 128-element rotation group.
pub fn slot_bytes(head_dim: usize) -> usize {
    assert_eq!(head_dim % ROT_GROUP, 0,
        "head_dim {head_dim} must be a multiple of {ROT_GROUP}");
    (head_dim / ROT_GROUP) * BLOCKS_PER_GROUP * BYTES_PER_BLOCK
}

/// Per-layer turbo3 KV cache. Like `Gemma4KvCache` but stores K/V in
/// turbo3 packed form (no separate scale buffer — the per-group norm
/// lives in the block).
pub struct TurboKvCache {
    /// `[max_seq, n_kv * slot_bytes]` bytes.
    pub k: DeviceBuf<u8>,
    pub v: DeviceBuf<u8>,
    pub n_kv: usize,
    pub head_dim: usize,
    pub max_seq: usize,
    pub len: usize,
    /// Cached HIP modules — built on first write so a TurboKvCache
    /// instantiation is cheap (no hipcc compile on each layer).
    write_module: Option<Module>,
    /// Sign-mask uploads — one each for K + V rotations. Resident.
    pub signs1_k: DeviceBuf<i8>,
    pub signs2_k: DeviceBuf<i8>,
    pub signs1_v: DeviceBuf<i8>,
    pub signs2_v: DeviceBuf<i8>,
}

impl TurboKvCache {
    pub fn new(n_kv: usize, head_dim: usize, max_seq: usize) -> Result<Self, String> {
        let sb = slot_bytes(head_dim);
        Ok(Self {
            k: DeviceBuf::new(max_seq * n_kv * sb)?,
            v: DeviceBuf::new(max_seq * n_kv * sb)?,
            n_kv, head_dim, max_seq, len: 0,
            write_module: None,
            signs1_k: DeviceBuf::from_slice(CacheKind::K.signs1())?,
            signs2_k: DeviceBuf::from_slice(CacheKind::K.signs2())?,
            signs1_v: DeviceBuf::from_slice(CacheKind::V.signs1())?,
            signs2_v: DeviceBuf::from_slice(CacheKind::V.signs2())?,
        })
    }

    pub fn reset(&mut self) { self.len = 0; }

    /// Lazily compile + load the write module on first use.
    fn ensure_module(&mut self, cache: &KernelCache) -> Result<(), String> {
        if self.write_module.is_none() {
            let hsaco = cache.compile("kv_write_turbo3", KV_WRITE_TURBO3_SOURCE)?;
            self.write_module = Some(Module::load(&hsaco)?);
        }
        Ok(())
    }

    /// Write one token's K (or V) into slot `pos`. `src` is `[n_kv × head_dim]`
    /// fp32 on device. `kind` picks the RHT sign masks.
    pub fn write_step(&mut self, cache: &KernelCache,
                      src: *mut c_void, pos: usize, kind: CacheKind,
                      target: KvTarget) -> Result<(), String>
    {
        assert!(pos < self.max_seq, "turbo3 KV cache full ({pos} ≥ {})", self.max_seq);
        self.ensure_module(cache)?;
        // Resolve all the &self borrows first to disjoint locals before
        // getting the function (which borrows self.write_module through &self).
        let (signs1_ptr, signs2_ptr) = match kind {
            CacheKind::K => (self.signs1_k.raw_ptr(), self.signs2_k.raw_ptr()),
            CacheKind::V => (self.signs1_v.raw_ptr(), self.signs2_v.raw_ptr()),
        };
        let dst_ptr = match target {
            KvTarget::K => self.k.raw_ptr(),
            KvTarget::V => self.v.raw_ptr(),
        };
        let groups_per_head = self.head_dim / ROT_GROUP;
        let module = self.write_module.as_ref().unwrap();
        let f = module.function("kv_write_turbo3_step_f32")?;

        let mut src_p = src;
        let mut s1_p  = signs1_ptr;
        let mut s2_p  = signs2_ptr;
        let mut dst_p = dst_ptr;
        let mut nkv = self.n_kv as u32;
        let mut hd  = self.head_dim as u32;
        let mut p   = pos as u32;
        let mut ms  = self.max_seq as u32;
        let mut args: [*mut c_void; 8] = [
            &mut src_p as *mut _ as *mut c_void,
            &mut s1_p  as *mut _ as *mut c_void,
            &mut s2_p  as *mut _ as *mut c_void,
            &mut dst_p as *mut _ as *mut c_void,
            &mut nkv   as *mut _ as *mut c_void,
            &mut hd    as *mut _ as *mut c_void,
            &mut p     as *mut _ as *mut c_void,
            &mut ms    as *mut _ as *mut c_void,
        ];
        let grid = (self.n_kv as u32, groups_per_head as u32, 1);
        unsafe { f.launch(grid, (128, 1, 1), 0, None, &mut args)?; }
        Ok(())
    }
}

/// Which buffer to write into.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvTarget { K, V }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::turbo3::{decode_rows, CacheKind};

    fn skip_if_no_gpu() -> Option<KernelCache> {
        if hip::device_count().ok().unwrap_or(0) < 1 {
            eprintln!("skip: no HIP device"); return None;
        }
        let _ = hip::Device::set(0).ok()?;
        KernelCache::new().ok()
    }

    /// End-to-end: write one fp32 K tensor through the kernel, copy the
    /// resulting bytes back to host, decode via the CPU reference, and
    /// verify the round-trip SNR matches what the standalone Phase-1
    /// pipeline gives. Proves the cache write path matches the encoder.
    #[test]
    fn write_step_round_trips_through_cpu_decode() {
        let Some(cache) = skip_if_no_gpu() else { return };

        let n_kv = 2;
        let head_dim = 256;
        let max_seq = 16;
        let mut kv = TurboKvCache::new(n_kv, head_dim, max_seq).expect("alloc");

        // Synthetic K projection: [n_kv × head_dim] f32, Gaussian-ish.
        let mut s: u64 = 0xCAFE_F00D;
        let mut rng = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let bits = ((s >> 33) as u32 & 0x007F_FFFF) | 0x3f80_0000;
            (f32::from_bits(bits) - 1.5) * 0.4
        };
        let src: Vec<f32> = (0..n_kv * head_dim).map(|_| rng()).collect();
        let dsrc = DeviceBuf::<f32>::from_slice(&src).expect("upload");

        // Write to slot 3 of K cache, kind=K.
        let mut src_p = dsrc.raw_ptr();
        kv.write_step(&cache, src_p as *mut c_void, 3, CacheKind::K, KvTarget::K)
            .expect("write_step");
        hip::Device(0).synchronize().expect("sync");
        let _ = src_p;

        // Copy back the slot 3 region of K cache.
        let sb = slot_bytes(head_dim);
        let slot_len = n_kv * sb;
        let mut full_k = vec![0u8; max_seq * slot_len];
        kv.k.copy_to_host(&mut full_k).expect("copy");
        let slot = &full_k[3 * slot_len..3 * slot_len + slot_len];

        // Decode and compare to original.
        let mut decoded = vec![0.0f32; n_kv * head_dim];
        decode_rows(slot, head_dim, CacheKind::K, &mut decoded);

        let mut s_sig = 0.0f64;
        let mut s_err = 0.0f64;
        for (xv, dv) in src.iter().zip(decoded.iter()) {
            s_sig += (*xv as f64).powi(2);
            s_err += ((*xv - *dv) as f64).powi(2);
        }
        let snr_db = 10.0 * (s_sig / s_err.max(1e-30)).log10();
        eprintln!("turbo3 KV cache write_step round-trip SNR: {snr_db:.1} dB");
        // Should match the standalone encode path's ~14-15 dB.
        assert!(snr_db > 12.0 && snr_db < 22.0,
                "KV cache round-trip SNR {snr_db:.1} dB outside expected band");

        // Verify other slots are still zero (no clobbering).
        let slot4 = &full_k[4 * slot_len..4 * slot_len + slot_len];
        assert!(slot4.iter().all(|&b| b == 0), "wrote past slot boundary");
        let slot2 = &full_k[2 * slot_len..2 * slot_len + slot_len];
        assert!(slot2.iter().all(|&b| b == 0), "wrote before slot boundary");
    }

    /// fp16 → int8 → fp32 round trip: scale should preserve magnitudes
    /// within ~0.4% (one ULP of int8 quantization).
    #[test]
    fn promote_fp16_to_q8_round_trip() {
        let Some(cache) = skip_if_no_gpu() else { return };

        let n_demote = 4;
        let n_kv     = 2;
        let head_dim = 256;

        // Synth K data, fp16 storage.
        let mut s: u64 = 0xF00D;
        let mut rng = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let bits = ((s >> 33) as u32 & 0x007F_FFFF) | 0x3f80_0000;
            (f32::from_bits(bits) - 1.5) * 0.4
        };
        let src_f32: Vec<f32> = (0..n_demote * n_kv * head_dim).map(|_| rng()).collect();
        let src_f16: Vec<u16> = src_f32.iter()
            .map(|&v| crate::quant::half::f32_to_f16(v))
            .collect();
        let dsrc: DeviceBuf<u16> = DeviceBuf::from_slice(&src_f16).unwrap();
        let dq:   DeviceBuf<i8>  = DeviceBuf::new(n_demote * n_kv * head_dim).unwrap();
        let ds:   DeviceBuf<f32> = DeviceBuf::new(n_demote * n_kv).unwrap();

        launch_promote_fp16_to_q8(&cache,
            dsrc.raw_ptr(), dq.raw_ptr(), ds.raw_ptr(),
            n_demote as u32, n_kv as u32, head_dim as u32).expect("promote");
        hip::Device(0).synchronize().unwrap();

        let mut hq = vec![0i8;  n_demote * n_kv * head_dim];
        let mut hs = vec![0.0f32; n_demote * n_kv];
        dq.copy_to_host(&mut hq).unwrap();
        ds.copy_to_host(&mut hs).unwrap();

        // SNR per (token,head) row — int8 quant should give > 30 dB on
        // any row where the values have reasonable dynamic range.
        // Per-element rel-err is the wrong metric (small values near 0
        // dominate noise; rel err -> inf as orig -> 0).
        let mut min_snr = f32::INFINITY;
        for p in 0..n_demote {
            for h in 0..n_kv {
                let scale = hs[p * n_kv + h];
                let mut s_sig = 0.0f64;
                let mut s_err = 0.0f64;
                for i in 0..head_dim {
                    let idx = (p * n_kv + h) * head_dim + i;
                    let recon = hq[idx] as f32 * scale;
                    let orig  = src_f32[idx];
                    s_sig += (orig as f64).powi(2);
                    s_err += ((orig - recon) as f64).powi(2);
                }
                let snr_db = 10.0 * (s_sig / s_err.max(1e-30)).log10() as f32;
                if snr_db < min_snr { min_snr = snr_db; }
            }
        }
        eprintln!("fp16→q8 worst-row SNR: {min_snr:.1} dB");
        // int8 symmetric quant with amax-based scale gives ≈ 48 dB on
        // unit-variance Gaussian-ish data. Even with the fp16→fp32→q8
        // round trip we should clear 35 dB.
        assert!(min_snr > 35.0, "fp16→q8 SNR {min_snr:.1} dB < 35");
    }

    /// int8 → turbo3 → fp32 round trip: SNR should land in the
    /// expected 12-22 dB band (matching the standalone encode path).
    #[test]
    fn promote_q8_to_turbo3_round_trip() {
        let Some(cache) = skip_if_no_gpu() else { return };
        use crate::quant::turbo3::{decode_rows, CacheKind};

        let n_demote = 4;
        let n_kv     = 2;
        let head_dim = 256;

        // Synth K as int8 + scale (simulate a populated Warm tier).
        let mut s: u64 = 0xBEEF_F00D;
        let mut rng_byte = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (((s >> 50) as i32 % 254) - 127) as i8
        };
        let src_q: Vec<i8> = (0..n_demote * n_kv * head_dim).map(|_| rng_byte()).collect();
        let src_s: Vec<f32> = (0..n_demote * n_kv).map(|_| 0.012f32).collect();

        let dq:   DeviceBuf<i8>  = DeviceBuf::from_slice(&src_q).unwrap();
        let dscale: DeviceBuf<f32> = DeviceBuf::from_slice(&src_s).unwrap();
        let signs1 = DeviceBuf::<i8>::from_slice(CacheKind::K.signs1()).unwrap();
        let signs2 = DeviceBuf::<i8>::from_slice(CacheKind::K.signs2()).unwrap();
        let sb = slot_bytes(head_dim);
        let dout: DeviceBuf<u8>  = DeviceBuf::new(n_demote * n_kv * sb).unwrap();

        launch_promote_q8_to_turbo3(&cache,
            dq.raw_ptr(), dscale.raw_ptr(),
            signs1.raw_ptr(), signs2.raw_ptr(),
            dout.raw_ptr(),
            n_demote as u32, n_kv as u32, head_dim as u32).expect("promote");
        hip::Device(0).synchronize().unwrap();

        // Reconstruct int8 row as fp32 ground truth, decode turbo3 to fp32, compare SNR.
        let mut packed = vec![0u8; n_demote * n_kv * sb];
        dout.copy_to_host(&mut packed).unwrap();

        let n_rows = n_demote * n_kv;
        let bytes_per_row = (head_dim / 128) * 4 * 16;
        assert_eq!(packed.len(), n_rows * bytes_per_row);

        let mut decoded = vec![0.0f32; n_rows * head_dim];
        decode_rows(&packed, head_dim, CacheKind::K, &mut decoded);

        // Ground truth: same int8 × scale we wrote.
        let mut gt = vec![0.0f32; n_rows * head_dim];
        for p in 0..n_demote {
            for h in 0..n_kv {
                let scale = src_s[p * n_kv + h];
                for i in 0..head_dim {
                    gt[(p * n_kv + h) * head_dim + i]
                        = src_q[(p * n_kv + h) * head_dim + i] as f32 * scale;
                }
            }
        }

        let mut s_sig = 0.0f64;
        let mut s_err = 0.0f64;
        for (xv, dv) in gt.iter().zip(decoded.iter()) {
            s_sig += (*xv as f64).powi(2);
            s_err += ((*xv - *dv) as f64).powi(2);
        }
        let snr_db = 10.0 * (s_sig / s_err.max(1e-30)).log10();
        eprintln!("q8→turbo3 SNR ({n_demote}×{n_kv}×hd={head_dim}): {snr_db:.1} dB");
        assert!(snr_db > 10.0 && snr_db < 25.0,
                "q8→turbo3 SNR {snr_db:.1} outside expected 10-25 band");
    }

    #[test]
    fn slot_bytes_panics_on_non_128_multiple() {
        let r = std::panic::catch_unwind(|| slot_bytes(100));
        assert!(r.is_err(), "should panic on non-128-multiple head_dim");
    }

    #[test]
    fn slot_bytes_correct() {
        assert_eq!(slot_bytes(128), 64);
        assert_eq!(slot_bytes(256), 128);
        assert_eq!(slot_bytes(512), 256);
    }
}
