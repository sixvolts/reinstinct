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
