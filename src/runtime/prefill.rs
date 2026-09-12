//! Batched prefill path: process N tokens in one pass instead of N
//! sequential `forward_token` calls.
//!
//! The decode path runs fused dequant+GEMV straight off the on-disk
//! quantized bytes. Prefill instead bulk-dequantizes each weight to an
//! fp16 scratch buffer and runs a real GEMM (rocBLAS HGEMM) so the
//! weight matrix is read once and reused across all N rows.
//!
//! This module currently exposes the building block — `batched_matmul`
//! (Y = X · Wᵀ via HGEMM) — plus its validation. The full batched
//! forward (attention, GDN, orchestration) builds on top.

use std::ffi::c_void;

use crate::gguf::GgmlType;
use crate::hip::{self, DeviceBuf, Module};
use super::KernelCache;

const CVT_SOURCE: &str = include_str!("../../kernels/cvt_f32_f16.cpp");
const GEMM_F16_ROWS_SOURCE: &str = include_str!("../../kernels/gemm_f16_rows.cpp");
/// Activation rows each `gemm_f16_rows_f32` block handles. Must match
/// `NR_TILE` in `kernels/gemm_f16_rows.cpp`.
const GEMM_F16_NR_TILE: usize = 8;

/// Token count at or below which the MMQ GEMM uses its narrow (BN=16)
/// tile instead of the default BN=64. A 16-token verify fills a quarter of
/// the wide tile and wastes the rest.
const NARROW_MMQ_ROWS: usize = 16;

/// Workgroup size for the narrow MMQ tile. Must match
/// `NARROW_THREADS` in the kernels. Smaller than 256 so each
/// thread owns more of the BM x BN tile: with BM*BN fixed at
/// 64x16, outputs-per-thread is 1024/THREADS, and that ratio is
/// what sets LDS reads per sdot4.
const NARROW_MMQ_THREADS: u32 = 64;

/// Largest activation-row count the batched K-quant matvecs handle.
/// Above this the MMQ GEMM takes over, and its BN=64 token tile wastes
/// (64 - n_rows)/64 of every workgroup — at 16 rows that was the single
/// biggest cost in a DFlash round (424 ms of verify). Matches
/// `N_ROWS_MAX` in the `batched16` kernel instantiations.
const MAX_BATCHED_ROWS: usize = 16;
const QUANTIZE_Q8_SOURCE: &str = include_str!("../../kernels/quantize_q8.cpp");
const MMQ_GEMM_Q4K_SOURCE: &str =
    include_str!("../../kernels/mmq_gemm_q4k_repacked.cpp");
const MMQ_GEMM_Q4_0_SOURCE: &str =
    include_str!("../../kernels/mmq_gemm_q4_0_repacked.cpp");
const MMQ_GEMM_IQ4XS_SOURCE: &str =
    include_str!("../../kernels/mmq_gemm_iq4xs_repacked.cpp");
const MMQ_GEMM_Q5K_SOURCE: &str =
    include_str!("../../kernels/mmq_gemm_q5k_repacked.cpp");
const MMQ_GEMM_Q6K_SOURCE: &str =
    include_str!("../../kernels/mmq_gemm_q6k_repacked.cpp");
const MMQ_GEMM_Q8_0_SOURCE: &str =
    include_str!("../../kernels/mmq_gemm_q8_0_repacked.cpp");
const MV_Q4_0_REPACKED_BATCHED_SOURCE: &str =
    include_str!("../../kernels/matvec_q4_0_repacked_batched.cpp");
const MV_IQ4XS_REPACKED_BATCHED_SOURCE: &str =
    include_str!("../../kernels/matvec_iq4xs_repacked_batched.cpp");
const MV_Q4K_REPACKED_BATCHED_SOURCE: &str =
    include_str!("../../kernels/matvec_q4k_repacked_batched.cpp");
const MV_Q5K_REPACKED_BATCHED_SOURCE: &str =
    include_str!("../../kernels/matvec_q5k_repacked_batched.cpp");
const MV_Q6K_REPACKED_BATCHED_SOURCE: &str =
    include_str!("../../kernels/matvec_q6k_repacked_batched.cpp");

/// Dequantize a quantized weight already resident on device to fp16.
/// Same kernels as `dequant_to_f16`, but the input bytes are not
/// re-uploaded — for the prefill forward, weights are already resident.
pub fn dequant_dev_to_f16(cache: &KernelCache, w_dev: &DeviceBuf<u8>,
                          dtype: GgmlType, n_elements: usize)
    -> Result<DeviceBuf<u16>, String>
{
    let (src, kname, weights_per_block, block_threads): (&str, &str, usize, u32) = match dtype {
        GgmlType::Q4_K => (include_str!("../../kernels/dequant_q4_k_f16.cpp"),
                           "dequant_q4_k_f16", 256, 256),
        GgmlType::Q5_K => (include_str!("../../kernels/dequant_q5_k_f16.cpp"),
                           "dequant_q5_k_f16", 256, 256),
        GgmlType::Q6_K => (include_str!("../../kernels/dequant_q6_k_f16.cpp"),
                           "dequant_q6_k_f16", 256, 256),
        GgmlType::Q8_0 => (include_str!("../../kernels/dequant_q8_0_f16.cpp"),
                           "dequant_q8_0_f16", 32, 32),
        other => return Err(format!("dequant_dev_to_f16: unsupported dtype {other:?}")),
    };
    assert_eq!(n_elements % weights_per_block, 0, "n_elements not a block multiple");
    let n_blocks = n_elements / weights_per_block;
    let module = Module::load(&cache.compile(kname, src)?)?;
    let f = module.function(kname)?;
    let out: DeviceBuf<u16> = DeviceBuf::new(n_elements)?;
    let mut w_ptr = w_dev.raw_ptr();
    let mut o_ptr = out.raw_ptr();
    let mut nb = n_blocks as u32;
    let mut args: [*mut c_void; 3] = [
        &mut w_ptr as *mut _ as *mut c_void, &mut o_ptr as *mut _ as *mut c_void,
        &mut nb as *mut _ as *mut c_void];
    unsafe { f.launch((n_blocks as u32,1,1),(block_threads,1,1), 0, None, &mut args)?; }
    hip::Device(0).synchronize()?;
    Ok(out)
}

/// Launch `gemm_f16_rows_f32`: `Y[n_rows, out_dim] = X[n_rows, in_dim] · Wᵀ`
/// for an fp16 weight, fp32 activations and output. Blocks are
/// (out_dim × ceil(n_rows/NR_TILE)); each reduces `in_dim` in LDS.
fn launch_gemm_f16_rows(module: &Module, stream: Option<&hip::Stream>,
                        w_f16: *mut c_void, x: *mut c_void, y: *mut c_void,
                        in_dim: usize, out_dim: usize, n_rows: usize)
    -> Result<(), String>
{
    let f = module.function("gemm_f16_rows_f32")?;
    let block: u32 = 256;
    let mut wp = w_f16;
    let mut xp = x;
    let mut yp = y;
    let mut ia = in_dim as u32;
    let mut oa = out_dim as u32;
    let mut na = n_rows as u32;
    let mut args: [*mut c_void; 6] = [
        &mut wp as *mut _ as *mut c_void,
        &mut xp as *mut _ as *mut c_void,
        &mut yp as *mut _ as *mut c_void,
        &mut ia as *mut _ as *mut c_void,
        &mut oa as *mut _ as *mut c_void,
        &mut na as *mut _ as *mut c_void,
    ];
    let grid_y = ((n_rows + GEMM_F16_NR_TILE - 1) / GEMM_F16_NR_TILE) as u32;
    let smem   = (GEMM_F16_NR_TILE * block as usize * std::mem::size_of::<f32>()) as u32;
    unsafe { f.launch((out_dim as u32, grid_y, 1), (block, 1, 1), smem, stream, &mut args) }
}

/// `Y = X · Wᵀ` via the fp16-weight GEMM kernel.
///
/// - `w_bytes` / `dtype`: the on-disk quantized weight, logical shape
///   `[out_dim, in_dim]` row-major (one output row per `j`).
/// - `x`: activations, `[n_rows, in_dim]` row-major, fp32.
/// - returns `Y` `[n_rows, out_dim]` row-major, fp32.
///
/// Internally: dequant W→fp16, then one fp32-accumulate GEMM that writes
/// `Y` straight out in row-major `[n_rows, out_dim]` order.
pub fn batched_matmul(cache: &KernelCache,
                      w_bytes: &[u8], dtype: GgmlType,
                      x: &[f32], n_rows: usize, in_dim: usize, out_dim: usize)
    -> Result<Vec<f32>, String>
{
    assert_eq!(x.len(), n_rows * in_dim, "x shape mismatch");
    let w_dev: DeviceBuf<u8> = DeviceBuf::from_slice(w_bytes)?;
    let x_dev: DeviceBuf<f32> = DeviceBuf::from_slice(x)?;
    let y_dev = batched_matmul_resident(cache, &w_dev, dtype,
                                        in_dim, out_dim, &x_dev, n_rows)?;
    let mut out = vec![0.0f32; n_rows * out_dim];
    y_dev.copy_to_host(&mut out)?;
    Ok(out)
}

/// Device-resident `Y = X · Wᵀ`: weights stay quantized on device, X/Y
/// are device fp32. Internally dequant W→fp16 then one fp32-accumulate
/// GEMM — the building block of the batched prefill forward.
pub fn batched_matmul_resident(cache: &KernelCache,
                               w_dev: &DeviceBuf<u8>, dtype: GgmlType,
                               in_dim: usize, out_dim: usize,
                               x: &DeviceBuf<f32>, n_rows: usize)
    -> Result<DeviceBuf<f32>, String>
{
    // 1. Dequant W → fp16 [out_dim, in_dim].
    let w_f16 = dequant_dev_to_f16(cache, w_dev, dtype, out_dim * in_dim)?;

    // 2. Y = X · Wᵀ. X and Y stay fp32 — the kernel widens each weight
    //    element in registers, so there is no activation round trip.
    let dy_f32: DeviceBuf<f32> = DeviceBuf::new(n_rows * out_dim)?;
    let module = Module::load(&cache.compile("gemm_f16_rows", GEMM_F16_ROWS_SOURCE)?)?;
    launch_gemm_f16_rows(&module, None, w_f16.raw_ptr(), x.raw_ptr(),
                         dy_f32.raw_ptr(), in_dim, out_dim, n_rows)?;
    hip::Device(0).synchronize()?;
    Ok(dy_f32)
}

/// Pooled prefill GEMM context: modules and fp16 scratch buffers loaded
/// once and reused across every `Y = X · Wᵀ` of a prefill pass.
///
/// `batched_matmul_resident` re-loads two modules and `hipMalloc`s a
/// fresh (up to ~300 MB) fp16 weight buffer on *every* call. Across a
/// 30-layer prefill that fixed per-weight cost dwarfs the actual GEMM
/// work — ~2.4 s on the 31B. This context hoists the modules and the
/// scratch out of the call so the cost is paid once.
pub struct PrefillGemm {
    cvt:       Module,
    /// Multi-row fp16-weight GEMM. The fallback for dtypes with no
    /// repacked MMQ kernel, in place of what used to be a rocBLAS HGEMM.
    gemm_f16_rows: Module,
    deq_q4k:   Module,
    deq_q5k:   Module,
    deq_q6k:   Module,
    deq_q8_0:  Module,
    deq_q4_0:  Module,
    deq_q4_0_repacked: Module,
    deq_iq4xs_repacked: Module,
    deq_iq3s_repacked: Module,   // IQ4_XS kernels, IQ3_S codebook (also mmq/mv below)
    deq_q4k_repacked: Module,
    deq_q5k_repacked: Module,
    deq_q6k_repacked: Module,
    deq_q8_0_repacked: Module,
    quantize_q8: Module,
    mmq_q4_0:    Module,
    mmq_iq4xs:   Module,
    mmq_iq3s:    Module,
    mmq_q4k:     Module,
    mmq_q5k:     Module,
    mmq_q6k:     Module,
    mmq_q8_0:    Module,
    mv_q4_0_batched: Module,   // K=2..4 batched Q4_0 matvec for verify
    mv_iq4xs_batched: Module,  // same, IQ4_XS
    mv_iq3s_batched: Module,   // same, IQ3_S
    mv_q4k_batched: Module,    // K=2..8 batched K-quant matvec for verify
    mv_q5k_batched: Module,
    mv_q6k_batched: Module,
    w_f16:  std::cell::RefCell<DeviceBuf<u16>>,   // dequantised weight
    xq8:    std::cell::RefCell<DeviceBuf<u8>>,    // int8 activations (MMQ path)
}

impl PrefillGemm {
    /// Pre-size the scratch to the largest weight/activation the caller
    /// will pass. Buffers still grow on demand as a safety net.
    pub fn new(cache: &KernelCache, max_w: usize, max_x: usize, _max_y: usize)
        -> Result<Self, String>
    {
        Ok(Self {
            cvt:       Module::load(&cache.compile("cvt_f32_f16", CVT_SOURCE)?)?,
            gemm_f16_rows: Module::load(&cache.compile("gemm_f16_rows",
                           GEMM_F16_ROWS_SOURCE)?)?,
            deq_q4k:   Module::load(&cache.compile("dequant_q4_k_f16",
                           include_str!("../../kernels/dequant_q4_k_f16.cpp"))?)?,
            deq_q5k:   Module::load(&cache.compile("dequant_q5_k_f16",
                           include_str!("../../kernels/dequant_q5_k_f16.cpp"))?)?,
            deq_q6k:   Module::load(&cache.compile("dequant_q6_k_f16",
                           include_str!("../../kernels/dequant_q6_k_f16.cpp"))?)?,
            deq_q8_0:  Module::load(&cache.compile("dequant_q8_0_f16",
                           include_str!("../../kernels/dequant_q8_0_f16.cpp"))?)?,
            deq_q4_0:  Module::load(&cache.compile("dequant_q4_0_f16",
                           include_str!("../../kernels/dequant_q4_0_f16.cpp"))?)?,
            deq_q4_0_repacked: Module::load(&cache.compile("dequant_q4_0_repacked_f16",
                           include_str!("../../kernels/dequant_q4_0_repacked_f16.cpp"))?)?,
            deq_iq4xs_repacked: Module::load(&cache.compile("dequant_iq4xs_repacked_f16",
                           include_str!("../../kernels/dequant_iq4xs_repacked_f16.cpp"))?)?,
            deq_iq3s_repacked: Module::load(&cache.compile("dequant_iq3s_repacked_f16",
                           &crate::quant::iq3_s::kernel_source(
                               include_str!("../../kernels/dequant_iq4xs_repacked_f16.cpp")))?)?,
            deq_q4k_repacked: Module::load(&cache.compile("dequant_q4k_repacked_f16",
                           include_str!("../../kernels/dequant_q4k_repacked_f16.cpp"))?)?,
            deq_q5k_repacked: Module::load(&cache.compile("dequant_q5k_repacked_f16",
                           include_str!("../../kernels/dequant_q5k_repacked_f16.cpp"))?)?,
            deq_q6k_repacked: Module::load(&cache.compile("dequant_q6k_repacked_f16",
                           include_str!("../../kernels/dequant_q6k_repacked_f16.cpp"))?)?,
            deq_q8_0_repacked: Module::load(&cache.compile("dequant_q8_0_repacked_f16",
                           include_str!("../../kernels/dequant_q8_0_repacked.cpp"))?)?,
            quantize_q8: Module::load(&cache.compile("quantize_q8", QUANTIZE_Q8_SOURCE)?)?,
            mmq_q4_0:    Module::load(&cache.compile("mmq_gemm_q4_0_repacked",
                                                     MMQ_GEMM_Q4_0_SOURCE)?)?,
            mmq_iq4xs:   Module::load(&cache.compile("mmq_gemm_iq4xs_repacked",
                                                     MMQ_GEMM_IQ4XS_SOURCE)?)?,
            mmq_iq3s:    Module::load(&cache.compile("mmq_gemm_iq3s_repacked",
                             &crate::quant::iq3_s::kernel_source(MMQ_GEMM_IQ4XS_SOURCE))?)?,
            mmq_q4k:     Module::load(&cache.compile("mmq_gemm_q4k_repacked",
                                                     MMQ_GEMM_Q4K_SOURCE)?)?,
            mmq_q5k:     Module::load(&cache.compile("mmq_gemm_q5k_repacked",
                                                     MMQ_GEMM_Q5K_SOURCE)?)?,
            mmq_q6k:     Module::load(&cache.compile("mmq_gemm_q6k_repacked",
                                                     MMQ_GEMM_Q6K_SOURCE)?)?,
            mmq_q8_0:    Module::load(&cache.compile("mmq_gemm_q8_0_repacked",
                                                     MMQ_GEMM_Q8_0_SOURCE)?)?,
            mv_q4_0_batched: Module::load(&cache.compile("matvec_q4_0_repacked_batched",
                                          MV_Q4_0_REPACKED_BATCHED_SOURCE)?)?,
            mv_iq4xs_batched: Module::load(&cache.compile("matvec_iq4xs_repacked_batched",
                                          MV_IQ4XS_REPACKED_BATCHED_SOURCE)?)?,
            mv_iq3s_batched: Module::load(&cache.compile("matvec_iq3s_repacked_batched",
                    &crate::quant::iq3_s::kernel_source(MV_IQ4XS_REPACKED_BATCHED_SOURCE))?)?,
            mv_q4k_batched: Module::load(&cache.compile("matvec_q4k_repacked_batched",
                                                     MV_Q4K_REPACKED_BATCHED_SOURCE)?)?,
            mv_q5k_batched: Module::load(&cache.compile("matvec_q5k_repacked_batched",
                                                     MV_Q5K_REPACKED_BATCHED_SOURCE)?)?,
            mv_q6k_batched: Module::load(&cache.compile("matvec_q6k_repacked_batched",
                                                     MV_Q6K_REPACKED_BATCHED_SOURCE)?)?,
            w_f16:  std::cell::RefCell::new(DeviceBuf::new(max_w.max(1))?),
            // int8 activations: one BlockQ8 (40 B) per 32-element sub-block.
            xq8:    std::cell::RefCell::new(DeviceBuf::new((max_x.max(32) / 32) * 40)?),
        })
    }

    fn deq(&self, dt: GgmlType) -> Result<(&Module, &'static str, usize, u32), String> {
        Ok(match dt {
            GgmlType::Q4_0   => (&self.deq_q4_0,  "dequant_q4_0_f16",    32,  32),
            GgmlType::Q4_K   => (&self.deq_q4k,   "dequant_q4_k_f16",   256, 256),
            GgmlType::Q5_K   => (&self.deq_q5k,   "dequant_q5_k_f16",   256, 256),
            GgmlType::Q6_K   => (&self.deq_q6k,   "dequant_q6_k_f16",   256, 256),
            GgmlType::Q8_0   => (&self.deq_q8_0,  "dequant_q8_0_f16",    32,  32),
            o => return Err(format!("PrefillGemm: unsupported weight dtype {o:?}")),
        })
    }

    /// Device-resident `Y = X · Wᵀ`, all kernels ordered on `stream` —
    /// no internal device syncs, no per-call module loads or weight
    /// allocations. Only `Y` (`[n_rows, out_dim]` fp32) is freshly
    /// allocated; the fp16 scratch is pooled.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    /// Allocating form of [`matmul_into`]: sizes and returns the output
    /// buffer instead of writing into a caller-owned one.
    pub fn matmul(&self, stream: &hip::Stream,
                  w_dev: &DeviceBuf<u8>, dtype: GgmlType, repacked: bool,
                  in_dim: usize, out_dim: usize,
                  x: &DeviceBuf<f32>, n_rows: usize)
        -> Result<DeviceBuf<f32>, String>
    {
        let dst: DeviceBuf<f32> = DeviceBuf::new(n_rows * out_dim)?;
        self.matmul_into(stream, &dst, w_dev, dtype, repacked,
                         in_dim, out_dim, x, n_rows)?;
        Ok(dst)
    }

    /// Like [`matmul`] but writes into a caller-owned `dst` buffer
    /// (first `n_rows * out_dim` elements) instead of allocating a
    /// fresh result. Used by `verify_forward`, which is called per
    /// spec-decode round and cannot afford ~600 hipMallocs each call.
    /// `dst` must be at least `n_rows * out_dim` long.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_into(&self, stream: &hip::Stream,
                       dst: &DeviceBuf<f32>,
                       w_dev: &DeviceBuf<u8>, dtype: GgmlType, repacked: bool,
                       in_dim: usize, out_dim: usize,
                       x: &DeviceBuf<f32>, n_rows: usize)
        -> Result<(), String>
    {
        let n_y = n_rows * out_dim;
        if dst.len() < n_y {
            return Err(format!("matmul_into: dst.len={} < n_rows*out_dim={n_y}", dst.len()));
        }
        self.matmul_into_raw(stream, dst.raw_ptr(), w_dev, dtype, repacked,
                             in_dim, out_dim, x.raw_ptr(), n_rows)
    }

    /// `matmul_into` over raw device pointers, for callers writing into a
    /// slice of a larger buffer — DFlash projects a block's K/V straight
    /// into the middle of its context cache, which no `&DeviceBuf` names.
    /// Caller owns the bounds check.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_into_raw(&self, stream: &hip::Stream,
                           dst: *mut c_void,
                           w_dev: &DeviceBuf<u8>, dtype: GgmlType, repacked: bool,
                           in_dim: usize, out_dim: usize,
                           x: *mut c_void, n_rows: usize)
        -> Result<(), String>
    {
        // Repacked Q8_0: no batched-matvec variant exists, so MMQ for
        // every K — still far beats the dequant→HGEMM fallback.
        if repacked && dtype == GgmlType::Q8_0 {
            return self.matmul_mmq_into(stream, dst, w_dev, dtype, in_dim, out_dim, x, n_rows);
        }
        if repacked && matches!(dtype,
            GgmlType::Q4_0 | GgmlType::IQ4_XS | GgmlType::IQ3_S
            | GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::Q6_K) {
            if n_rows >= 1 && n_rows <= 4 {
                // K=1..4 small-batch: a per-dtype batched matvec that reads
                // each weight sub-block once and dots against all n_rows
                // activation rows. Cap at 4 to bound per-thread accumulator
                // pressure (= ROWS*N_ROWS_MAX VGPRs); higher K falls back
                // to MMQ. K=4 is the empirical sweet spot for accept × tok/s.
                //
                // K=1 used to fall through to MMQ (BN=64 row tile wastes
                // 98% of each workgroup → 390 ms verify(K=1) on 31B).
                // Letting the batched kernel handle K=1 drops it to the
                // ~50 ms range. The kernel always reserves N_ROWS_MAX=4
                // per-thread accumulator slots regardless of actual K, but
                // the wasted slots cost a few VGPRs, not bandwidth.
                //
                // A/B history on this path at K=4 (env-gated paths since
                // removed): MMQ was ~3.3x slower because BN=64 wastes 94%
                // of each tile; dequant→HGEMM was ~20x slower (per-call
                // Q4_K dequant of [5376, 21504] is ~314us; 180 FFN GEMMs
                // per verify = ~56ms of pure dequant). gfx906 has no
                // tensor cores, so fp16 GEMM has no compute advantage —
                // dp4a does 4 int8 multiplies per cycle vs fp16's 2, and
                // reads ¼ the weight bytes. dp4a wins both axes.
                return self.matmul_kquant_batched_into(stream, dst, w_dev, dtype,
                                                       in_dim, out_dim, x, n_rows);
            }
            return self.matmul_mmq_into(stream, dst, w_dev, dtype, in_dim, out_dim, x, n_rows);
        }

        let n_w = out_dim * in_dim;
        Self::grow(&self.w_f16,  n_w, stream)?;
        let w_f16  = self.w_f16.borrow();

        let mut w_ptr = w_dev.raw_ptr();
        let mut o_ptr = w_f16.raw_ptr();
        if repacked {
            let (module, kname) = match dtype {
                GgmlType::Q4_0 => (&self.deq_q4_0_repacked, "dequant_q4_0_repacked_f16"),
                GgmlType::IQ4_XS => (&self.deq_iq4xs_repacked, "dequant_iq4xs_repacked_f16"),
                GgmlType::IQ3_S => (&self.deq_iq3s_repacked, "dequant_iq4xs_repacked_f16"),
                GgmlType::Q5_K => (&self.deq_q5k_repacked, "dequant_q5k_repacked_f16"),
                GgmlType::Q6_K => (&self.deq_q6k_repacked, "dequant_q6k_repacked_f16"),
                GgmlType::Q8_0 => (&self.deq_q8_0_repacked, "dequant_q8_0_repacked_f16"),
                _              => (&self.deq_q4k_repacked, "dequant_q4k_repacked_f16"),
            };
            let f = module.function(kname)?;
            let mut ia = in_dim as u32;
            let mut oa = out_dim as u32;
            let mut da: [*mut c_void; 4] = [
                &mut w_ptr as *mut _ as *mut c_void, &mut o_ptr as *mut _ as *mut c_void,
                &mut ia    as *mut _ as *mut c_void, &mut oa as *mut _ as *mut c_void];
            unsafe { f.launch(((n_w / 32) as u32, 1, 1), (32, 1, 1),
                              0, Some(stream), &mut da)?; }
        } else if dtype == GgmlType::F32 {
            let f = self.cvt.function("cvt_f32_to_f16")?;
            let block: u32 = 256;
            let mut nb = n_w as u32;
            let mut da: [*mut c_void; 3] = [
                &mut w_ptr as *mut _ as *mut c_void, &mut o_ptr as *mut _ as *mut c_void,
                &mut nb    as *mut _ as *mut c_void];
            unsafe { f.launch(((n_w as u32 + block - 1) / block, 1, 1), (block, 1, 1),
                              0, Some(stream), &mut da)?; }
        } else {
            let (module, kname, wpb, bt) = self.deq(dtype)?;
            assert_eq!(n_w % wpb, 0, "weight elems not a block multiple");
            let n_blocks = (n_w / wpb) as u32;
            let f = module.function(kname)?;
            let mut nb = n_blocks;
            let mut da: [*mut c_void; 3] = [
                &mut w_ptr as *mut _ as *mut c_void, &mut o_ptr as *mut _ as *mut c_void,
                &mut nb    as *mut _ as *mut c_void];
            unsafe { f.launch((n_blocks,1,1),(bt,1,1), 0, Some(stream), &mut da)?; }
        }

        // Our own fp16 GEMM rather than rocBLAS: x and dst stay fp32 on
        // both sides (no narrow-in/widen-out round trip), and gfx906 keeps
        // working on ROCm 7.x, whose rocBLAS ships no kernels for it and
        // aborts the process the moment a handle is created.
        launch_gemm_f16_rows(&self.gemm_f16_rows, Some(stream), w_f16.raw_ptr(),
                             x, dst, in_dim, out_dim, n_rows)
    }

    fn grow(buf: &std::cell::RefCell<DeviceBuf<u16>>, n: usize, stream: &hip::Stream)
        -> Result<(), String>
    {
        if buf.borrow().len() < n {
            stream.synchronize()?;          // old buffer may still be in flight
            *buf.borrow_mut() = DeviceBuf::new(n)?;
        }
        Ok(())
    }
    fn matmul_kquant_batched_into(&self, stream: &hip::Stream, dst: *mut c_void,
                                  w_dev: &DeviceBuf<u8>, dtype: GgmlType,
                                  in_dim: usize, out_dim: usize,
                                  x: *mut c_void, n_rows: usize)
        -> Result<(), String>
    {
        debug_assert!(n_rows >= 1 && n_rows <= MAX_BATCHED_ROWS,
                      "matmul_kquant_batched_into: n_rows must be 1..={MAX_BATCHED_ROWS}");
        // Two instantiations per dtype. Up to 4 rows the tuned
        // (ROWS=2, N_ROWS_MAX=4) kernel puts 8 output rows in a workgroup;
        // beyond that the wide (ROWS=1, N_ROWS_MAX=16) one puts 4 there,
        // trading output rows per workgroup for accumulator headroom so
        // the weight is still streamed once for every activation row.
        let wide = n_rows > 4;
        let (module, kname) = match (dtype, wide) {
            (GgmlType::Q5_K, false) => (&self.mv_q5k_batched, "matvec_q5k_repacked_batched_f32"),
            (GgmlType::Q5_K, true)  => (&self.mv_q5k_batched, "matvec_q5k_repacked_batched16_f32"),
            (GgmlType::Q6_K, false) => (&self.mv_q6k_batched, "matvec_q6k_repacked_batched_f32"),
            (GgmlType::Q6_K, true)  => (&self.mv_q6k_batched, "matvec_q6k_repacked_batched16_f32"),
            (GgmlType::Q4_K, false) => (&self.mv_q4k_batched, "matvec_q4k_repacked_batched_f32"),
            (GgmlType::Q4_K, true)  => (&self.mv_q4k_batched, "matvec_q4k_repacked_batched16_f32"),
            (GgmlType::Q4_0, false) => (&self.mv_q4_0_batched, "matvec_q4_0_repacked_batched_f32"),
            (GgmlType::Q4_0, true)  => (&self.mv_q4_0_batched, "matvec_q4_0_repacked_batched16_f32"),
            (GgmlType::IQ4_XS, false) => (&self.mv_iq4xs_batched, "matvec_iq4xs_repacked_batched_f32"),
            (GgmlType::IQ4_XS, true)  => (&self.mv_iq4xs_batched, "matvec_iq4xs_repacked_batched16_f32"),
            (GgmlType::IQ3_S, false) => (&self.mv_iq3s_batched, "matvec_iq4xs_repacked_batched_f32"),
            (GgmlType::IQ3_S, true)  => (&self.mv_iq3s_batched, "matvec_iq4xs_repacked_batched16_f32"),
            (other, _) => return Err(format!("matmul_kquant_batched_into: unsupported {other:?}")),
        };
        let rows_per_wg: u32 = if wide { 4 } else { 8 };

        // Reuse the MMQ path's int8 activation scratch.
        let n_xq8 = (n_rows * in_dim / 32) * 40;
        if self.xq8.borrow().len() < n_xq8 {
            stream.synchronize()?;
            *self.xq8.borrow_mut() = DeviceBuf::new(n_xq8)?;
        }
        let xq8 = self.xq8.borrow();

        // 1. Quantise X[n_rows, in_dim] → BlockQ8[n_rows, in_dim/32].
        let qf = self.quantize_q8.function("quantize_q8_f32")?;
        let mut xp = x; let mut qp = xq8.raw_ptr();
        let mut ind = in_dim as u32;
        let mut qa: [*mut c_void; 3] = [
            &mut xp as *mut _ as *mut c_void, &mut qp as *mut _ as *mut c_void,
            &mut ind as *mut _ as *mut c_void];
        unsafe { qf.launch((((in_dim as u32) + 255) / 256, n_rows as u32, 1),
                           (256, 1, 1), 0, Some(stream), &mut qa)?; }

        // 2. Batched matvec — grid = ceil(out_dim / rows_per_wg).
        let gf = module.function(kname)?;
        let mut wp = w_dev.raw_ptr(); let mut xqp = xq8.raw_ptr(); let mut yp = dst;
        let mut ia = in_dim as u32; let mut oa = out_dim as u32; let mut nr = n_rows as u32;
        let mut ga: [*mut c_void; 6] = [
            &mut wp as *mut _ as *mut c_void, &mut xqp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void, &mut ia as *mut _ as *mut c_void,
            &mut oa as *mut _ as *mut c_void, &mut nr as *mut _ as *mut c_void];
        let grid_x = (out_dim as u32 + rows_per_wg - 1) / rows_per_wg;
        unsafe { gf.launch((grid_x, 1, 1), (256, 1, 1), 0, Some(stream), &mut ga)?; }
        Ok(())
    }

    /// Caller-owned-output sibling of [`matmul_mmq`].
    #[allow(clippy::too_many_arguments)]
    fn matmul_mmq_into(&self, stream: &hip::Stream, dst: *mut c_void,
                       w_dev: &DeviceBuf<u8>, dtype: GgmlType,
                       in_dim: usize, out_dim: usize,
                       x: *mut c_void, n_rows: usize)
        -> Result<(), String>
    {
        // BN=64 by default; BN=16 once the token count no longer fills a
        // wide tile. At 16 tokens the wide tile leaves 48 of its 64 column
        // slots masked off, which is pure waste on the dominant kernel of a
        // DFlash verify.
        let narrow = n_rows <= NARROW_MMQ_ROWS;
        let (module, kname) = match (dtype, narrow) {
            (GgmlType::Q4_0, false) => (&self.mmq_q4_0, "mmq_gemm_q4_0_repacked_f32"),
            (GgmlType::Q4_0, true)  => (&self.mmq_q4_0, "mmq_gemm_q4_0_repacked_narrow_f32"),
            (GgmlType::IQ4_XS, false) => (&self.mmq_iq4xs, "mmq_gemm_iq4xs_repacked_f32"),
            (GgmlType::IQ4_XS, true)  => (&self.mmq_iq4xs, "mmq_gemm_iq4xs_repacked_narrow_f32"),
            (GgmlType::IQ3_S, false) => (&self.mmq_iq3s, "mmq_gemm_iq4xs_repacked_f32"),
            (GgmlType::IQ3_S, true)  => (&self.mmq_iq3s, "mmq_gemm_iq4xs_repacked_narrow_f32"),
            (GgmlType::Q5_K, false) => (&self.mmq_q5k,  "mmq_gemm_q5k_repacked_f32"),
            (GgmlType::Q5_K, true)  => (&self.mmq_q5k,  "mmq_gemm_q5k_repacked_narrow_f32"),
            (GgmlType::Q6_K, false) => (&self.mmq_q6k,  "mmq_gemm_q6k_repacked_f32"),
            (GgmlType::Q6_K, true)  => (&self.mmq_q6k,  "mmq_gemm_q6k_repacked_narrow_f32"),
            (GgmlType::Q8_0, false) => (&self.mmq_q8_0, "mmq_gemm_q8_0_repacked_f32"),
            (GgmlType::Q8_0, true)  => (&self.mmq_q8_0, "mmq_gemm_q8_0_repacked_narrow_f32"),
            (_, false)              => (&self.mmq_q4k,  "mmq_gemm_q4k_repacked_f32"),
            (_, true)               => (&self.mmq_q4k,  "mmq_gemm_q4k_repacked_narrow_f32"),
        };
        let bm: u32 = if narrow { 16 } else { 64 };
        let bn: u32 = if narrow { 16 } else { 64 };
        let n_xq8 = (n_rows * in_dim / 32) * 40;
        if self.xq8.borrow().len() < n_xq8 {
            stream.synchronize()?;
            *self.xq8.borrow_mut() = DeviceBuf::new(n_xq8)?;
        }
        let xq8 = self.xq8.borrow();
        let qf = self.quantize_q8.function("quantize_q8_f32")?;
        let mut xp = x; let mut qp = xq8.raw_ptr();
        let mut ind = in_dim as u32;
        let mut qa: [*mut c_void; 3] = [
            &mut xp as *mut _ as *mut c_void, &mut qp as *mut _ as *mut c_void,
            &mut ind as *mut _ as *mut c_void];
        unsafe { qf.launch((((in_dim as u32) + 255) / 256, n_rows as u32, 1),
                           (256, 1, 1), 0, Some(stream), &mut qa)?; }

        let gf = module.function(kname)?;
        let mut wp = w_dev.raw_ptr(); let mut xqp = xq8.raw_ptr(); let mut yp = dst;
        let mut ia = in_dim as u32; let mut oa = out_dim as u32; let mut pa = n_rows as u32;
        let mut ga: [*mut c_void; 6] = [
            &mut wp as *mut _ as *mut c_void, &mut xqp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void, &mut ia as *mut _ as *mut c_void,
            &mut oa as *mut _ as *mut c_void, &mut pa as *mut _ as *mut c_void];
        let threads: u32 = if narrow { NARROW_MMQ_THREADS } else { 256 };
        unsafe { gf.launch(((out_dim as u32 + bm - 1) / bm, (n_rows as u32 + bn - 1) / bn, 1),
                           (threads, 1, 1), 0, Some(stream), &mut ga)?; }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `matmul_into` must agree with the CPU dequant oracle at every row
    /// count that changes which kernel runs: 1..4 take the tuned batched
    /// matvec, 5..16 the wide (ROWS=1, N_ROWS_MAX=16) one, and >16 falls
    /// through to the MMQ GEMM. The 5..16 band is new and is what DFlash's
    /// 16-token block depends on.
    #[test]
    fn matmul_into_matches_oracle_across_row_counts() {
        let Some(cache) = crate::test_support::kernel_cache() else { return };
        let _dev = hip::Device::set(0).unwrap();
        let stream = hip::Stream::new().expect("stream");

        let in_dim = 1024usize;
        let out_dim = 256usize;
        let max_rows = 20usize;

        let mut seed: u64 = 0xB47C_4ED0;
        let mut rng_u8 = || -> u8 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 56) as u8
        };
        let mut xs: u64 = 0x0BAD_CAFE;
        let mut x_rng = || { xs = xs.wrapping_mul(6364136223846793005)
                                    .wrapping_add(1442695040888963407);
                             ((xs >> 33) as u32 as f32 / u32::MAX as f32) - 0.5 };
        let x: Vec<f32> = (0..max_rows * in_dim).map(|_| x_rng()).collect();
        let dx: DeviceBuf<f32> = DeviceBuf::from_slice(&x).unwrap();

        let gemm = PrefillGemm::new(&cache, out_dim * in_dim,
                                    max_rows * in_dim, max_rows * out_dim).unwrap();

        // Every repacked dtype the GEMM dispatch can see. Q8_0 has no
        // batched matvec and goes straight to MMQ at every row count, so
        // it is the only dtype that exercises the narrow tile at 1..4 too —
        // and it is what a DFlash drafter is made of.
        for dtype in [GgmlType::Q4_0, GgmlType::Q4_K, GgmlType::Q5_K,
                      GgmlType::Q6_K, GgmlType::Q8_0, GgmlType::IQ4_XS,
                      GgmlType::IQ3_S] {
            let (bs, bpb) = (dtype.block_size_elements() as usize,
                             dtype.bytes_per_block() as usize);
            let n_blocks = out_dim * (in_dim / bs);
            let mut w = vec![0u8; n_blocks * bpb];
            for b in &mut w { *b = rng_u8(); }
            // Tame the fp16 scales so synthetic blocks stay in range.
            // Q4_0/Q4_K/Q5_K put `d` first; Q6_K puts it LAST, after
            // ql[128]+qh[64]+scales[16]. Writing it at offset 0 there
            // leaves the real `d` as random bytes, which decode to
            // NaN/Inf often enough to poison the whole comparison.
            for blk in 0..n_blocks {
                let off = blk * bpb;
                let d = ((blk % 23) as f32 - 11.0) * 0.004;
                let d_off = if dtype == GgmlType::Q6_K { off + bpb - 2 } else { off };
                w[d_off..d_off + 2].copy_from_slice(
                    &crate::quant::half::f32_to_f16(d).to_le_bytes());
                if matches!(dtype, GgmlType::Q4_K | GgmlType::Q5_K) {
                    let dmin = ((blk % 13) as f32 - 6.0) * 0.002;
                    w[off + 2..off + 4].copy_from_slice(
                        &crate::quant::half::f32_to_f16(dmin).to_le_bytes());
                }
            }
            let mut w_f32 = vec![0.0f32; out_dim * in_dim];
            match dtype {
                GgmlType::Q4_0 => crate::quant::q4_0::dequantize_to_f32(&w, &mut w_f32),
                GgmlType::Q4_K => crate::quant::q4_k::dequantize_to_f32(&w, &mut w_f32),
                GgmlType::Q5_K => crate::quant::q5_k::dequantize_to_f32(&w, &mut w_f32),
                GgmlType::Q8_0 => crate::quant::q8_0::dequantize_to_f32(&w, &mut w_f32),
                GgmlType::IQ4_XS => crate::quant::iq4_xs::dequantize_to_f32(&w, &mut w_f32),
                GgmlType::IQ3_S => crate::quant::iq3_s::dequantize_to_f32(&w, &mut w_f32),
                _              => crate::quant::q6_k::dequantize_to_f32(&w, &mut w_f32),
            }

            let packed = match dtype {
                GgmlType::Q4_0 => crate::quant::q4_0::repack_for_matvec(&w, in_dim, out_dim),
                GgmlType::Q4_K => crate::quant::q4_k::repack_for_matvec(&w, in_dim, out_dim),
                GgmlType::Q5_K => crate::quant::q5_k::repack_for_matvec(&w, in_dim, out_dim),
                GgmlType::Q8_0 => crate::quant::q8_0::repack_for_matvec(&w, in_dim, out_dim),
                GgmlType::IQ4_XS => crate::quant::iq4_xs::repack_for_matvec(&w, in_dim, out_dim),
                GgmlType::IQ3_S => crate::quant::iq3_s::repack_for_matvec(&w, in_dim, out_dim),
                _              => crate::quant::q6_k::repack_for_matvec(&w, in_dim, out_dim),
            };
            let dw: DeviceBuf<u8> = DeviceBuf::from_slice(&packed).unwrap();
            let dst: DeviceBuf<f32> = DeviceBuf::new(max_rows * out_dim).unwrap();

            for &n_rows in &[1usize, 4, 5, 8, 15, 16, 20] {
                gemm.matmul_into(&stream, &dst, &dw, dtype, true,
                                 in_dim, out_dim, &dx, n_rows).expect("matmul_into");
                stream.synchronize().unwrap();
                let mut got = vec![0.0f32; max_rows * out_dim];
                dst.copy_to_host(&mut got).unwrap();

                let mut want = vec![0.0f32; n_rows * out_dim];
                for r in 0..n_rows {
                    let mut row = vec![0.0f32; out_dim];
                    crate::cpu::ops::matvec(&x[r * in_dim..(r + 1) * in_dim], &w_f32,
                                            in_dim, out_dim, &mut row);
                    want[r * out_dim..(r + 1) * out_dim].copy_from_slice(&row);
                }
                let mut num = 0.0f64;
                let mut den = 0.0f64;
                for i in 0..n_rows * out_dim {
                    num += ((got[i] - want[i]) as f64).powi(2);
                    den += (want[i] as f64).powi(2);
                }
                let e = (num / den.max(1e-30)).sqrt() as f32;
                eprintln!("matmul_into {dtype:?} rows={n_rows}: rel_l2={e:.3e}");
                assert!(e < 1.5e-2,
                    "{dtype:?} at {n_rows} rows: rel_l2 {e:.3e} too large");
            }
        }
    }

    #[test]
    fn batched_matmul_matches_sequential_q4_k() {
        let Some(cache) = crate::test_support::kernel_cache() else { return };
        let _dev = hip::Device::set(0).unwrap();
        // Synthesise a Q4_K weight + a batch of activations.
        use crate::quant::q4_k::{BLOCK_SIZE, BYTES_PER_BLOCK};
        let in_dim = 2048usize;
        let out_dim = 512usize;
        let n_rows = 8usize;
        let n_blocks = out_dim * (in_dim / BLOCK_SIZE);
        let mut w = vec![0u8; n_blocks * BYTES_PER_BLOCK];
        let mut s: u64 = 0xBEEF_F00D;
        let mut rng_u8 = || { s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                              (s >> 56) as u8 };
        for blk in 0..n_blocks {
            let off = blk * BYTES_PER_BLOCK;
            w[off..off+2].copy_from_slice(&crate::quant::half::f32_to_f16(0.01).to_le_bytes());
            w[off+2..off+4].copy_from_slice(&crate::quant::half::f32_to_f16(0.005).to_le_bytes());
            for i in 0..12  { w[off + 4  + i] = rng_u8(); }
            for i in 0..128 { w[off + 16 + i] = rng_u8(); }
        }
        let mut xs: u64 = 0xCAFE;
        let mut x_rng = || { xs = xs.wrapping_mul(6364136223846793005).wrapping_add(1);
                             ((xs >> 40) as u32 as f32 / (1u32<<24) as f32) - 0.5 };
        let x: Vec<f32> = (0..n_rows*in_dim).map(|_| x_rng()).collect();

        // Batched GEMM result.
        let gpu = batched_matmul(&cache, &w, GgmlType::Q4_K,
                                 &x, n_rows, in_dim, out_dim).expect("batched_matmul");

        // Reference: per-row fused-dequant matvec (the fp32 decode path).
        // The HGEMM path is fp16-storage / fp32-accumulate, so it carries
        // genuine fp16-input rounding the decode path doesn't. The check
        // is therefore against the *output magnitude*: a transpose/layout
        // bug shows up as ~100% error, fp16 noise as ≪1%.
        let mut worst_abs = 0.0f32;
        let mut peak_mag  = 0.0f32;
        for r in 0..n_rows {
            let row_x = &x[r*in_dim..(r+1)*in_dim];
            let row_y = super::super::kernels::matvec_q4_k_f32(
                &cache, &w, row_x, in_dim, out_dim).expect("matvec ref");
            for j in 0..out_dim {
                let d = (gpu[r*out_dim + j] - row_y[j]).abs();
                if d > worst_abs { worst_abs = d; }
                if row_y[j].abs() > peak_mag { peak_mag = row_y[j].abs(); }
            }
        }
        let rel_to_peak = worst_abs / peak_mag;
        eprintln!("batched HGEMM vs fp32 matvec: worst_abs={worst_abs:.3e}, \
                   peak |y|={peak_mag:.2}, worst/peak={rel_to_peak:.4}");
        // fp16 inputs over a 2048-term contraction: error is a small
        // fraction of the peak output. >2% would indicate a real bug.
        assert!(rel_to_peak < 0.02,
            "batched HGEMM error {rel_to_peak:.4} of peak — too large for fp16 noise");
    }
}
