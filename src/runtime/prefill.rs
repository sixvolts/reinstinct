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
pub(crate) const GEMM_F32_TN_SOURCE: &str = include_str!("../../kernels/gemm_f32_tn.cpp");

/// `Y[n_rows, out_dim] = X[n_rows, in_dim] · Wᵀ` for an fp32 weight:
/// kernels/gemm_f32_tn.cpp (the llama fork's gcn_f32_gemm_tn_rb, 64x64
/// tiles) straight off the weight — no fp16 conversion per call.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_gemm_f32_tn(module: &Module, stream: &hip::Stream,
                                 w: *mut c_void, x: *mut c_void, y: *mut c_void,
                                 in_dim: usize, out_dim: usize, n_rows: usize) -> Result<(), String> {
    let vec = in_dim % 4 == 0 && (w as usize) % 16 == 0 && (x as usize) % 16 == 0;
    let f = module.function(if vec { "gemm_f32_tn_vec_f32" } else { "gemm_f32_tn_f32" })?;
    let (mut a, mut b, mut c) = (w, x, y);
    let (mut m, mut n, mut k) = (out_dim as i32, n_rows as i32, in_dim as i32);
    let (mut lda, mut ldb, mut ldc) = (in_dim as i32, in_dim as i32, out_dim as i32);
    let mut args: [*mut c_void; 9] = [
        &mut a as *mut _ as *mut c_void, &mut b as *mut _ as *mut c_void,
        &mut c as *mut _ as *mut c_void, &mut m as *mut _ as *mut c_void,
        &mut n as *mut _ as *mut c_void, &mut k as *mut _ as *mut c_void,
        &mut lda as *mut _ as *mut c_void, &mut ldb as *mut _ as *mut c_void,
        &mut ldc as *mut _ as *mut c_void];
    let grid = (n_rows.div_ceil(64) as u32, out_dim.div_ceil(64) as u32, 1);
    unsafe { f.launch(grid, (256, 1, 1), 0, Some(stream), &mut args) }
}
/// K-split count for `launch_gemm_f32_tn_split`: enough slices that a
/// skinny GEMM fills the card (~2 workgroups per CU on a 60-CU MI50),
/// each slice at least 128 deep. 1 = no split.
pub(crate) fn gemm_f32_splits(in_dim: usize, out_dim: usize, n_rows: usize) -> usize {
    let tiles = n_rows.div_ceil(64) * out_dim.div_ceil(64);
    let mut s = 1;
    while tiles * s < 120 && in_dim / (2 * s) >= 128 { s *= 2; }
    s
}

/// `launch_gemm_f32_tn` with K split `splits` ways: partial products go
/// to `scratch` (`splits * n_rows * out_dim` floats) and a second launch
/// sums them in fixed order into `y`.
pub(crate) fn launch_gemm_f32_tn_split(module: &Module, stream: &hip::Stream,
                                       w: *mut c_void, x: *mut c_void, y: *mut c_void,
                                       in_dim: usize, out_dim: usize, n_rows: usize,
                                       splits: usize, scratch: *mut c_void) -> Result<(), String> {
    let vec = in_dim % 4 == 0 && (w as usize) % 16 == 0 && (x as usize) % 16 == 0;
    let f = module.function(if vec { "gemm_f32_tn_splitk_vec_f32" } else { "gemm_f32_tn_splitk_f32" })?;
    let (mut a, mut b, mut c) = (w, x, scratch);
    let (mut m, mut n, mut k) = (out_dim as i32, n_rows as i32, in_dim as i32);
    let (mut lda, mut ldb, mut ldc) = (in_dim as i32, in_dim as i32, out_dim as i32);
    let mut kc = (in_dim.div_ceil(splits).div_ceil(16) * 16) as i32;
    let mut ss = (n_rows * out_dim) as i64;
    let mut args: [*mut c_void; 11] = [
        &mut a as *mut _ as *mut c_void, &mut b as *mut _ as *mut c_void,
        &mut c as *mut _ as *mut c_void, &mut m as *mut _ as *mut c_void,
        &mut n as *mut _ as *mut c_void, &mut k as *mut _ as *mut c_void,
        &mut lda as *mut _ as *mut c_void, &mut ldb as *mut _ as *mut c_void,
        &mut ldc as *mut _ as *mut c_void, &mut kc as *mut _ as *mut c_void,
        &mut ss as *mut _ as *mut c_void];
    let grid = (n_rows.div_ceil(64) as u32, out_dim.div_ceil(64) as u32, splits as u32);
    unsafe { f.launch(grid, (256, 1, 1), 0, Some(stream), &mut args)?; }

    let f = module.function("gemm_f32_splitk_reduce")?;
    let (mut p, mut yy) = (scratch, y);
    let mut nn = ss;
    let mut sp = splits as i32;
    let mut args: [*mut c_void; 4] = [
        &mut p as *mut _ as *mut c_void, &mut yy as *mut _ as *mut c_void,
        &mut nn as *mut _ as *mut c_void, &mut sp as *mut _ as *mut c_void];
    unsafe { f.launch(((ss as usize).div_ceil(256) as u32, 1, 1), (256, 1, 1), 0, Some(stream), &mut args) }
}

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

/// Largest activation-row count the small-batch matvecs handle (the
/// `w8` entry of kernels/matvec_batched_nr_entries.h). Above this the
/// narrow MMQ tile takes over: at 16 rows it beats a 16-row matvec, whose
/// per-lane dot work and 16 x ROWS wave reductions outgrow the saved
/// weight traffic; at 8 the matvec is 1.4-2.5x faster.
pub(crate) const MAX_BATCHED_ROWS: usize = 8;

/// The small-batch matvecs (kernels/matvec_batched_nr.h), one module per
/// repacked weight format: `y[n, out] = x[n, in] · Wᵀ` for 1..=
/// MAX_BATCHED_ROWS activation rows, already quantised to BlockQ8. Shared
/// by PrefillGemm and the Qwen 3.5 runtime.
pub(crate) struct SmallBatchMatvec {
    q4_0: Module, iq4xs: Module, iq3s: Module, q4k: Module, q5k: Module, q6k: Module, q8_0: Module,
}

impl SmallBatchMatvec {
    pub(crate) fn new(cache: &KernelCache) -> Result<Self, String> {
        let ld = |name: &str, src: &str| -> Result<Module, String> { Module::load(&cache.compile(name, src)?) };
        Ok(Self {
            q4_0:  ld("matvec_q4_0_repacked_batched", MV_Q4_0_REPACKED_BATCHED_SOURCE)?,
            iq4xs: ld("matvec_iq4xs_repacked_batched", MV_IQ4XS_REPACKED_BATCHED_SOURCE)?,
            iq3s:  ld("matvec_iq3s_repacked_batched",
                      &crate::quant::iq3_s::kernel_source(MV_IQ4XS_REPACKED_BATCHED_SOURCE))?,
            q4k:   ld("matvec_q4k_repacked_batched", MV_Q4K_REPACKED_BATCHED_SOURCE)?,
            q5k:   ld("matvec_q5k_repacked_batched", MV_Q5K_REPACKED_BATCHED_SOURCE)?,
            q6k:   ld("matvec_q6k_repacked_batched", MV_Q6K_REPACKED_BATCHED_SOURCE)?,
            q8_0:  ld("matvec_q8_0_repacked_batched", MV_Q8_0_REPACKED_BATCHED_SOURCE)?,
        })
    }

    pub(crate) fn supports(dtype: GgmlType) -> bool {
        matches!(dtype, GgmlType::Q4_0 | GgmlType::IQ4_XS | GgmlType::IQ3_S | GgmlType::Q4_K
                      | GgmlType::Q5_K | GgmlType::Q6_K | GgmlType::Q8_0)
    }

    /// Entry for `n_rows` and its rows per wave — BNR_R* in
    /// kernels/matvec_batched_nr_entries.h and the per-dtype overrides in
    /// the kernel files. A workgroup is 4 waves.
    fn entry(dtype: GgmlType, n_rows: usize) -> (&'static str, u32) {
        let (entry, rows) = match n_rows { 1 => ("n1", 2), 2 => ("n2", 4), 3 => ("n3", 4),
                                           4 => ("n4", 4), _ => ("w8", 2) };
        let rows = match (dtype, entry) {
            (GgmlType::Q5_K, "n4") | (GgmlType::Q6_K, "n3" | "n4") => 2,
            _ => rows,
        };
        (entry, rows)
    }

    /// `w`: the repacked weight; `xq`: BlockQ8 [n_rows, in_dim / 32];
    /// `y`: f32 [n_rows, out_dim].
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn launch(&self, stream: &hip::Stream, w: *mut c_void, dtype: GgmlType,
                         xq: *mut c_void, y: *mut c_void,
                         in_dim: usize, out_dim: usize, n_rows: usize) -> Result<(), String> {
        if !(1..=MAX_BATCHED_ROWS).contains(&n_rows) {
            return Err(format!("SmallBatchMatvec: n_rows {n_rows} not in 1..={MAX_BATCHED_ROWS}"));
        }
        let (module, prefix) = match dtype {
            GgmlType::Q4_K   => (&self.q4k,   "matvec_q4k_repacked_batched"),
            GgmlType::Q5_K   => (&self.q5k,   "matvec_q5k_repacked_batched"),
            GgmlType::Q6_K   => (&self.q6k,   "matvec_q6k_repacked_batched"),
            GgmlType::Q4_0   => (&self.q4_0,  "matvec_q4_0_repacked_batched"),
            GgmlType::Q8_0   => (&self.q8_0,  "matvec_q8_0_repacked_batched"),
            GgmlType::IQ4_XS => (&self.iq4xs, "matvec_iq4xs_repacked_batched"),
            GgmlType::IQ3_S  => (&self.iq3s,  "matvec_iq4xs_repacked_batched"),
            other => return Err(format!("SmallBatchMatvec: unsupported {other:?}")),
        };
        let (entry, rows) = Self::entry(dtype, n_rows);
        let f = module.function(&format!("{prefix}_{entry}_f32"))?;
        let (mut wp, mut xp, mut yp) = (w, xq, y);
        let (mut ia, mut oa, mut nr) = (in_dim as u32, out_dim as u32, n_rows as u32);
        let mut args: [*mut c_void; 6] = [
            &mut wp as *mut _ as *mut c_void, &mut xp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void, &mut ia as *mut _ as *mut c_void,
            &mut oa as *mut _ as *mut c_void, &mut nr as *mut _ as *mut c_void];
        let grid_x = (out_dim as u32).div_ceil(4 * rows);
        unsafe { f.launch((grid_x, 1, 1), (256, 1, 1), 0, Some(stream), &mut args) }
    }
}
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
const MV_Q8_0_REPACKED_BATCHED_SOURCE: &str =
    include_str!("../../kernels/matvec_q8_0_repacked_batched.cpp");

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
    gemm_f32_tn:   Module,
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
    small: SmallBatchMatvec,   // K=1..8 small-batch matvecs for verify
    w_f16:  std::cell::RefCell<DeviceBuf<u16>>,   // dequantised weight
    xq8:    std::cell::RefCell<DeviceBuf<u8>>,    // int8 activations (MMQ path)
    splitk: std::cell::RefCell<DeviceBuf<f32>>,   // F32 GEMM split-K partials
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
            gemm_f32_tn: Module::load(&cache.compile("gemm_f32_tn", GEMM_F32_TN_SOURCE)?)?,
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
            small: SmallBatchMatvec::new(cache)?,
            w_f16:  std::cell::RefCell::new(DeviceBuf::new(max_w.max(1))?),
            // int8 activations: one BlockQ8 (40 B) per 32-element sub-block.
            xq8:    std::cell::RefCell::new(DeviceBuf::new((max_x.max(32) / 32) * 40)?),
            splitk: std::cell::RefCell::new(DeviceBuf::new(1)?),
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
        if repacked && SmallBatchMatvec::supports(dtype) {
            if n_rows >= 1 && n_rows <= MAX_BATCHED_ROWS {
                // K=1..8 small-batch: a per-dtype matvec that reads each
                // weight sub-block once and dots it against all n_rows
                // activation rows (kernels/matvec_batched_nr.h); above
                // MAX_BATCHED_ROWS the narrow MMQ tile wins.
                //
                // K=1 used to fall through to MMQ (BN=64 row tile wastes
                // 98% of each workgroup → 390 ms verify(K=1) on 31B).
                //
                // A/B history on this path at K=4 (env-gated paths since
                // removed): MMQ was ~3.3x slower because BN=64 wastes 94%
                // of each tile; dequant→HGEMM was ~20x slower (per-call
                // Q4_K dequant of [5376, 21504] is ~314us; 180 FFN GEMMs
                // per verify = ~56ms of pure dequant). gfx906 has no
                // tensor cores, so fp16 GEMM has no compute advantage —
                // dp4a does 4 int8 multiplies per cycle vs fp16's 2, and
                // reads ¼ the weight bytes. dp4a wins both axes.
                return self.matmul_small_batch_into(stream, dst, w_dev, dtype,
                                                       in_dim, out_dim, x, n_rows);
            }
            return self.matmul_mmq_into(stream, dst, w_dev, dtype, in_dim, out_dim, x, n_rows);
        }

        // F32 weights (Gemma's per-layer-embedding projection, Unsloth's
        // unquantized small tensors): a tiled F32 GEMM off the weight.
        if dtype == GgmlType::F32 && !repacked && std::env::var_os("REINSTINCT_NO_F32_GEMM").is_none() {
            let splits = gemm_f32_splits(in_dim, out_dim, n_rows);
            if splits > 1 && std::env::var_os("REINSTINCT_NO_F32_SPLITK").is_none() {
                Self::grow(&self.splitk, splits * n_rows * out_dim, stream)?;
                return launch_gemm_f32_tn_split(&self.gemm_f32_tn, stream, w_dev.raw_ptr(), x, dst,
                                                in_dim, out_dim, n_rows, splits,
                                                self.splitk.borrow().raw_ptr());
            }
            return launch_gemm_f32_tn(&self.gemm_f32_tn, stream, w_dev.raw_ptr(), x, dst,
                                      in_dim, out_dim, n_rows);
        }

        crate::runtime::fallback_once(&format!("prefill-gemm {dtype:?} repacked={repacked}"),
            || "no int8 GEMM / small-batch matvec / F32 GEMM for this weight; dequant to fp16 per call + gemm_f16_rows".into());
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

    fn grow<T: Copy>(buf: &std::cell::RefCell<DeviceBuf<T>>, n: usize, stream: &hip::Stream)
        -> Result<(), String>
    {
        if buf.borrow().len() < n {
            stream.synchronize()?;          // old buffer may still be in flight
            *buf.borrow_mut() = DeviceBuf::new(n)?;
        }
        Ok(())
    }
    fn matmul_small_batch_into(&self, stream: &hip::Stream, dst: *mut c_void,
                                  w_dev: &DeviceBuf<u8>, dtype: GgmlType,
                                  in_dim: usize, out_dim: usize,
                                  x: *mut c_void, n_rows: usize)
        -> Result<(), String>
    {
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

        // 2. Small-batch matvec.
        self.small.launch(stream, w_dev.raw_ptr(), dtype, xq8.raw_ptr(), dst, in_dim, out_dim, n_rows)
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
            (GgmlType::Q8_0, false) => (&self.mmq_q8_0, "mmq_gemm_q8_0_rp_f32"),
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

    /// Small-batch (spec-decode verify) matmul cost: per dtype, at the
    /// 31B's FFN shapes, time of `matmul_into_raw` with n activation rows
    /// (quantize + batched matvec / MMQ, as verify runs it) and its ratio
    /// to n = 1. An HBM-bound weight stream would keep r(n) near 1.
    /// REINSTINCT_BMM_BENCH_SHAPES="in x out,..." overrides the shapes.
    /// `--ignored --nocapture`.
    #[test]
    #[ignore]
    fn bench_small_batch_matmul() {
        let Some(()) = crate::test_support::gpu() else { return };
        let _dev = hip::Device::set(0).unwrap();
        let cache = crate::runtime::KernelCache::new().unwrap();
        let shapes: Vec<(usize, usize)> = std::env::var("REINSTINCT_BMM_BENCH_SHAPES").ok()
            .map(|g| g.split(',').map(|p| { let v: Vec<usize> = p.split('x').map(|x| x.trim().parse().unwrap()).collect(); (v[0], v[1]) }).collect())
            .unwrap_or(vec![(5376, 21504), (21504, 5376), (5376, 8192)]);
        let max_rows = 16usize;
        let stream = hip::Stream::new().unwrap();
        let mut seed: u64 = 0x5BA7_C4ED;
        let mut rng_u8 = || -> u8 { seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (seed >> 56) as u8 };
        for &(in_dim, out_dim) in &shapes {
            let x: Vec<f32> = (0..max_rows * in_dim).map(|i| ((i * 37 % 101) as f32) * 0.01 - 0.5).collect();
            let dx = DeviceBuf::from_slice(&x).unwrap();
            let dy: DeviceBuf<f32> = DeviceBuf::new(max_rows * out_dim).unwrap();
            let gemm = PrefillGemm::new(&cache, out_dim * in_dim, max_rows * in_dim, max_rows * out_dim).unwrap();
            for dtype in [GgmlType::Q4_0, GgmlType::Q4_K, GgmlType::Q5_K, GgmlType::Q6_K, GgmlType::Q8_0] {
                if let Ok(only) = std::env::var("REINSTINCT_BMM_BENCH_DTYPES") {
                    if !only.split(',').any(|d| d.eq_ignore_ascii_case(&format!("{dtype:?}"))) { continue; }
                }
                let (bs, bpb) = (dtype.block_size_elements() as usize, dtype.bytes_per_block() as usize);
                let n_blocks = out_dim * (in_dim / bs);
                let mut w = vec![0u8; n_blocks * bpb];
                for b in &mut w { *b = rng_u8(); }
                for blk in 0..n_blocks {
                    let off = blk * bpb;
                    let d_off = if dtype == GgmlType::Q6_K { off + bpb - 2 } else { off };
                    w[d_off..d_off + 2].copy_from_slice(&crate::quant::half::f32_to_f16(0.001).to_le_bytes());
                    if matches!(dtype, GgmlType::Q4_K | GgmlType::Q5_K) {
                        w[off + 2..off + 4].copy_from_slice(&crate::quant::half::f32_to_f16(0.001).to_le_bytes());
                    }
                }
                let t = crate::runtime::qwen35::GpuMatvecTensor::from_bytes_matvec(&w, dtype, in_dim as u32, out_dim as u32).unwrap();
                let mut line = format!("{in_dim}x{out_dim} {dtype:?}:");
                let mut t1 = 0f64;
                for n in [1usize, 2, 3, 4, 8, 16] {
                    let run = || gemm.matmul_into_raw(&stream, dy.raw_ptr(), &t.data, t.dtype, t.repacked,
                                                      in_dim, out_dim, dx.raw_ptr(), n).unwrap();
                    for _ in 0..10 { run(); }
                    stream.synchronize().unwrap();
                    // Median of 7 timed batches of 40.
                    let (e0, e1) = (hip::Event::new().unwrap(), hip::Event::new().unwrap());
                    let mut reps: Vec<f64> = (0..7).map(|_| {
                        e0.record(&stream).unwrap();
                        for _ in 0..40 { run(); }
                        e1.record(&stream).unwrap(); e1.synchronize().unwrap();
                        hip::Event::elapsed_time(&e0, &e1).unwrap() as f64 / 40.0
                    }).collect();
                    reps.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let ms = reps[3];
                    if n == 1 { t1 = ms; }
                    line += &format!("  n{n} {ms:.3}ms (r {:.2})", ms / t1);
                }
                eprintln!("{line}");
            }
        }
    }

    /// `matmul_into` must agree with the CPU dequant oracle at every row
    /// count that changes which kernel runs: 1..4 the exact-row small-batch
    /// matvecs, 5..8 the masked 8-row one, 9..16 the narrow MMQ tile and
    /// >16 the wide one. Two shapes: a power-of-two n_sub below one 64-sub-
    /// block chunk, and n_sub = 136 (a chunk tail) with out_dim = 100 (a
    /// partial workgroup of clamped rows at every ROWS).
    #[test]
    fn matmul_into_matches_oracle_across_row_counts() {
        let Some(cache) = crate::test_support::kernel_cache() else { return };
        for (in_dim, out_dim) in [(1024usize, 256usize), (4352, 100)] {
            check_matmul_into_oracle(&cache, in_dim, out_dim);
        }
    }

    fn check_matmul_into_oracle(cache: &crate::runtime::KernelCache, in_dim: usize, out_dim: usize) {
        let _dev = hip::Device::set(0).unwrap();
        let stream = hip::Stream::new().expect("stream");
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

        let gemm = PrefillGemm::new(cache, out_dim * in_dim,
                                    max_rows * in_dim, max_rows * out_dim).unwrap();

        // Every repacked dtype the GEMM dispatch can see.
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

            let poison = vec![f32::NAN; max_rows * out_dim];
            for &n_rows in &[1usize, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 20] {
                // Every row this call should write starts as NaN, so an
                // unwritten one fails rather than passing on a stale copy.
                dst.copy_from_host(&poison).unwrap();
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
                eprintln!("matmul_into {in_dim}x{out_dim} {dtype:?} rows={n_rows}: rel_l2={e:.3e}");
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

    /// The F32 GEMM, plain and split-K, against an f64 CPU reference.
    /// Shapes: the GDN alpha/beta and MoE router (split 16 / 8), a K that
    /// is not a multiple of the 16-wide slice, and an odd K (scalar path).
    #[test]
    fn gemm_f32_tn_split_matches_reference() {
        let Some(cache) = crate::test_support::kernel_cache() else { return };
        let _dev = hip::Device::set(0).unwrap();
        let stream = hip::Stream::new().unwrap();
        let module = Module::load(&cache.compile("gemm_f32_tn", GEMM_F32_TN_SOURCE).unwrap()).unwrap();
        let mut xs: u64 = 0xF32_5EED;
        let mut rng = || { xs = xs.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                           ((xs >> 33) as u32 as f32 / u32::MAX as f32) - 0.5 };
        for (in_dim, out_dim, n_rows) in [(2048usize, 32usize, 512usize), (2048, 256, 256), (1000, 37, 70), (1001, 64, 9)] {
            let w: Vec<f32> = (0..out_dim * in_dim).map(|_| rng()).collect();
            let x: Vec<f32> = (0..n_rows * in_dim).map(|_| rng()).collect();
            let (dw, dx) = (DeviceBuf::from_slice(&w).unwrap(), DeviceBuf::from_slice(&x).unwrap());
            let dy: DeviceBuf<f32> = DeviceBuf::from_slice(&vec![f32::NAN; n_rows * out_dim]).unwrap();
            let mut want = vec![0f64; n_rows * out_dim];
            for r in 0..n_rows { for o in 0..out_dim {
                want[r * out_dim + o] = (0..in_dim).map(|k| w[o * in_dim + k] as f64 * x[r * in_dim + k] as f64).sum();
            }}
            for splits in [1usize, 2, 8, 16] {
                if splits == 1 {
                    launch_gemm_f32_tn(&module, &stream, dw.raw_ptr(), dx.raw_ptr(), dy.raw_ptr(),
                                       in_dim, out_dim, n_rows).unwrap();
                } else {
                    let scratch: DeviceBuf<f32> = DeviceBuf::new(splits * n_rows * out_dim).unwrap();
                    launch_gemm_f32_tn_split(&module, &stream, dw.raw_ptr(), dx.raw_ptr(), dy.raw_ptr(),
                                             in_dim, out_dim, n_rows, splits, scratch.raw_ptr()).unwrap();
                }
                stream.synchronize().unwrap();
                let mut got = vec![0f32; n_rows * out_dim];
                dy.copy_to_host(&mut got).unwrap();
                let worst = got.iter().zip(&want).map(|(&g, &e)| (g as f64 - e).abs()).fold(0f64, f64::max);
                assert!(worst < 1e-4 * (in_dim as f64).sqrt(),
                        "{in_dim}x{out_dim} n={n_rows} splits={splits}: max abs err {worst}");
            }
        }
        assert_eq!(gemm_f32_splits(2048, 32, 512), 16);
        assert_eq!(gemm_f32_splits(2048, 256, 256), 8);
        assert_eq!(gemm_f32_splits(2048, 4096, 512), 1);
    }
}
