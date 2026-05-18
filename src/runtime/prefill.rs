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
use crate::hip::rocblas::{self, Handle, RocblasOp};
use super::KernelCache;

const CVT_SOURCE: &str = include_str!("../../kernels/cvt_f32_f16.cpp");
const QUANTIZE_Q8_SOURCE: &str = include_str!("../../kernels/quantize_q8.cpp");
const MMQ_GEMM_Q4K_SOURCE: &str =
    include_str!("../../kernels/mmq_gemm_q4k_repacked.cpp");

/// Bulk-dequantize a quantized weight tensor to an fp16 device buffer.
/// `n_elements` is the logical weight count (out_dim * in_dim).
fn dequant_to_f16(cache: &KernelCache, w_bytes: &[u8], dtype: GgmlType, n_elements: usize)
    -> Result<DeviceBuf<u16>, String>
{
    // F16 weights need no dequant — the bytes are already fp16.
    if dtype == GgmlType::F16 {
        assert_eq!(w_bytes.len(), n_elements * 2, "F16 byte count mismatch");
        let words: &[u16] = bytemuck::cast_slice(w_bytes);
        return DeviceBuf::from_slice(words);
    }

    let (src, kname, weights_per_block, block_threads): (&str, &str, usize, u32) = match dtype {
        GgmlType::Q4_K   => (include_str!("../../kernels/dequant_q4_k_f16.cpp"),
                             "dequant_q4_k_f16", 256, 256),
        GgmlType::Q5_K   => (include_str!("../../kernels/dequant_q5_k_f16.cpp"),
                             "dequant_q5_k_f16", 256, 256),
        GgmlType::Q6_K   => (include_str!("../../kernels/dequant_q6_k_f16.cpp"),
                             "dequant_q6_k_f16", 256, 256),
        GgmlType::Q8_0   => (include_str!("../../kernels/dequant_q8_0_f16.cpp"),
                             "dequant_q8_0_f16", 32, 32),
        GgmlType::IQ4_XS => (include_str!("../../kernels/dequant_iq4_xs_f16.cpp"),
                             "dequant_iq4_xs_f16", 256, 256),
        other => return Err(format!("dequant_to_f16: unsupported dtype {other:?}")),
    };
    assert_eq!(n_elements % weights_per_block, 0,
        "n_elements not a multiple of block size");
    let n_blocks = n_elements / weights_per_block;

    let hsaco = cache.compile(kname, src)?;
    let module = Module::load(&hsaco)?;
    let f = module.function(kname)?;

    let dw: DeviceBuf<u8>  = DeviceBuf::from_slice(w_bytes)?;
    let out: DeviceBuf<u16> = DeviceBuf::new(n_elements)?;
    let mut w_ptr = dw.raw_ptr();
    let mut o_ptr = out.raw_ptr();
    let mut nb = n_blocks as u32;
    let mut args: [*mut c_void; 3] = [
        &mut w_ptr as *mut _ as *mut c_void,
        &mut o_ptr as *mut _ as *mut c_void,
        &mut nb    as *mut _ as *mut c_void,
    ];
    unsafe { f.launch((n_blocks as u32, 1, 1), (block_threads, 1, 1), 0, None, &mut args)?; }
    hip::Device(0).synchronize()?;
    Ok(out)
}

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

/// `Y = X · Wᵀ` via rocBLAS HGEMM.
///
/// - `w_bytes` / `dtype`: the on-disk quantized weight, logical shape
///   `[out_dim, in_dim]` row-major (one output row per `j`).
/// - `x`: activations, `[n_rows, in_dim]` row-major, fp32.
/// - returns `Y` `[n_rows, out_dim]` row-major, fp32.
///
/// Internally: dequant W→fp16, X→fp16, HGEMM, Y→fp32. The HGEMM is set
/// up so the column-major result lands directly in row-major `[n_rows,
/// out_dim]` order (see the layout derivation in the body).
pub fn batched_matmul(cache: &KernelCache, handle: &Handle,
                      w_bytes: &[u8], dtype: GgmlType,
                      x: &[f32], n_rows: usize, in_dim: usize, out_dim: usize)
    -> Result<Vec<f32>, String>
{
    assert_eq!(x.len(), n_rows * in_dim, "x shape mismatch");
    let w_dev: DeviceBuf<u8> = DeviceBuf::from_slice(w_bytes)?;
    let x_dev: DeviceBuf<f32> = DeviceBuf::from_slice(x)?;
    let y_dev = batched_matmul_resident(cache, handle, &w_dev, dtype,
                                        in_dim, out_dim, &x_dev, n_rows)?;
    let mut out = vec![0.0f32; n_rows * out_dim];
    y_dev.copy_to_host(&mut out)?;
    Ok(out)
}

/// Device-resident `Y = X · Wᵀ`: weights stay quantized on device, X/Y
/// are device fp32. Internally dequant W→fp16, X→fp16, fp32-accumulate
/// HGEMM, Y→fp32 — the building block of the batched prefill forward.
pub fn batched_matmul_resident(cache: &KernelCache, handle: &Handle,
                               w_dev: &DeviceBuf<u8>, dtype: GgmlType,
                               in_dim: usize, out_dim: usize,
                               x: &DeviceBuf<f32>, n_rows: usize)
    -> Result<DeviceBuf<f32>, String>
{
    // 1. Dequant W → fp16 [out_dim, in_dim].
    let w_f16 = dequant_dev_to_f16(cache, w_dev, dtype, out_dim * in_dim)?;

    // 2. X → fp16 [n_rows, in_dim].
    let cvt_module = Module::load(&cache.compile("cvt_f32_f16", CVT_SOURCE)?)?;
    let to_f16 = cvt_module.function("cvt_f32_to_f16")?;
    let to_f32 = cvt_module.function("cvt_f16_to_f32")?;
    let cvt = |f: &crate::hip::Function, src: *mut c_void, dst: *mut c_void, n: u32|
        -> Result<(), String> {
        let block: u32 = 256;
        let mut i=src; let mut o=dst; let mut na=n;
        let mut args: [*mut c_void; 3] = [
            &mut i as *mut _ as *mut c_void, &mut o as *mut _ as *mut c_void,
            &mut na as *mut _ as *mut c_void];
        unsafe { f.launch(((n+block-1)/block,1,1),(block,1,1),0,None,&mut args) }
    };
    let dx_f16: DeviceBuf<u16> = DeviceBuf::new(n_rows * in_dim)?;
    cvt(&to_f16, x.raw_ptr(), dx_f16.raw_ptr(), (n_rows * in_dim) as u32)?;

    // 3. HGEMM. W row-major [out,in] == col-major [in,out]; X r-m [rows,in]
    //    == c-m [in,rows]; col-major C = Wᵀ·X [out,rows] == Y r-m [rows,out].
    //    transA=T, transB=N, m=out, n=rows, k=in, lda=ldb=in, ldc=out.
    let dy_f16: DeviceBuf<u16> = DeviceBuf::new(n_rows * out_dim)?;
    hip::Device(0).synchronize()?;
    unsafe {
        handle.gemm_f16_f32acc(
            RocblasOp::Transpose, RocblasOp::None,
            out_dim as i32, n_rows as i32, in_dim as i32,
            1.0,
            w_f16.as_ptr() as *const c_void, in_dim as i32,
            dx_f16.as_ptr() as *const c_void, in_dim as i32,
            0.0,
            dy_f16.as_ptr() as *mut c_void, out_dim as i32,
        )?;
    }

    // 4. Y fp16 → fp32.
    let dy_f32: DeviceBuf<f32> = DeviceBuf::new(n_rows * out_dim)?;
    cvt(&to_f32, dy_f16.raw_ptr(), dy_f32.raw_ptr(), (n_rows * out_dim) as u32)?;
    hip::Device(0).synchronize()?;
    let _ = rocblas::rocblas;
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
    deq_q4k:   Module,
    deq_q5k:   Module,
    deq_q6k:   Module,
    deq_q8_0:  Module,
    deq_iq4xs: Module,
    deq_q4k_repacked: Module,
    deq_q5k_repacked: Module,
    deq_q6k_repacked: Module,
    quantize_q8: Module,
    mmq_q4k:     Module,
    w_f16:  std::cell::RefCell<DeviceBuf<u16>>,   // dequantised weight
    dx_f16: std::cell::RefCell<DeviceBuf<u16>>,   // fp16 activations
    dy_f16: std::cell::RefCell<DeviceBuf<u16>>,   // fp16 GEMM output
    xq8:    std::cell::RefCell<DeviceBuf<u8>>,    // int8 activations (MMQ path)
}

impl PrefillGemm {
    /// Pre-size the scratch to the largest weight/activation/output the
    /// caller will pass. Buffers still grow on demand as a safety net.
    pub fn new(cache: &KernelCache, max_w: usize, max_x: usize, max_y: usize)
        -> Result<Self, String>
    {
        Ok(Self {
            cvt:       Module::load(&cache.compile("cvt_f32_f16", CVT_SOURCE)?)?,
            deq_q4k:   Module::load(&cache.compile("dequant_q4_k_f16",
                           include_str!("../../kernels/dequant_q4_k_f16.cpp"))?)?,
            deq_q5k:   Module::load(&cache.compile("dequant_q5_k_f16",
                           include_str!("../../kernels/dequant_q5_k_f16.cpp"))?)?,
            deq_q6k:   Module::load(&cache.compile("dequant_q6_k_f16",
                           include_str!("../../kernels/dequant_q6_k_f16.cpp"))?)?,
            deq_q8_0:  Module::load(&cache.compile("dequant_q8_0_f16",
                           include_str!("../../kernels/dequant_q8_0_f16.cpp"))?)?,
            deq_iq4xs: Module::load(&cache.compile("dequant_iq4_xs_f16",
                           include_str!("../../kernels/dequant_iq4_xs_f16.cpp"))?)?,
            deq_q4k_repacked: Module::load(&cache.compile("dequant_q4k_repacked_f16",
                           include_str!("../../kernels/dequant_q4k_repacked_f16.cpp"))?)?,
            deq_q5k_repacked: Module::load(&cache.compile("dequant_q5k_repacked_f16",
                           include_str!("../../kernels/dequant_q5k_repacked_f16.cpp"))?)?,
            deq_q6k_repacked: Module::load(&cache.compile("dequant_q6k_repacked_f16",
                           include_str!("../../kernels/dequant_q6k_repacked_f16.cpp"))?)?,
            quantize_q8: Module::load(&cache.compile("quantize_q8", QUANTIZE_Q8_SOURCE)?)?,
            mmq_q4k:     Module::load(&cache.compile("mmq_gemm_q4k_repacked",
                                                     MMQ_GEMM_Q4K_SOURCE)?)?,
            w_f16:  std::cell::RefCell::new(DeviceBuf::new(max_w.max(1))?),
            dx_f16: std::cell::RefCell::new(DeviceBuf::new(max_x.max(1))?),
            dy_f16: std::cell::RefCell::new(DeviceBuf::new(max_y.max(1))?),
            // int8 activations: one BlockQ8 (40 B) per 32-element sub-block.
            xq8:    std::cell::RefCell::new(DeviceBuf::new((max_x.max(32) / 32) * 40)?),
        })
    }

    fn deq(&self, dt: GgmlType) -> Result<(&Module, &'static str, usize, u32), String> {
        Ok(match dt {
            GgmlType::Q4_K   => (&self.deq_q4k,   "dequant_q4_k_f16",   256, 256),
            GgmlType::Q5_K   => (&self.deq_q5k,   "dequant_q5_k_f16",   256, 256),
            GgmlType::Q6_K   => (&self.deq_q6k,   "dequant_q6_k_f16",   256, 256),
            GgmlType::Q8_0   => (&self.deq_q8_0,  "dequant_q8_0_f16",    32,  32),
            GgmlType::IQ4_XS => (&self.deq_iq4xs, "dequant_iq4_xs_f16", 256, 256),
            o => return Err(format!("PrefillGemm: unsupported weight dtype {o:?}")),
        })
    }

    /// Device-resident `Y = X · Wᵀ`, all kernels ordered on `stream` —
    /// no internal device syncs, no per-call module loads or weight
    /// allocations. Only `Y` (`[n_rows, out_dim]` fp32) is freshly
    /// allocated; the fp16 scratch is pooled.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    pub fn matmul(&self, handle: &Handle, stream: &hip::Stream,
                  w_dev: &DeviceBuf<u8>, dtype: GgmlType, repacked: bool,
                  in_dim: usize, out_dim: usize,
                  x: &DeviceBuf<f32>, n_rows: usize)
        -> Result<DeviceBuf<f32>, String>
    {
        // Repacked Q4_K: the 2D-tiled int8 MMQ GEMM consumes the
        // quantised weight directly — no dequant to fp16, no HGEMM.
        if repacked && dtype == GgmlType::Q4_K {
            return self.matmul_mmq_q4k(stream, w_dev, in_dim, out_dim, x, n_rows);
        }

        let n_w = out_dim * in_dim;
        let n_x = n_rows * in_dim;
        let n_y = n_rows * out_dim;
        Self::grow(&self.w_f16,  n_w, stream)?;
        Self::grow(&self.dx_f16, n_x, stream)?;
        Self::grow(&self.dy_f16, n_y, stream)?;
        let w_f16  = self.w_f16.borrow();
        let dx_f16 = self.dx_f16.borrow();
        let dy_f16 = self.dy_f16.borrow();

        // 1. Dequant W → fp16 scratch (in place reuse).
        let mut w_ptr = w_dev.raw_ptr();
        let mut o_ptr = w_f16.raw_ptr();
        if repacked {
            // Repacked K-quant: one HIP block per 32-weight sub-block.
            let (module, kname) = match dtype {
                GgmlType::Q5_K => (&self.deq_q5k_repacked, "dequant_q5k_repacked_f16"),
                GgmlType::Q6_K => (&self.deq_q6k_repacked, "dequant_q6k_repacked_f16"),
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

        // 2. X → fp16 scratch.
        let to_f16 = self.cvt.function("cvt_f32_to_f16")?;
        let to_f32 = self.cvt.function("cvt_f16_to_f32")?;
        let cvt = |f: &crate::hip::Function, src: *mut c_void, dst: *mut c_void, n: u32|
            -> Result<(), String> {
            let block: u32 = 256;
            let mut i=src; let mut o=dst; let mut na=n;
            let mut args: [*mut c_void; 3] = [
                &mut i as *mut _ as *mut c_void, &mut o as *mut _ as *mut c_void,
                &mut na as *mut _ as *mut c_void];
            unsafe { f.launch(((n+block-1)/block,1,1),(block,1,1),0,Some(stream),&mut args) }
        };
        cvt(&to_f16, x.raw_ptr(), dx_f16.raw_ptr(), n_x as u32)?;

        // 3. HGEMM (see batched_matmul_resident for the layout derivation).
        unsafe {
            handle.gemm_f16_f32acc(
                RocblasOp::Transpose, RocblasOp::None,
                out_dim as i32, n_rows as i32, in_dim as i32,
                1.0,
                w_f16.as_ptr() as *const c_void,  in_dim as i32,
                dx_f16.as_ptr() as *const c_void, in_dim as i32,
                0.0,
                dy_f16.as_ptr() as *mut c_void,   out_dim as i32,
            )?;
        }

        // 4. Y fp16 → fresh fp32.
        let dy_f32: DeviceBuf<f32> = DeviceBuf::new(n_y)?;
        cvt(&to_f32, dy_f16.raw_ptr(), dy_f32.raw_ptr(), n_y as u32)?;
        Ok(dy_f32)
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

    /// Device-resident `Y = X · Wᵀ` for a repacked Q4_K weight via the
    /// int8 MMQ GEMM: quantise X → BlockQ8, then one dp4a GEMM straight
    /// off the quantised weight. All kernels ordered on `stream`.
    fn matmul_mmq_q4k(&self, stream: &hip::Stream, w_dev: &DeviceBuf<u8>,
                      in_dim: usize, out_dim: usize,
                      x: &DeviceBuf<f32>, n_rows: usize)
        -> Result<DeviceBuf<f32>, String>
    {
        let n_xq8 = (n_rows * in_dim / 32) * 40;          // BlockQ8 bytes
        if self.xq8.borrow().len() < n_xq8 {
            stream.synchronize()?;
            *self.xq8.borrow_mut() = DeviceBuf::new(n_xq8)?;
        }
        let xq8 = self.xq8.borrow();

        // 1. Quantise X → BlockQ8 [n_rows, in_dim/32] (grid.y = row).
        let qf = self.quantize_q8.function("quantize_q8_f32")?;
        let mut xp = x.raw_ptr(); let mut qp = xq8.raw_ptr();
        let mut ind = in_dim as u32;
        let mut qa: [*mut c_void; 3] = [
            &mut xp as *mut _ as *mut c_void, &mut qp as *mut _ as *mut c_void,
            &mut ind as *mut _ as *mut c_void];
        unsafe { qf.launch((((in_dim as u32) + 255) / 256, n_rows as u32, 1),
                           (256, 1, 1), 0, Some(stream), &mut qa)?; }

        // 2. MMQ GEMM → fresh fp32 Y [n_rows, out_dim].
        let dy: DeviceBuf<f32> = DeviceBuf::new(n_rows * out_dim)?;
        let gf = self.mmq_q4k.function("mmq_gemm_q4k_repacked_f32")?;
        let mut wp = w_dev.raw_ptr(); let mut xqp = xq8.raw_ptr(); let mut yp = dy.raw_ptr();
        let mut ia = in_dim as u32; let mut oa = out_dim as u32; let mut pa = n_rows as u32;
        let mut ga: [*mut c_void; 6] = [
            &mut wp as *mut _ as *mut c_void, &mut xqp as *mut _ as *mut c_void,
            &mut yp as *mut _ as *mut c_void, &mut ia as *mut _ as *mut c_void,
            &mut oa as *mut _ as *mut c_void, &mut pa as *mut _ as *mut c_void];
        // grid: 8 output rows × TN=32 tokens per workgroup.
        // grid: BM=64 output rows × BN=64 tokens per workgroup.
        unsafe { gf.launch(((out_dim as u32 + 63) / 64, (n_rows as u32 + 63) / 64, 1),
                           (256, 1, 1), 0, Some(stream), &mut ga)?; }
        Ok(dy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hip::Stream;

    #[test]
    fn batched_matmul_matches_sequential_q4_k() {
        if hip::device_count().ok().unwrap_or(0) < 1 { eprintln!("skip: no HIP"); return; }
        let _dev = hip::Device::set(0).unwrap();
        let cache = match KernelCache::new() {
            Ok(c) => c, Err(e) => { eprintln!("skip: {e}"); return; }
        };
        let stream = Stream::new().expect("stream");
        let handle = match Handle::new() {
            Ok(h) => h, Err(e) => { eprintln!("skip: rocblas: {e}"); return; }
        };
        handle.set_stream(&stream).expect("set_stream");

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

        // Batched HGEMM result.
        let gpu = batched_matmul(&cache, &handle, &w, GgmlType::Q4_K,
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
