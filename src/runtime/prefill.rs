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

    // 1. Dequant W → fp16 [out_dim, in_dim].
    let w_f16 = dequant_to_f16(cache, w_bytes, dtype, out_dim * in_dim)?;

    // 2. Convert X → fp16 [n_rows, in_dim].
    let cvt_hsaco = cache.compile("cvt_f32_f16", CVT_SOURCE)?;
    let cvt_module = Module::load(&cvt_hsaco)?;
    let to_f16 = cvt_module.function("cvt_f32_to_f16")?;
    let to_f32 = cvt_module.function("cvt_f16_to_f32")?;

    let dx_f32: DeviceBuf<f32> = DeviceBuf::from_slice(x)?;
    let dx_f16: DeviceBuf<u16> = DeviceBuf::new(n_rows * in_dim)?;
    {
        let block: u32 = 256;
        let n = (n_rows * in_dim) as u32;
        let grid = (n + block - 1) / block;
        let mut i_ptr = dx_f32.raw_ptr();
        let mut o_ptr = dx_f16.raw_ptr();
        let mut n_arg = n;
        let mut args: [*mut c_void; 3] = [
            &mut i_ptr as *mut _ as *mut c_void,
            &mut o_ptr as *mut _ as *mut c_void,
            &mut n_arg as *mut _ as *mut c_void,
        ];
        unsafe { to_f16.launch((grid, 1, 1), (block, 1, 1), 0, None, &mut args)?; }
    }

    // 3. HGEMM. Layout derivation:
    //    W row-major [out_dim, in_dim]  == W_cm col-major [in_dim, out_dim]
    //    X row-major [n_rows, in_dim]   == X_cm col-major [in_dim, n_rows]
    //    want Y[n][j] = Σ_i X[n][i]·W[j][i] = Σ_i W_cm[i][j]·X_cm[i][n]
    //    col-major BLAS: C = Wᵀ·X  with C col-major [out_dim, n_rows]
    //                  = Y row-major [n_rows, out_dim]  (same bytes)
    //    so transA=T, transB=N, m=out_dim, n=n_rows, k=in_dim,
    //       lda=ldb=in_dim, ldc=out_dim.
    let dy_f16: DeviceBuf<u16> = DeviceBuf::new(n_rows * out_dim)?;
    hip::Device(0).synchronize()?;  // dequant + cvt must finish before GEMM
    // fp16 storage, fp32 accumulate — plain hgemm's fp16 accumulate loses
    // ~20% on a 2048-term reduction.
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
    {
        let block: u32 = 256;
        let n = (n_rows * out_dim) as u32;
        let grid = (n + block - 1) / block;
        let mut i_ptr = dy_f16.raw_ptr();
        let mut o_ptr = dy_f32.raw_ptr();
        let mut n_arg = n;
        let mut args: [*mut c_void; 3] = [
            &mut i_ptr as *mut _ as *mut c_void,
            &mut o_ptr as *mut _ as *mut c_void,
            &mut n_arg as *mut _ as *mut c_void,
        ];
        unsafe { to_f32.launch((grid, 1, 1), (block, 1, 1), 0, None, &mut args)?; }
    }
    hip::Device(0).synchronize()?;

    let mut out = vec![0.0f32; n_rows * out_dim];
    dy_f32.copy_to_host(&mut out)?;
    let _ = rocblas::rocblas;  // keep the symbol referenced
    Ok(out)
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
