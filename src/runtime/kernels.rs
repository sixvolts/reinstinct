//! HIP kernels used by the inference path, paired with per-op convenience
//! launchers. Each launcher allocates and copies per call — fine for
//! validation, but the real forward will reuse Module/Function handles
//! and pre-allocated device buffers.

use std::ffi::c_void;

use super::KernelCache;
use crate::hip::{self, DeviceBuf, Module};

const RMSNORM_SOURCE: &str = include_str!("../../kernels/rmsnorm.cpp");
const RMSNORM_KERNEL: &str = "rmsnorm_f32";

const EMBED_LOOKUP_SOURCE: &str = include_str!("../../kernels/embed_lookup.cpp");
const EMBED_LOOKUP_KERNEL: &str = "embed_lookup_f32";

const MATVEC_SOURCE: &str = include_str!("../../kernels/matvec.cpp");
const MATVEC_KERNEL: &str = "matvec_f32";

const MATVEC_Q8_0_SOURCE: &str = include_str!("../../kernels/matvec_q8_0.cpp");
const MATVEC_Q8_0_KERNEL: &str = "matvec_q8_0_f32";

const MATVEC_Q4_K_SOURCE: &str = include_str!("../../kernels/matvec_q4_k.cpp");
const MATVEC_Q4_K_KERNEL: &str = "matvec_q4_k_f32";

const MATVEC_Q6_K_SOURCE: &str = include_str!("../../kernels/matvec_q6_k.cpp");
const MATVEC_Q6_K_KERNEL: &str = "matvec_q6_k_f32";

const MATVEC_Q5_K_SOURCE: &str = include_str!("../../kernels/matvec_q5_k.cpp");
const MATVEC_Q5_K_KERNEL: &str = "matvec_q5_k_f32";

const MATVEC_IQ4_XS_SOURCE: &str = include_str!("../../kernels/matvec_iq4_xs.cpp");
const MATVEC_IQ4_XS_KERNEL: &str = "matvec_iq4_xs_f32";

const SWIGLU_SOURCE: &str = include_str!("../../kernels/swiglu.cpp");
const SWIGLU_KERNEL: &str = "swiglu_mul_f32";

/// `out[i] = silu(gate[i]) * up[i]` (Qwen / Llama FFN gate fusion).
pub fn swiglu_mul_f32(cache: &KernelCache, gate: &[f32], up: &[f32]) -> Result<Vec<f32>, String> {
    assert_eq!(gate.len(), up.len());
    let n = gate.len();

    let hsaco = cache.compile("swiglu", SWIGLU_SOURCE)?;
    let module = Module::load(&hsaco)?;
    let f = module.function(SWIGLU_KERNEL)?;

    let dg: DeviceBuf<f32> = DeviceBuf::from_slice(gate)?;
    let du: DeviceBuf<f32> = DeviceBuf::from_slice(up)?;
    let dy: DeviceBuf<f32> = DeviceBuf::new(n)?;

    let block: u32 = 256;
    let grid: u32 = (n as u32 + block - 1) / block;
    let mut g_ptr = dg.raw_ptr();
    let mut u_ptr = du.raw_ptr();
    let mut y_ptr = dy.raw_ptr();
    let mut n_arg = n as u32;
    let mut args: [*mut c_void; 4] = [
        &mut g_ptr  as *mut _ as *mut c_void,
        &mut u_ptr  as *mut _ as *mut c_void,
        &mut y_ptr  as *mut _ as *mut c_void,
        &mut n_arg  as *mut _ as *mut c_void,
    ];
    unsafe { f.launch((grid, 1, 1), (block, 1, 1), 0, None, &mut args)?; }
    hip::Device(0).synchronize()?;

    let mut out = vec![0.0f32; n];
    dy.copy_to_host(&mut out)?;
    Ok(out)
}

/// Fused IQ4_XS dequant + GEMV. 136 bytes per super-block; quants are
/// 4-bit indices into a fixed 16-entry non-uniform codebook
/// (KVALUES_IQ4NL), with per-sub-block 6-bit scale.
pub fn matvec_iq4_xs_f32(cache: &KernelCache, w_bytes: &[u8], x: &[f32],
                         in_dim: usize, out_dim: usize) -> Result<Vec<f32>, String>
{
    use crate::quant::iq4_xs::{BLOCK_SIZE, BYTES_PER_BLOCK};
    assert_eq!(in_dim % BLOCK_SIZE, 0, "in_dim must be a multiple of {}", BLOCK_SIZE);
    let n_blocks = in_dim / BLOCK_SIZE;
    let expect_bytes = out_dim * n_blocks * BYTES_PER_BLOCK;
    assert_eq!(w_bytes.len(), expect_bytes,
        "w_bytes len {} != expected {} ({}*{}*{})",
        w_bytes.len(), expect_bytes, out_dim, n_blocks, BYTES_PER_BLOCK);
    assert_eq!(x.len(), in_dim);

    let hsaco = cache.compile("matvec_iq4_xs", MATVEC_IQ4_XS_SOURCE)?;
    let module = Module::load(&hsaco)?;
    let f = module.function(MATVEC_IQ4_XS_KERNEL)?;

    let dw: DeviceBuf<u8>  = DeviceBuf::from_slice(w_bytes)?;
    let dx: DeviceBuf<f32> = DeviceBuf::from_slice(x)?;
    let dy: DeviceBuf<f32> = DeviceBuf::new(out_dim)?;

    let block: u32 = 256;
    let grid: u32 = out_dim as u32;
    let mut w_ptr = dw.raw_ptr();
    let mut x_ptr = dx.raw_ptr();
    let mut y_ptr = dy.raw_ptr();
    let mut in_arg = in_dim as u32;
    let mut out_arg = out_dim as u32;
    let mut args: [*mut c_void; 5] = [
        &mut w_ptr   as *mut _ as *mut c_void,
        &mut x_ptr   as *mut _ as *mut c_void,
        &mut y_ptr   as *mut _ as *mut c_void,
        &mut in_arg  as *mut _ as *mut c_void,
        &mut out_arg as *mut _ as *mut c_void,
    ];
    let smem_bytes = block * std::mem::size_of::<f32>() as u32;
    unsafe { f.launch((grid, 1, 1), (block, 1, 1), smem_bytes, None, &mut args)?; }
    hip::Device(0).synchronize()?;

    let mut out = vec![0.0f32; out_dim];
    dy.copy_to_host(&mut out)?;
    Ok(out)
}

/// Fused Q5_K dequant + GEMV. 176 bytes per super-block. Like Q4_K plus
/// a 32-byte qh array contributing the high bit of each 5-bit quant.
pub fn matvec_q5_k_f32(cache: &KernelCache, w_bytes: &[u8], x: &[f32],
                       in_dim: usize, out_dim: usize) -> Result<Vec<f32>, String>
{
    use crate::quant::q5_k::{BLOCK_SIZE, BYTES_PER_BLOCK};
    assert_eq!(in_dim % BLOCK_SIZE, 0, "in_dim must be a multiple of {}", BLOCK_SIZE);
    let n_blocks = in_dim / BLOCK_SIZE;
    let expect_bytes = out_dim * n_blocks * BYTES_PER_BLOCK;
    assert_eq!(w_bytes.len(), expect_bytes,
        "w_bytes len {} != expected {} ({}*{}*{})",
        w_bytes.len(), expect_bytes, out_dim, n_blocks, BYTES_PER_BLOCK);
    assert_eq!(x.len(), in_dim);

    let hsaco = cache.compile("matvec_q5_k", MATVEC_Q5_K_SOURCE)?;
    let module = Module::load(&hsaco)?;
    let f = module.function(MATVEC_Q5_K_KERNEL)?;

    let dw: DeviceBuf<u8>  = DeviceBuf::from_slice(w_bytes)?;
    let dx: DeviceBuf<f32> = DeviceBuf::from_slice(x)?;
    let dy: DeviceBuf<f32> = DeviceBuf::new(out_dim)?;

    let block: u32 = 256;
    let grid: u32 = out_dim as u32;
    let mut w_ptr = dw.raw_ptr();
    let mut x_ptr = dx.raw_ptr();
    let mut y_ptr = dy.raw_ptr();
    let mut in_arg = in_dim as u32;
    let mut out_arg = out_dim as u32;
    let mut args: [*mut c_void; 5] = [
        &mut w_ptr   as *mut _ as *mut c_void,
        &mut x_ptr   as *mut _ as *mut c_void,
        &mut y_ptr   as *mut _ as *mut c_void,
        &mut in_arg  as *mut _ as *mut c_void,
        &mut out_arg as *mut _ as *mut c_void,
    ];
    let smem_bytes = block * std::mem::size_of::<f32>() as u32;
    unsafe { f.launch((grid, 1, 1), (block, 1, 1), smem_bytes, None, &mut args)?; }
    hip::Device(0).synchronize()?;

    let mut out = vec![0.0f32; out_dim];
    dy.copy_to_host(&mut out)?;
    Ok(out)
}

/// Fused Q6_K dequant + GEMV. Symmetric format: w = d * sc * (q6 - 32),
/// where q6 is 6 bits split between ql (low 4) and qh (high 2). 210 bytes
/// per super-block / 256 weights.
pub fn matvec_q6_k_f32(cache: &KernelCache, w_bytes: &[u8], x: &[f32],
                       in_dim: usize, out_dim: usize) -> Result<Vec<f32>, String>
{
    use crate::quant::q6_k::{BLOCK_SIZE, BYTES_PER_BLOCK};
    assert_eq!(in_dim % BLOCK_SIZE, 0, "in_dim must be a multiple of {}", BLOCK_SIZE);
    let n_blocks = in_dim / BLOCK_SIZE;
    let expect_bytes = out_dim * n_blocks * BYTES_PER_BLOCK;
    assert_eq!(w_bytes.len(), expect_bytes,
        "w_bytes len {} != expected {} ({}*{}*{})",
        w_bytes.len(), expect_bytes, out_dim, n_blocks, BYTES_PER_BLOCK);
    assert_eq!(x.len(), in_dim);

    let hsaco = cache.compile("matvec_q6_k", MATVEC_Q6_K_SOURCE)?;
    let module = Module::load(&hsaco)?;
    let f = module.function(MATVEC_Q6_K_KERNEL)?;

    let dw: DeviceBuf<u8>  = DeviceBuf::from_slice(w_bytes)?;
    let dx: DeviceBuf<f32> = DeviceBuf::from_slice(x)?;
    let dy: DeviceBuf<f32> = DeviceBuf::new(out_dim)?;

    let block: u32 = 256;
    let grid: u32 = out_dim as u32;
    let mut w_ptr = dw.raw_ptr();
    let mut x_ptr = dx.raw_ptr();
    let mut y_ptr = dy.raw_ptr();
    let mut in_arg = in_dim as u32;
    let mut out_arg = out_dim as u32;
    let mut args: [*mut c_void; 5] = [
        &mut w_ptr   as *mut _ as *mut c_void,
        &mut x_ptr   as *mut _ as *mut c_void,
        &mut y_ptr   as *mut _ as *mut c_void,
        &mut in_arg  as *mut _ as *mut c_void,
        &mut out_arg as *mut _ as *mut c_void,
    ];
    let smem_bytes = block * std::mem::size_of::<f32>() as u32;
    unsafe { f.launch((grid, 1, 1), (block, 1, 1), smem_bytes, None, &mut args)?; }
    hip::Device(0).synchronize()?;

    let mut out = vec![0.0f32; out_dim];
    dy.copy_to_host(&mut out)?;
    Ok(out)
}

/// Fused Q4_K dequant + GEMV. `w_bytes` is the raw on-disk Q4_K stream
/// (144 bytes per super-block, 256 weights each, in_dim/256 super-blocks
/// per row, out_dim rows). Returns y of length out_dim.
pub fn matvec_q4_k_f32(cache: &KernelCache, w_bytes: &[u8], x: &[f32],
                       in_dim: usize, out_dim: usize) -> Result<Vec<f32>, String>
{
    use crate::quant::q4_k::{BLOCK_SIZE, BYTES_PER_BLOCK};
    assert_eq!(in_dim % BLOCK_SIZE, 0, "in_dim must be a multiple of {}", BLOCK_SIZE);
    let n_blocks = in_dim / BLOCK_SIZE;
    let expect_bytes = out_dim * n_blocks * BYTES_PER_BLOCK;
    assert_eq!(w_bytes.len(), expect_bytes,
        "w_bytes len {} != expected {} ({}*{}*{})",
        w_bytes.len(), expect_bytes, out_dim, n_blocks, BYTES_PER_BLOCK);
    assert_eq!(x.len(), in_dim);

    let hsaco = cache.compile("matvec_q4_k", MATVEC_Q4_K_SOURCE)?;
    let module = Module::load(&hsaco)?;
    let f = module.function(MATVEC_Q4_K_KERNEL)?;

    let dw: DeviceBuf<u8>  = DeviceBuf::from_slice(w_bytes)?;
    let dx: DeviceBuf<f32> = DeviceBuf::from_slice(x)?;
    let dy: DeviceBuf<f32> = DeviceBuf::new(out_dim)?;

    let block: u32 = 256;
    let grid: u32 = out_dim as u32;
    let mut w_ptr = dw.raw_ptr();
    let mut x_ptr = dx.raw_ptr();
    let mut y_ptr = dy.raw_ptr();
    let mut in_arg = in_dim as u32;
    let mut out_arg = out_dim as u32;
    let mut args: [*mut c_void; 5] = [
        &mut w_ptr   as *mut _ as *mut c_void,
        &mut x_ptr   as *mut _ as *mut c_void,
        &mut y_ptr   as *mut _ as *mut c_void,
        &mut in_arg  as *mut _ as *mut c_void,
        &mut out_arg as *mut _ as *mut c_void,
    ];
    let smem_bytes = block * std::mem::size_of::<f32>() as u32;
    unsafe { f.launch((grid, 1, 1), (block, 1, 1), smem_bytes, None, &mut args)?; }
    hip::Device(0).synchronize()?;

    let mut out = vec![0.0f32; out_dim];
    dy.copy_to_host(&mut out)?;
    Ok(out)
}

/// Fused Q8_0 dequant + GEMV. `w_bytes` is the raw on-disk Q8_0 byte
/// stream (34 bytes per block, in_dim/32 blocks per row, out_dim rows).
///
/// Returns y of length out_dim.
pub fn matvec_q8_0_f32(cache: &KernelCache, w_bytes: &[u8], x: &[f32],
                       in_dim: usize, out_dim: usize) -> Result<Vec<f32>, String>
{
    use crate::quant::q8_0::{BLOCK_SIZE, BYTES_PER_BLOCK};
    assert_eq!(in_dim % BLOCK_SIZE, 0, "in_dim must be a multiple of {}", BLOCK_SIZE);
    let n_blocks = in_dim / BLOCK_SIZE;
    let expect_bytes = out_dim * n_blocks * BYTES_PER_BLOCK;
    assert_eq!(w_bytes.len(), expect_bytes,
        "w_bytes len {} != expected {} ({}*{}*{})",
        w_bytes.len(), expect_bytes, out_dim, n_blocks, BYTES_PER_BLOCK);
    assert_eq!(x.len(), in_dim);

    let hsaco = cache.compile("matvec_q8_0", MATVEC_Q8_0_SOURCE)?;
    let module = Module::load(&hsaco)?;
    let f = module.function(MATVEC_Q8_0_KERNEL)?;

    let dw: DeviceBuf<u8>  = DeviceBuf::from_slice(w_bytes)?;
    let dx: DeviceBuf<f32> = DeviceBuf::from_slice(x)?;
    let dy: DeviceBuf<f32> = DeviceBuf::new(out_dim)?;

    let block: u32 = 256;
    let grid: u32 = out_dim as u32;
    let mut w_ptr = dw.raw_ptr();
    let mut x_ptr = dx.raw_ptr();
    let mut y_ptr = dy.raw_ptr();
    let mut in_arg = in_dim as u32;
    let mut out_arg = out_dim as u32;
    let mut args: [*mut c_void; 5] = [
        &mut w_ptr   as *mut _ as *mut c_void,
        &mut x_ptr   as *mut _ as *mut c_void,
        &mut y_ptr   as *mut _ as *mut c_void,
        &mut in_arg  as *mut _ as *mut c_void,
        &mut out_arg as *mut _ as *mut c_void,
    ];
    let smem_bytes = block * std::mem::size_of::<f32>() as u32;
    unsafe { f.launch((grid, 1, 1), (block, 1, 1), smem_bytes, None, &mut args)?; }
    hip::Device(0).synchronize()?;

    let mut out = vec![0.0f32; out_dim];
    dy.copy_to_host(&mut out)?;
    Ok(out)
}

/// `y = W·x` where W is row-major `[out_dim, in_dim]` fp32.
///
/// One block per output row, parallel reduction across `in_dim`. Reduction
/// order differs from CPU sequential, so output isn't bit-identical;
/// relative error stays under ~5e-4 for in_dim ≤ 8192 with reasonable
/// inputs.
pub fn matvec_f32(cache: &KernelCache, w: &[f32], x: &[f32], in_dim: usize, out_dim: usize)
    -> Result<Vec<f32>, String>
{
    assert_eq!(w.len(), in_dim * out_dim, "matvec: w must be in_dim*out_dim");
    assert_eq!(x.len(), in_dim, "matvec: x must be in_dim");

    let hsaco = cache.compile("matvec", MATVEC_SOURCE)?;
    let module = Module::load(&hsaco)?;
    let f = module.function(MATVEC_KERNEL)?;

    let dw: DeviceBuf<f32> = DeviceBuf::from_slice(w)?;
    let dx: DeviceBuf<f32> = DeviceBuf::from_slice(x)?;
    let dy: DeviceBuf<f32> = DeviceBuf::new(out_dim)?;

    let block: u32 = 256;
    let grid: u32 = out_dim as u32;
    let mut w_ptr = dw.raw_ptr();
    let mut x_ptr = dx.raw_ptr();
    let mut y_ptr = dy.raw_ptr();
    let mut in_arg = in_dim as u32;
    let mut out_arg = out_dim as u32;
    let mut args: [*mut c_void; 5] = [
        &mut w_ptr   as *mut _ as *mut c_void,
        &mut x_ptr   as *mut _ as *mut c_void,
        &mut y_ptr   as *mut _ as *mut c_void,
        &mut in_arg  as *mut _ as *mut c_void,
        &mut out_arg as *mut _ as *mut c_void,
    ];
    let smem_bytes = block * std::mem::size_of::<f32>() as u32;
    unsafe { f.launch((grid, 1, 1), (block, 1, 1), smem_bytes, None, &mut args)?; }
    hip::Device(0).synchronize()?;

    let mut out = vec![0.0f32; out_dim];
    dy.copy_to_host(&mut out)?;
    Ok(out)
}

/// Gather one row of `table` (shape [vocab, hidden]) into a fresh host
/// vector. Validates the embed-lookup kernel; the production forward will
/// keep the table resident on device.
pub fn embed_lookup_f32(cache: &KernelCache, table: &[f32], hidden: usize, token: u32) -> Result<Vec<f32>, String> {
    assert_eq!(table.len() % hidden, 0, "table length must be vocab * hidden");
    let vocab = table.len() / hidden;
    assert!((token as usize) < vocab, "token {token} out of range for vocab {vocab}");

    let hsaco = cache.compile("embed_lookup", EMBED_LOOKUP_SOURCE)?;
    let module = Module::load(&hsaco)?;
    let f = module.function(EMBED_LOOKUP_KERNEL)?;

    let dt: DeviceBuf<f32> = DeviceBuf::from_slice(table)?;
    let dy: DeviceBuf<f32> = DeviceBuf::new(hidden)?;

    let block: u32 = 256;
    let grid: u32 = (hidden as u32 + block - 1) / block;
    let mut t_ptr = dt.raw_ptr();
    let mut y_ptr = dy.raw_ptr();
    let mut row = token;
    let mut h_arg = hidden as u32;
    let mut args: [*mut c_void; 4] = [
        &mut t_ptr  as *mut _ as *mut c_void,
        &mut y_ptr  as *mut _ as *mut c_void,
        &mut row    as *mut _ as *mut c_void,
        &mut h_arg  as *mut _ as *mut c_void,
    ];
    unsafe { f.launch((grid, 1, 1), (block, 1, 1), 0, None, &mut args)?; }
    hip::Device(0).synchronize()?;

    let mut out = vec![0.0f32; hidden];
    dy.copy_to_host(&mut out)?;
    Ok(out)
}

/// Launch `rmsnorm_f32(x, w, y, n, eps)` on the GPU and return the result.
///
/// Matches `cpu::ops::rmsnorm`: `y = x * rsqrt(mean(x^2) + eps) * w`. Norm
/// weights are applied directly (no `1+w` shift); GGUF stores the shifted
/// values for Qwen3_5 already.
///
/// The reduction is a parallel tree, so floats won't be bit-identical to
/// the sequential CPU sum — relative error is well under 1e-5 for the
/// hidden sizes we care about.
pub fn rmsnorm_f32(cache: &KernelCache, x: &[f32], w: &[f32], eps: f32) -> Result<Vec<f32>, String> {
    assert_eq!(x.len(), w.len(), "rmsnorm: x and w must have the same length");
    let n = x.len();

    let hsaco = cache.compile("rmsnorm", RMSNORM_SOURCE)?;
    let module = Module::load(&hsaco)?;
    let f = module.function(RMSNORM_KERNEL)?;

    let dx: DeviceBuf<f32> = DeviceBuf::from_slice(x)?;
    let dw: DeviceBuf<f32> = DeviceBuf::from_slice(w)?;
    let dy: DeviceBuf<f32> = DeviceBuf::new(n)?;

    let block: u32 = 256;
    let mut x_ptr = dx.raw_ptr();
    let mut w_ptr = dw.raw_ptr();
    let mut y_ptr = dy.raw_ptr();
    let mut n_arg = n as u32;
    let mut eps_arg = eps;
    let mut args: [*mut c_void; 5] = [
        &mut x_ptr   as *mut _ as *mut c_void,
        &mut w_ptr   as *mut _ as *mut c_void,
        &mut y_ptr   as *mut _ as *mut c_void,
        &mut n_arg   as *mut _ as *mut c_void,
        &mut eps_arg as *mut _ as *mut c_void,
    ];
    let smem_bytes = block * std::mem::size_of::<f32>() as u32;

    // SAFETY: kernel signature matches; args live until sync below.
    unsafe { f.launch((1, 1, 1), (block, 1, 1), smem_bytes, None, &mut args)?; }
    hip::Device(0).synchronize()?;

    let mut out = vec![0.0f32; n];
    dy.copy_to_host(&mut out)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::ops::rmsnorm as cpu_rmsnorm;

    fn skip_if_no_gpu() -> Option<KernelCache> {
        if hip::device_count().ok().unwrap_or(0) < 1 {
            eprintln!("skip: no HIP device"); return None;
        }
        let _ = hip::Device::set(0).ok()?;
        match KernelCache::new() {
            Ok(c) => Some(c),
            Err(e) => { eprintln!("skip: kernel cache: {e}"); None }
        }
    }

    fn check_rmsnorm(cache: &KernelCache, n: usize, eps: f32, seed: u64, tol_rel: f32) {
        // Deterministic synthetic input + weight — values bracket realistic
        // activations (rms ~ 1) and weights (centered on 1, since GGUF
        // stores 1+w pre-shifted).
        let mut s = seed;
        let mut rng = || { s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                           ((s >> 33) as u32 as f32) / u32::MAX as f32 };
        let x: Vec<f32> = (0..n).map(|_| rng() * 2.0 - 1.0).collect();
        let w: Vec<f32> = (0..n).map(|_| 1.0 + (rng() - 0.5) * 0.2).collect();

        let mut cpu = vec![0.0f32; n];
        cpu_rmsnorm(&x, &w, eps, &mut cpu);

        let gpu = rmsnorm_f32(cache, &x, &w, eps).expect("gpu rmsnorm");

        let mut max_abs = 0.0_f32;
        let mut max_rel = 0.0_f32;
        for i in 0..n {
            let d = (gpu[i] - cpu[i]).abs();
            let r = d / cpu[i].abs().max(1e-30);
            if d > max_abs { max_abs = d; }
            if r > max_rel { max_rel = r; }
        }
        eprintln!("rmsnorm n={n} eps={eps}: max_abs={max_abs:.3e} max_rel={max_rel:.3e}");
        assert!(max_rel < tol_rel,
                "rmsnorm n={n}: max_rel {max_rel:.3e} exceeds tol {tol_rel:.3e}");
    }

    #[test]
    fn rmsnorm_matches_cpu_across_shapes() {
        let Some(cache) = skip_if_no_gpu() else { return };
        // Realistic eps + a wide range of shapes used by Qwen 3.5 / Gemma 4.
        // Tree-reduction error grows with n; widen tol for the largest case.
        check_rmsnorm(&cache, 1024, 1e-6, 0xA5A5_F00D, 1e-5);
        check_rmsnorm(&cache, 2048, 1e-6, 0xDEAD_BEEF, 1e-5);
        check_rmsnorm(&cache, 4096, 1e-6, 0x1234_5678, 1e-5);
        check_rmsnorm(&cache, 6144, 1e-6, 0xCAFEBABE, 1e-5);
    }

    #[test]
    fn swiglu_matches_cpu() {
        let Some(cache) = skip_if_no_gpu() else { return };
        let n = 4096;
        let mut s: u64 = 0xCAFE_DEAD_BEEF;
        let mut rng = || { s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                           ((s >> 33) as u32 as f32 / u32::MAX as f32) - 0.5 };
        let gate: Vec<f32> = (0..n).map(|_| rng() * 4.0).collect();   // wide range exercises silu
        let up:   Vec<f32> = (0..n).map(|_| rng() * 4.0).collect();

        let mut cpu = gate.clone();
        crate::cpu::ops::swiglu_mul(&mut cpu, &up);

        let gpu = swiglu_mul_f32(&cache, &gate, &up).expect("gpu swiglu");

        for i in 0..n {
            let d = (gpu[i] - cpu[i]).abs();
            let r = d / cpu[i].abs().max(1e-6);
            assert!(d < 1e-5 || r < 1e-5,
                "swiglu[{i}]: gpu {} cpu {} diff {d:.3e}", gpu[i], cpu[i]);
        }
    }

    #[test]
    fn matvec_iq4_xs_matches_dequant_path() {
        let Some(cache) = skip_if_no_gpu() else { return };
        use crate::quant::iq4_xs::{BLOCK_SIZE, BYTES_PER_BLOCK};
        use crate::quant::half::f32_to_f16;

        let in_dim = 2048usize;
        let out_dim = 384usize;
        let n_blocks_per_row = in_dim / BLOCK_SIZE;
        let total_blocks = out_dim * n_blocks_per_row;
        let mut w_bytes = vec![0u8; total_blocks * BYTES_PER_BLOCK];

        let mut s: u64 = 0xBEAD_F00D_C0DE;
        let mut rng_u8 = || -> u8 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 56) as u8
        };

        for blk in 0..total_blocks {
            let off = blk * BYTES_PER_BLOCK;
            let d = ((blk % 23) as f32 - 11.0) * 0.005;
            w_bytes[off..off+2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
            // scales_h: u16 — random
            w_bytes[off+2] = rng_u8();
            w_bytes[off+3] = rng_u8();
            // scales_l: 4 bytes
            for i in 0..4   { w_bytes[off + 4 + i] = rng_u8(); }
            // qs: 128 bytes of nibble pairs
            for i in 0..128 { w_bytes[off + 8 + i] = rng_u8(); }
        }

        let mut x_seed: u64 = 0x9876_FACE;
        let mut x_rng = || { x_seed = x_seed.wrapping_mul(6364136223846793005)
                                            .wrapping_add(1442695040888963407);
                             ((x_seed >> 33) as u32 as f32 / u32::MAX as f32) - 0.5 };
        let x: Vec<f32> = (0..in_dim).map(|_| x_rng()).collect();

        let mut w_fp32 = vec![0.0f32; out_dim * in_dim];
        crate::quant::iq4_xs::dequantize_to_f32(&w_bytes, &mut w_fp32);
        let mut cpu = vec![0.0f32; out_dim];
        crate::cpu::ops::matvec(&x, &w_fp32, in_dim, out_dim, &mut cpu);

        let gpu = matvec_iq4_xs_f32(&cache, &w_bytes, &x, in_dim, out_dim).expect("gpu iq4_xs matvec");

        let mut max_abs = 0.0_f32;
        let mut max_rel = 0.0_f32;
        for j in 0..out_dim {
            let d = (gpu[j] - cpu[j]).abs();
            let r = d / cpu[j].abs().max(1e-8);
            if d > max_abs { max_abs = d; }
            if r > max_rel { max_rel = r; }
        }
        eprintln!("matvec_iq4_xs {out_dim}x{in_dim}: max_abs={max_abs:.3e} max_rel={max_rel:.3e}");
        assert!(max_rel < 5e-4, "matvec_iq4_xs max_rel {max_rel:.3e} exceeds 5e-4");
    }

    #[test]
    fn matvec_q5_k_matches_dequant_path() {
        let Some(cache) = skip_if_no_gpu() else { return };
        use crate::quant::q5_k::{BLOCK_SIZE, BYTES_PER_BLOCK};
        use crate::quant::half::f32_to_f16;

        let in_dim = 2048usize;
        let out_dim = 384usize;
        let n_blocks_per_row = in_dim / BLOCK_SIZE;
        let total_blocks = out_dim * n_blocks_per_row;
        let mut w_bytes = vec![0u8; total_blocks * BYTES_PER_BLOCK];

        let mut s: u64 = 0xA5A5_5A5A;
        let mut rng_u8 = || -> u8 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 56) as u8
        };

        for blk in 0..total_blocks {
            let off = blk * BYTES_PER_BLOCK;
            let d    = ((blk %  29) as f32 -  14.0) * 0.003;
            let dmin = ((blk %  17) as f32 -   8.0) * 0.0015;
            w_bytes[off..off+2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
            w_bytes[off+2..off+4].copy_from_slice(&f32_to_f16(dmin).to_le_bytes());
            for i in 0..12  { w_bytes[off + 4  + i] = rng_u8(); }    // scales
            for i in 0..32  { w_bytes[off + 16 + i] = rng_u8(); }    // qh
            for i in 0..128 { w_bytes[off + 48 + i] = rng_u8(); }    // qs
        }

        let mut x_seed: u64 = 0x1234_BABE;
        let mut x_rng = || { x_seed = x_seed.wrapping_mul(6364136223846793005)
                                            .wrapping_add(1442695040888963407);
                             ((x_seed >> 33) as u32 as f32 / u32::MAX as f32) - 0.5 };
        let x: Vec<f32> = (0..in_dim).map(|_| x_rng()).collect();

        let mut w_fp32 = vec![0.0f32; out_dim * in_dim];
        crate::quant::q5_k::dequantize_to_f32(&w_bytes, &mut w_fp32);
        let mut cpu = vec![0.0f32; out_dim];
        crate::cpu::ops::matvec(&x, &w_fp32, in_dim, out_dim, &mut cpu);

        let gpu = matvec_q5_k_f32(&cache, &w_bytes, &x, in_dim, out_dim).expect("gpu q5_k matvec");

        let mut max_abs = 0.0_f32;
        let mut max_rel = 0.0_f32;
        for j in 0..out_dim {
            let d = (gpu[j] - cpu[j]).abs();
            let r = d / cpu[j].abs().max(1e-8);
            if d > max_abs { max_abs = d; }
            if r > max_rel { max_rel = r; }
        }
        eprintln!("matvec_q5_k {out_dim}x{in_dim}: max_abs={max_abs:.3e} max_rel={max_rel:.3e}");
        assert!(max_rel < 5e-4, "matvec_q5_k max_rel {max_rel:.3e} exceeds 5e-4");
    }

    #[test]
    fn matvec_q6_k_matches_dequant_path() {
        let Some(cache) = skip_if_no_gpu() else { return };
        use crate::quant::q6_k::{BLOCK_SIZE, BYTES_PER_BLOCK};
        use crate::quant::half::f32_to_f16;

        let in_dim = 2048usize;
        let out_dim = 384usize;
        let n_blocks_per_row = in_dim / BLOCK_SIZE;
        let total_blocks = out_dim * n_blocks_per_row;
        let mut w_bytes = vec![0u8; total_blocks * BYTES_PER_BLOCK];

        let mut s: u64 = 0xC0DECAFE;
        let mut rng_u8 = || -> u8 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 56) as u8
        };

        for blk in 0..total_blocks {
            let off = blk * BYTES_PER_BLOCK;
            for i in 0..128 { w_bytes[off + i]        = rng_u8(); }     // ql
            for i in 0..64  { w_bytes[off + 128 + i]  = rng_u8(); }     // qh
            for i in 0..16  {
                // signed int8 scales in a small range so dequant stays moderate
                let v = ((blk + i) % 11) as i8 - 5;
                w_bytes[off + 192 + i] = v as u8;
            }
            let d = ((blk % 23) as f32 - 11.0) * 0.005;                  // ~ ±0.06
            w_bytes[off + 208..off + 210].copy_from_slice(&f32_to_f16(d).to_le_bytes());
        }

        let mut x_seed: u64 = 0xFACEFEED;
        let mut x_rng = || { x_seed = x_seed.wrapping_mul(6364136223846793005)
                                            .wrapping_add(1442695040888963407);
                             ((x_seed >> 33) as u32 as f32 / u32::MAX as f32) - 0.5 };
        let x: Vec<f32> = (0..in_dim).map(|_| x_rng()).collect();

        let mut w_fp32 = vec![0.0f32; out_dim * in_dim];
        crate::quant::q6_k::dequantize_to_f32(&w_bytes, &mut w_fp32);
        let mut cpu = vec![0.0f32; out_dim];
        crate::cpu::ops::matvec(&x, &w_fp32, in_dim, out_dim, &mut cpu);

        let gpu = matvec_q6_k_f32(&cache, &w_bytes, &x, in_dim, out_dim).expect("gpu q6_k matvec");

        let mut max_abs = 0.0_f32;
        let mut max_rel = 0.0_f32;
        for j in 0..out_dim {
            let d = (gpu[j] - cpu[j]).abs();
            let r = d / cpu[j].abs().max(1e-8);
            if d > max_abs { max_abs = d; }
            if r > max_rel { max_rel = r; }
        }
        eprintln!("matvec_q6_k {out_dim}x{in_dim}: max_abs={max_abs:.3e} max_rel={max_rel:.3e}");
        assert!(max_rel < 5e-4, "matvec_q6_k max_rel {max_rel:.3e} exceeds 5e-4");
    }

    #[test]
    fn matvec_q4_k_matches_dequant_path() {
        let Some(cache) = skip_if_no_gpu() else { return };
        use crate::quant::q4_k::{BLOCK_SIZE, BYTES_PER_BLOCK};
        use crate::quant::half::f32_to_f16;

        // Synthesise a Q4_K weight buffer. Random nibbles + 6-bit packed
        // scales+mins are valid by construction; we set d/dmin to small
        // realistic values per super-block.
        let in_dim = 2048usize;     // 8 super-blocks per row
        let out_dim = 384usize;
        let n_blocks_per_row = in_dim / BLOCK_SIZE;
        let total_blocks = out_dim * n_blocks_per_row;
        let mut w_bytes = vec![0u8; total_blocks * BYTES_PER_BLOCK];

        let mut s: u64 = 0xBADC0FFEE;
        let mut rng_u8 = || -> u8 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 56) as u8
        };

        for blk in 0..total_blocks {
            let off = blk * BYTES_PER_BLOCK;
            // d ~ ±0.04, dmin ~ ±0.02 — typical magnitudes for K-quant scales.
            let d    = ((blk %  29) as f32 -  14.0) * 0.003;
            let dmin = ((blk %  17) as f32 -   8.0) * 0.0015;
            w_bytes[off..off+2].copy_from_slice(&f32_to_f16(d   ).to_le_bytes());
            w_bytes[off+2..off+4].copy_from_slice(&f32_to_f16(dmin).to_le_bytes());
            // 12-byte packed scales — any byte values are valid; the
            // dequant unpacks with mask 0x3F + high-bit shifts.
            for i in 0..12  { w_bytes[off + 4  + i] = rng_u8(); }
            // 128 bytes of nibbles.
            for i in 0..128 { w_bytes[off + 16 + i] = rng_u8(); }
        }

        let mut x_seed: u64 = 0xFEEDFACE;
        let mut x_rng = || { x_seed = x_seed.wrapping_mul(6364136223846793005)
                                            .wrapping_add(1442695040888963407);
                             ((x_seed >> 33) as u32 as f32 / u32::MAX as f32) - 0.5 };
        let x: Vec<f32> = (0..in_dim).map(|_| x_rng()).collect();

        let mut w_fp32 = vec![0.0f32; out_dim * in_dim];
        crate::quant::q4_k::dequantize_to_f32(&w_bytes, &mut w_fp32);
        let mut cpu = vec![0.0f32; out_dim];
        crate::cpu::ops::matvec(&x, &w_fp32, in_dim, out_dim, &mut cpu);

        let gpu = matvec_q4_k_f32(&cache, &w_bytes, &x, in_dim, out_dim).expect("gpu q4_k matvec");

        let mut max_abs = 0.0_f32;
        let mut max_rel = 0.0_f32;
        for j in 0..out_dim {
            let d = (gpu[j] - cpu[j]).abs();
            let r = d / cpu[j].abs().max(1e-8);
            if d > max_abs { max_abs = d; }
            if r > max_rel { max_rel = r; }
        }
        eprintln!("matvec_q4_k {out_dim}x{in_dim}: max_abs={max_abs:.3e} max_rel={max_rel:.3e}");
        assert!(max_rel < 5e-4,
            "matvec_q4_k max_rel {max_rel:.3e} exceeds 5e-4");
    }

    #[test]
    fn matvec_q8_0_matches_dequant_path() {
        let Some(cache) = skip_if_no_gpu() else { return };
        use crate::quant::q8_0::{BLOCK_SIZE, BYTES_PER_BLOCK};
        use crate::quant::half::f32_to_f16;

        // Synthesise a Q8_0 weight buffer with deterministic varying scales
        // and qs, then validate GPU matvec against (dequant + CPU matvec).
        let in_dim = 2048usize;
        let out_dim = 384usize;
        let n_blocks_per_row = in_dim / BLOCK_SIZE;
        let total_blocks = out_dim * n_blocks_per_row;
        let mut w_bytes = vec![0u8; total_blocks * BYTES_PER_BLOCK];

        let mut s: u64 = 0xC0FFEE_BEEF;
        let mut rng_u8 = || -> u8 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 56) as u8
        };

        for blk in 0..total_blocks {
            let off = blk * BYTES_PER_BLOCK;
            let scale = ((blk % 17) as f32 - 8.0) * 0.005;  // ~ ±0.04
            w_bytes[off..off+2].copy_from_slice(&f32_to_f16(scale).to_le_bytes());
            for i in 0..32 { w_bytes[off + 2 + i] = rng_u8(); }
        }

        let mut x_seed: u64 = 0xDEADBEEF;
        let mut x_rng = || { x_seed = x_seed.wrapping_mul(6364136223846793005)
                                            .wrapping_add(1442695040888963407);
                             ((x_seed >> 33) as u32 as f32 / u32::MAX as f32) - 0.5 };
        let x: Vec<f32> = (0..in_dim).map(|_| x_rng()).collect();

        // CPU oracle: dequant W, then matvec.
        let mut w_fp32 = vec![0.0f32; out_dim * in_dim];
        crate::quant::q8_0::dequantize_to_f32(&w_bytes, &mut w_fp32);
        let mut cpu = vec![0.0f32; out_dim];
        crate::cpu::ops::matvec(&x, &w_fp32, in_dim, out_dim, &mut cpu);

        let gpu = matvec_q8_0_f32(&cache, &w_bytes, &x, in_dim, out_dim).expect("gpu q8_0 matvec");

        let mut max_abs = 0.0_f32;
        let mut max_rel = 0.0_f32;
        for j in 0..out_dim {
            let d = (gpu[j] - cpu[j]).abs();
            let r = d / cpu[j].abs().max(1e-8);
            if d > max_abs { max_abs = d; }
            if r > max_rel { max_rel = r; }
        }
        eprintln!("matvec_q8_0 {out_dim}x{in_dim}: max_abs={max_abs:.3e} max_rel={max_rel:.3e}");
        assert!(max_rel < 5e-4,
            "matvec_q8_0 max_rel {max_rel:.3e} exceeds 5e-4");
    }

    #[test]
    fn matvec_matches_cpu_across_shapes() {
        let Some(cache) = skip_if_no_gpu() else { return };
        // Mix of square and rectangular shapes that cover the ranges we use:
        // attention QKV (in=2048..2560, out=384..2560), FFN gate/up (in=2048,
        // out=ffn=6144..8192), output proj (in=2048, out=vocab=250112).
        for &(in_dim, out_dim) in &[(2048usize, 2048usize), (2048, 6144), (2560, 384), (4096, 4096)] {
            let mut s: u64 = (in_dim as u64) ^ ((out_dim as u64) << 13);
            let mut rng = || { s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                               (((s >> 33) as u32 as f32) / u32::MAX as f32) - 0.5 };
            let w: Vec<f32> = (0..in_dim*out_dim).map(|_| rng() * 0.1).collect();
            let x: Vec<f32> = (0..in_dim).map(|_| rng()).collect();

            let mut cpu = vec![0.0f32; out_dim];
            crate::cpu::ops::matvec(&x, &w, in_dim, out_dim, &mut cpu);
            let gpu = matvec_f32(&cache, &w, &x, in_dim, out_dim).expect("gpu matvec");

            let mut max_abs = 0.0_f32;
            let mut max_rel = 0.0_f32;
            for j in 0..out_dim {
                let d = (gpu[j] - cpu[j]).abs();
                let r = d / cpu[j].abs().max(1e-8);
                if d > max_abs { max_abs = d; }
                if r > max_rel { max_rel = r; }
            }
            eprintln!("matvec {}x{}: max_abs={max_abs:.3e} max_rel={max_rel:.3e}",
                      out_dim, in_dim);
            assert!(max_rel < 5e-4,
                "matvec {}x{}: max_rel {max_rel:.3e} exceeds 5e-4", out_dim, in_dim);
        }
    }

    #[test]
    fn embed_lookup_returns_exact_row() {
        let Some(cache) = skip_if_no_gpu() else { return };
        // Synthetic table where row r contains [r*100 + 0, r*100 + 1, ...].
        let vocab = 1024;
        let hidden = 256;
        let mut table = vec![0.0f32; vocab * hidden];
        for r in 0..vocab {
            for i in 0..hidden {
                table[r * hidden + i] = (r as f32) * 100.0 + (i as f32);
            }
        }
        for &tok in &[0u32, 1, 100, 1023] {
            let got = embed_lookup_f32(&cache, &table, hidden, tok).expect("gpu lookup");
            for i in 0..hidden {
                let expect = (tok as f32) * 100.0 + (i as f32);
                assert_eq!(got[i].to_bits(), expect.to_bits(),
                           "row {tok} idx {i}: got {} expect {}", got[i], expect);
            }
        }
    }

    #[test]
    fn rmsnorm_handles_unit_weight() {
        let Some(cache) = skip_if_no_gpu() else { return };
        let n = 2048;
        let x: Vec<f32> = (0..n).map(|i| ((i as f32) - n as f32 / 2.0) * 0.01).collect();
        let w = vec![1.0_f32; n];
        let gpu = rmsnorm_f32(&cache, &x, &w, 1e-6).unwrap();
        let mean_sq: f32 = x.iter().map(|v| v * v).sum::<f32>() / (n as f32);
        let rrms = (mean_sq + 1e-6).sqrt().recip();
        for i in 0..n {
            let expect = x[i] * rrms;
            let d = (gpu[i] - expect).abs();
            assert!(d < 1e-5, "unit-weight rmsnorm[{i}]: gpu {} vs expect {} (d={d:.3e})",
                    gpu[i], expect);
        }
    }
}
