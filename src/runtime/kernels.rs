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

const QUANTIZE_Q8_SOURCE:   &str = include_str!("../../kernels/quantize_q8.cpp");
const ATTN_PREFILL_SRC:     &str = include_str!("../../kernels/attn_prefill.cpp");

// Test-only kernel sources for the consistency suites at the bottom of
// this file. None of these are loaded in the production forward path —
// the gemma4 / qwen35 runtimes pull their kernels directly via their
// own const SRC declarations.
#[cfg(test)] const MATVEC_Q4_K_DP4A_SRC: &str = include_str!("../../kernels/matvec_q4_k_dp4a.cpp");
#[cfg(test)] const MATVEC_Q5_K_DP4A_SRC: &str = include_str!("../../kernels/matvec_q5_k_dp4a.cpp");
#[cfg(test)] const MATVEC_Q6_K_DP4A_SRC: &str = include_str!("../../kernels/matvec_q6_k_dp4a.cpp");
#[cfg(test)] const MATVEC_Q8_0_DP4A_SRC: &str = include_str!("../../kernels/matvec_q8_0_dp4a.cpp");
#[cfg(test)] const MATVEC_Q4K_REPACKED_SRC: &str = include_str!("../../kernels/matvec_q4k_repacked.cpp");
#[cfg(test)] const MATVEC_Q5K_REPACKED_SRC: &str = include_str!("../../kernels/matvec_q5k_repacked.cpp");
#[cfg(test)] const MATVEC_Q6K_REPACKED_SRC: &str = include_str!("../../kernels/matvec_q6k_repacked.cpp");
#[cfg(test)] const MMQ_GEMM_Q4K_REPACKED_SRC: &str = include_str!("../../kernels/mmq_gemm_q4k_repacked.cpp");
#[cfg(test)] const MMQ_GEMM_Q5K_REPACKED_SRC: &str = include_str!("../../kernels/mmq_gemm_q5k_repacked.cpp");
#[cfg(test)] const MMQ_GEMM_Q6K_REPACKED_SRC: &str = include_str!("../../kernels/mmq_gemm_q6k_repacked.cpp");
const ATTN_PREFILL_FLASH_SRC: &str = include_str!("../../kernels/attn_prefill_flash.cpp");

/// Batched causal attention over `p` query tokens. Q/K/V are row-major
/// `[p, n_heads|n_kv, head_dim]`; returns `out [p, n_heads, head_dim]`.
pub fn attn_prefill_f32(cache: &KernelCache, q: &[f32], k: &[f32], v: &[f32],
                        p: usize, n_heads: usize, n_kv: usize, head_dim: usize,
                        window: u32) -> Result<Vec<f32>, String>
{
    let hsaco = cache.compile("attn_prefill", ATTN_PREFILL_SRC)?;
    let module = Module::load(&hsaco)?;
    let f = module.function("attn_prefill_f32")?;
    let dq: DeviceBuf<f32> = DeviceBuf::from_slice(q)?;
    let dk: DeviceBuf<f32> = DeviceBuf::from_slice(k)?;
    let dv: DeviceBuf<f32> = DeviceBuf::from_slice(v)?;
    let dout: DeviceBuf<f32> = DeviceBuf::new(p * n_heads * head_dim)?;
    let block: u32 = 256;
    let smem = (head_dim as u32 + p as u32 + block) * 4;
    let mut qa=dq.raw_ptr(); let mut ka=dk.raw_ptr(); let mut va=dv.raw_ptr();
    let mut oa=dout.raw_ptr();
    let mut nh=n_heads as u32; let mut nkv=n_kv as u32; let mut hd=head_dim as u32;
    let mut wn=window; let mut sc=1.0f32;
    let mut args: [*mut c_void; 9] = [
        &mut qa as *mut _ as *mut c_void, &mut ka as *mut _ as *mut c_void,
        &mut va as *mut _ as *mut c_void, &mut oa as *mut _ as *mut c_void,
        &mut nh as *mut _ as *mut c_void, &mut nkv as *mut _ as *mut c_void,
        &mut hd as *mut _ as *mut c_void, &mut wn as *mut _ as *mut c_void,
        &mut sc as *mut _ as *mut c_void];
    unsafe { f.launch((n_heads as u32, p as u32, 1), (block,1,1), smem, None, &mut args)?; }
    hip::Device(0).synchronize()?;
    let mut out = vec![0.0f32; p * n_heads * head_dim];
    dout.copy_to_host(&mut out)?;
    Ok(out)
}

/// Flash-attention prefill — same contract as `attn_prefill_f32`, run
/// through the tiled online-softmax kernel.
pub fn attn_prefill_flash_f32(cache: &KernelCache, q: &[f32], k: &[f32], v: &[f32],
                              p: usize, n_heads: usize, n_kv: usize, head_dim: usize,
                              window: u32) -> Result<Vec<f32>, String>
{
    const BQ: u32 = 8;
    const BK: u32 = 8;
    let module = Module::load(&cache.compile("attn_prefill_flash", ATTN_PREFILL_FLASH_SRC)?)?;
    let f = module.function("attn_prefill_flash_f32")?;
    let dq: DeviceBuf<f32> = DeviceBuf::from_slice(q)?;
    let dk: DeviceBuf<f32> = DeviceBuf::from_slice(k)?;
    let dv: DeviceBuf<f32> = DeviceBuf::from_slice(v)?;
    let dout: DeviceBuf<f32> = DeviceBuf::new(p * n_heads * head_dim)?;
    let block: u32 = 64 * BQ;
    let smem = 2 * BK * head_dim as u32 * 4;
    let mut qa=dq.raw_ptr(); let mut ka=dk.raw_ptr(); let mut va=dv.raw_ptr();
    let mut oa=dout.raw_ptr();
    let mut nh=n_heads as u32; let mut nkv=n_kv as u32; let mut hd=head_dim as u32;
    let mut wn=window; let mut sc=1.0f32; let mut pr=p as u32; let mut bp=0u32;
    let mut args: [*mut c_void; 11] = [
        &mut qa as *mut _ as *mut c_void, &mut ka as *mut _ as *mut c_void,
        &mut va as *mut _ as *mut c_void, &mut oa as *mut _ as *mut c_void,
        &mut nh as *mut _ as *mut c_void, &mut nkv as *mut _ as *mut c_void,
        &mut hd as *mut _ as *mut c_void, &mut wn as *mut _ as *mut c_void,
        &mut sc as *mut _ as *mut c_void, &mut pr as *mut _ as *mut c_void,
        &mut bp as *mut _ as *mut c_void];
    unsafe { f.launch((n_heads as u32, (p as u32 + BQ - 1) / BQ, 1), (block,1,1),
                      smem, None, &mut args)?; }
    hip::Device(0).synchronize()?;
    let mut out = vec![0.0f32; p * n_heads * head_dim];
    dout.copy_to_host(&mut out)?;
    Ok(out)
}

/// Quantize an f32 activation to int8 q8 blocks on the GPU, then run a
/// K-quant dp4a matvec. Used by the dp4a correctness tests. The q8
/// quantization of the activation makes this lossier than the f32
/// matvec path — callers compare with a q8-appropriate tolerance.
pub fn matvec_kquant_dp4a(cache: &KernelCache, compile_name: &str, mv_src: &str,
                          mv_kernel: &str, w_bytes: &[u8], x: &[f32],
                          in_dim: usize, out_dim: usize) -> Result<Vec<f32>, String>
{
    assert_eq!(in_dim % 32, 0, "in_dim must be a multiple of 32");
    let qmod = Module::load(&cache.compile("quantize_q8", QUANTIZE_Q8_SOURCE)?)?;
    let qf = qmod.function("quantize_q8_f32")?;
    let mvmod = Module::load(&cache.compile(compile_name, mv_src)?)?;
    let mvf = mvmod.function(mv_kernel)?;

    let dx: DeviceBuf<f32> = DeviceBuf::from_slice(x)?;
    let dxq: DeviceBuf<u8> = DeviceBuf::new((in_dim / 32) * 40)?;
    let dw: DeviceBuf<u8>  = DeviceBuf::from_slice(w_bytes)?;
    let dy: DeviceBuf<f32> = DeviceBuf::new(out_dim)?;

    let mut xp = dx.raw_ptr(); let mut qp = dxq.raw_ptr();
    let mut ind = in_dim as u32;
    let mut qargs: [*mut c_void; 3] = [
        &mut xp as *mut _ as *mut c_void, &mut qp as *mut _ as *mut c_void,
        &mut ind as *mut _ as *mut c_void];
    unsafe { qf.launch((((in_dim as u32) + 255) / 256, 1, 1), (256, 1, 1),
                       0, None, &mut qargs)?; }

    let mut wp = dw.raw_ptr(); let mut qp2 = dxq.raw_ptr(); let mut yp = dy.raw_ptr();
    let mut ia = in_dim as u32; let mut oa = out_dim as u32;
    let mut margs: [*mut c_void; 5] = [
        &mut wp as *mut _ as *mut c_void, &mut qp2 as *mut _ as *mut c_void,
        &mut yp as *mut _ as *mut c_void, &mut ia as *mut _ as *mut c_void,
        &mut oa as *mut _ as *mut c_void];
    // Rows per workgroup / workgroup size — must match the kernel's
    // launch contract. The Q4_K kernel uses 256-thread workgroups (4
    // wavefronts, 2 rows each = 8 rows); the others 64-thread, 2 rows.
    let (block, rows): (u32, u32) = if mv_kernel == "matvec_q4_k_dp4a_f32" {
        (256, 8)
    } else {
        (64, 2)
    };
    let grid = (out_dim as u32 + rows - 1) / rows;
    unsafe { mvf.launch((grid, 1, 1), (block, 1, 1), 0, None, &mut margs)?; }
    hip::Device(0).synchronize()?;

    let mut out = vec![0.0f32; out_dim];
    dy.copy_to_host(&mut out)?;
    Ok(out)
}

/// Test-only: run a repacked K-quant matvec kernel against a quantized
/// activation. `packed` is the repacked weight (the caller picks the
/// per-dtype `repack_for_matvec`).
pub fn run_repacked_matvec(cache: &KernelCache, compile_name: &str, mv_src: &str,
                           mv_kernel: &str, packed: &[u8], x: &[f32],
                           in_dim: usize, out_dim: usize) -> Result<Vec<f32>, String>
{
    let qmod = Module::load(&cache.compile("quantize_q8", QUANTIZE_Q8_SOURCE)?)?;
    let qf = qmod.function("quantize_q8_f32")?;
    let mvmod = Module::load(&cache.compile(compile_name, mv_src)?)?;
    let mvf = mvmod.function(mv_kernel)?;

    let dx: DeviceBuf<f32> = DeviceBuf::from_slice(x)?;
    let dxq: DeviceBuf<u8> = DeviceBuf::new((in_dim / 32) * 40)?;
    let dw: DeviceBuf<u8>  = DeviceBuf::from_slice(packed)?;
    let dy: DeviceBuf<f32> = DeviceBuf::new(out_dim)?;

    let mut xp = dx.raw_ptr(); let mut qp = dxq.raw_ptr();
    let mut ind = in_dim as u32;
    let mut qargs: [*mut c_void; 3] = [
        &mut xp as *mut _ as *mut c_void, &mut qp as *mut _ as *mut c_void,
        &mut ind as *mut _ as *mut c_void];
    unsafe { qf.launch((((in_dim as u32) + 255) / 256, 1, 1), (256, 1, 1),
                       0, None, &mut qargs)?; }

    let mut wp = dw.raw_ptr(); let mut qp2 = dxq.raw_ptr(); let mut yp = dy.raw_ptr();
    let mut ia = in_dim as u32; let mut oa = out_dim as u32;
    let mut margs: [*mut c_void; 5] = [
        &mut wp as *mut _ as *mut c_void, &mut qp2 as *mut _ as *mut c_void,
        &mut yp as *mut _ as *mut c_void, &mut ia as *mut _ as *mut c_void,
        &mut oa as *mut _ as *mut c_void];
    unsafe { mvf.launch(((out_dim as u32 + 7) / 8, 1, 1), (256, 1, 1),
                        0, None, &mut margs)?; }
    hip::Device(0).synchronize()?;

    let mut out = vec![0.0f32; out_dim];
    dy.copy_to_host(&mut out)?;
    Ok(out)
}

/// Run the 2D-tiled int8 MMQ GEMM over `p_rows` activation rows. `x` is
/// `[p_rows, in_dim]` f32; returns `Y` `[p_rows, out_dim]`. `compile_name`
/// / `src` / `kernel` select the Q4_K, Q5_K or Q6_K kernel.
pub fn run_mmq_gemm(cache: &KernelCache, compile_name: &str, src: &str, kernel: &str,
                    packed: &[u8], x: &[f32],
                    p_rows: usize, in_dim: usize, out_dim: usize)
    -> Result<Vec<f32>, String>
{
    let qmod = Module::load(&cache.compile("quantize_q8", QUANTIZE_Q8_SOURCE)?)?;
    let qf = qmod.function("quantize_q8_f32")?;
    let gmod = Module::load(&cache.compile(compile_name, src)?)?;
    let gf = gmod.function(kernel)?;

    let dx: DeviceBuf<f32> = DeviceBuf::from_slice(x)?;
    let dxq: DeviceBuf<u8> = DeviceBuf::new(p_rows * (in_dim / 32) * 40)?;
    let dw: DeviceBuf<u8>  = DeviceBuf::from_slice(packed)?;
    let dy: DeviceBuf<f32> = DeviceBuf::new(p_rows * out_dim)?;

    // Quantize all p_rows activations → BlockQ8 (grid.y = row).
    let mut xp = dx.raw_ptr(); let mut qp = dxq.raw_ptr();
    let mut ind = in_dim as u32;
    let mut qargs: [*mut c_void; 3] = [
        &mut xp as *mut _ as *mut c_void, &mut qp as *mut _ as *mut c_void,
        &mut ind as *mut _ as *mut c_void];
    unsafe { qf.launch((((in_dim as u32) + 255) / 256, p_rows as u32, 1), (256, 1, 1),
                       0, None, &mut qargs)?; }

    let mut wp = dw.raw_ptr(); let mut qp2 = dxq.raw_ptr(); let mut yp = dy.raw_ptr();
    let mut ia = in_dim as u32; let mut oa = out_dim as u32; let mut pa = p_rows as u32;
    let mut gargs: [*mut c_void; 6] = [
        &mut wp as *mut _ as *mut c_void, &mut qp2 as *mut _ as *mut c_void,
        &mut yp as *mut _ as *mut c_void, &mut ia as *mut _ as *mut c_void,
        &mut oa as *mut _ as *mut c_void, &mut pa as *mut _ as *mut c_void];
    // grid: BM=64 output rows × BN=64 tokens per workgroup.
    unsafe { gf.launch(((out_dim as u32 + 63) / 64, (p_rows as u32 + 63) / 64, 1),
                       (256, 1, 1), 0, None, &mut gargs)?; }
    hip::Device(0).synchronize()?;

    let mut out = vec![0.0f32; p_rows * out_dim];
    dy.copy_to_host(&mut out)?;
    Ok(out)
}

const SWIGLU_SOURCE: &str = include_str!("../../kernels/swiglu.cpp");
const SWIGLU_KERNEL: &str = "swiglu_mul_f32";

const ROPE_SOURCE: &str = include_str!("../../kernels/rope.cpp");
const ROPE_KERNEL: &str = "rope_apply_f32";

const ATTN_STEP_SOURCE: &str = include_str!("../../kernels/attn_step.cpp");
const ATTN_STEP_KERNEL: &str = "attn_step_f32";

const ROPE_BATCHED_SOURCE: &str = include_str!("../../kernels/rope_batched.cpp");
const ATTN_STEP_BATCHED_SOURCE: &str = include_str!("../../kernels/attn_step_batched.cpp");

/// Batched partial RoPE: rotate `n_rows` rows, row r at sequence
/// position `base_pos + r`. Returns the rotated buffer.
pub fn rope_apply_batched_f32(cache: &KernelCache, x: &[f32], cos: &[f32], sin: &[f32],
                              head_dim: usize, rotary_dim: usize, n_heads: usize,
                              n_rows: usize, base_pos: usize) -> Result<Vec<f32>, String>
{
    assert_eq!(x.len(), n_rows * n_heads * head_dim);
    let hsaco = cache.compile("rope_batched", ROPE_BATCHED_SOURCE)?;
    let module = Module::load(&hsaco)?;
    let f = module.function("rope_apply_batched_f32")?;

    let dx: DeviceBuf<f32> = DeviceBuf::from_slice(x)?;
    let dc: DeviceBuf<f32> = DeviceBuf::from_slice(cos)?;
    let ds: DeviceBuf<f32> = DeviceBuf::from_slice(sin)?;

    let half = (rotary_dim / 2) as u32;
    let block: u32 = 64;
    let grid_x = (half + block - 1) / block;
    let mut x_ptr = dx.raw_ptr();
    let mut c_ptr = dc.raw_ptr();
    let mut s_ptr = ds.raw_ptr();
    let mut hd = head_dim as u32;
    let mut rd = rotary_dim as u32;
    let mut nh = n_heads as u32;
    let mut bp = base_pos as u32;
    let mut args: [*mut c_void; 7] = [
        &mut x_ptr as *mut _ as *mut c_void,
        &mut c_ptr as *mut _ as *mut c_void,
        &mut s_ptr as *mut _ as *mut c_void,
        &mut hd    as *mut _ as *mut c_void,
        &mut rd    as *mut _ as *mut c_void,
        &mut nh    as *mut _ as *mut c_void,
        &mut bp    as *mut _ as *mut c_void,
    ];
    unsafe { f.launch((grid_x, n_heads as u32, n_rows as u32), (block, 1, 1),
                      0, None, &mut args)?; }
    hip::Device(0).synchronize()?;
    let mut out = vec![0.0f32; x.len()];
    dx.copy_to_host(&mut out)?;
    Ok(out)
}

/// Batched causal GQA attention. `q` is `[n_rows, n_heads, head_dim]`;
/// `k_cache`/`v_cache` hold `base_pos + n_rows` populated positions.
/// Query row r attends causally to `[0, base_pos + r]`.
pub fn attn_step_batched_f32(cache: &KernelCache, q: &[f32], k_cache: &[f32], v_cache: &[f32],
                             n_heads: usize, n_kv_heads: usize, head_dim: usize,
                             base_pos: usize, n_rows: usize, scaling: f32)
    -> Result<Vec<f32>, String>
{
    assert_eq!(q.len(), n_rows * n_heads * head_dim);
    let kv_row = n_kv_heads * head_dim;
    let max_total = base_pos + n_rows;
    assert!(k_cache.len() >= max_total * kv_row);

    let hsaco = cache.compile("attn_step_batched", ATTN_STEP_BATCHED_SOURCE)?;
    let module = Module::load(&hsaco)?;
    let f = module.function("attn_step_batched_f32")?;

    let dq: DeviceBuf<f32> = DeviceBuf::from_slice(q)?;
    let dk: DeviceBuf<f32> = DeviceBuf::from_slice(k_cache)?;
    let dv: DeviceBuf<f32> = DeviceBuf::from_slice(v_cache)?;
    let dy: DeviceBuf<f32> = DeviceBuf::new(n_rows * n_heads * head_dim)?;

    let block: u32 = 256;
    // LDS sized for the largest row: q_lds + scores(max_total) + tmp(bs).
    let smem = ((head_dim + max_total) as u32 + block) * std::mem::size_of::<f32>() as u32;
    let mut q_ptr = dq.raw_ptr();
    let mut k_ptr = dk.raw_ptr();
    let mut v_ptr = dv.raw_ptr();
    let mut o_ptr = dy.raw_ptr();
    let mut nh = n_heads as u32;
    let mut nkv = n_kv_heads as u32;
    let mut hd = head_dim as u32;
    let mut bp = base_pos as u32;
    let mut nr = n_rows as u32;
    let mut sc = scaling;
    let mut args: [*mut c_void; 10] = [
        &mut q_ptr as *mut _ as *mut c_void,
        &mut k_ptr as *mut _ as *mut c_void,
        &mut v_ptr as *mut _ as *mut c_void,
        &mut o_ptr as *mut _ as *mut c_void,
        &mut nh    as *mut _ as *mut c_void,
        &mut nkv   as *mut _ as *mut c_void,
        &mut hd    as *mut _ as *mut c_void,
        &mut bp    as *mut _ as *mut c_void,
        &mut nr    as *mut _ as *mut c_void,
        &mut sc    as *mut _ as *mut c_void,
    ];
    unsafe { f.launch((n_heads as u32, n_rows as u32, 1), (block, 1, 1),
                      smem, None, &mut args)?; }
    hip::Device(0).synchronize()?;
    let mut out = vec![0.0f32; n_rows * n_heads * head_dim];
    dy.copy_to_host(&mut out)?;
    Ok(out)
}

/// Single-step GQA attention. Computes per-head softmax(Q·Kᵀ/√d) · V.
///
/// `k_cache` and `v_cache` are full pre-allocated caches of shape
/// `[max_seq, n_kv_heads, head_dim]` flat. Only the first `total_len`
/// positions are attended over.
pub fn attn_step_f32(cache: &KernelCache,
                     q: &[f32], k_cache: &[f32], v_cache: &[f32],
                     n_heads: usize, n_kv_heads: usize, head_dim: usize,
                     total_len: usize, scaling: f32) -> Result<Vec<f32>, String>
{
    assert_eq!(q.len(), n_heads * head_dim);
    let kv_row = n_kv_heads * head_dim;
    assert!(k_cache.len() >= total_len * kv_row);
    assert!(v_cache.len() >= total_len * kv_row);
    assert_eq!(n_heads % n_kv_heads, 0);

    let hsaco = cache.compile("attn_step", ATTN_STEP_SOURCE)?;
    let module = Module::load(&hsaco)?;
    let f = module.function(ATTN_STEP_KERNEL)?;

    let dq: DeviceBuf<f32> = DeviceBuf::from_slice(q)?;
    let dk: DeviceBuf<f32> = DeviceBuf::from_slice(k_cache)?;
    let dv: DeviceBuf<f32> = DeviceBuf::from_slice(v_cache)?;
    let dy: DeviceBuf<f32> = DeviceBuf::new(n_heads * head_dim)?;

    let block: u32 = 256;
    let grid: u32 = n_heads as u32;
    let smem_bytes = ((head_dim + total_len) as u32 + block) * std::mem::size_of::<f32>() as u32;

    let mut q_ptr = dq.raw_ptr();
    let mut k_ptr = dk.raw_ptr();
    let mut v_ptr = dv.raw_ptr();
    let mut o_ptr = dy.raw_ptr();
    let mut nh = n_heads as u32;
    let mut nkv = n_kv_heads as u32;
    let mut hd = head_dim as u32;
    let mut tl = total_len as u32;
    let mut sc = scaling;
    let mut args: [*mut c_void; 9] = [
        &mut q_ptr as *mut _ as *mut c_void,
        &mut k_ptr as *mut _ as *mut c_void,
        &mut v_ptr as *mut _ as *mut c_void,
        &mut o_ptr as *mut _ as *mut c_void,
        &mut nh    as *mut _ as *mut c_void,
        &mut nkv   as *mut _ as *mut c_void,
        &mut hd    as *mut _ as *mut c_void,
        &mut tl    as *mut _ as *mut c_void,
        &mut sc    as *mut _ as *mut c_void,
    ];
    unsafe { f.launch((grid, 1, 1), (block, 1, 1), smem_bytes, None, &mut args)?; }
    hip::Device(0).synchronize()?;

    let mut out = vec![0.0f32; n_heads * head_dim];
    dy.copy_to_host(&mut out)?;
    Ok(out)
}

/// Apply partial RoPE in place to `x` (multi-head layout). Returns the
/// rotated buffer. `cos` and `sin` are full tables of shape
/// `[max_seq, rotary_dim]`.
pub fn rope_apply_f32(cache: &KernelCache, x: &[f32], cos: &[f32], sin: &[f32],
                      head_dim: usize, rotary_dim: usize, n_heads: usize, pos: usize)
    -> Result<Vec<f32>, String>
{
    assert_eq!(x.len(), n_heads * head_dim);
    assert_eq!(cos.len(), sin.len());
    assert_eq!(cos.len() % rotary_dim, 0);
    let max_seq = cos.len() / rotary_dim;
    assert!(pos < max_seq, "pos {pos} out of range max_seq {max_seq}");
    assert!(rotary_dim <= head_dim);
    assert!(rotary_dim % 2 == 0);

    let hsaco = cache.compile("rope", ROPE_SOURCE)?;
    let module = Module::load(&hsaco)?;
    let f = module.function(ROPE_KERNEL)?;

    let dx: DeviceBuf<f32>  = DeviceBuf::from_slice(x)?;
    let dc: DeviceBuf<f32>  = DeviceBuf::from_slice(cos)?;
    let ds: DeviceBuf<f32>  = DeviceBuf::from_slice(sin)?;

    let half = (rotary_dim / 2) as u32;
    let block: u32 = 64;
    let grid_x = (half + block - 1) / block;
    let grid_y = n_heads as u32;

    let mut x_ptr = dx.raw_ptr();
    let mut c_ptr = dc.raw_ptr();
    let mut s_ptr = ds.raw_ptr();
    let mut hd = head_dim as u32;
    let mut rd = rotary_dim as u32;
    let mut nh = n_heads as u32;
    let mut p  = pos as u32;
    let mut args: [*mut c_void; 7] = [
        &mut x_ptr as *mut _ as *mut c_void,
        &mut c_ptr as *mut _ as *mut c_void,
        &mut s_ptr as *mut _ as *mut c_void,
        &mut hd    as *mut _ as *mut c_void,
        &mut rd    as *mut _ as *mut c_void,
        &mut nh    as *mut _ as *mut c_void,
        &mut p     as *mut _ as *mut c_void,
    ];
    unsafe { f.launch((grid_x, grid_y, 1), (block, 1, 1), 0, None, &mut args)?; }
    hip::Device(0).synchronize()?;

    let mut out = vec![0.0f32; x.len()];
    dx.copy_to_host(&mut out)?;
    Ok(out)
}

/// `out[i] = gelu(gate[i]) * up[i]` (Gemma 4 FFN gate fusion).
pub fn geglu_mul_f32(cache: &KernelCache, gate: &[f32], up: &[f32]) -> Result<Vec<f32>, String> {
    assert_eq!(gate.len(), up.len());
    let n = gate.len();
    let hsaco = cache.compile("geglu", include_str!("../../kernels/geglu.cpp"))?;
    let module = Module::load(&hsaco)?;
    let f = module.function("geglu_mul_f32")?;
    let dg: DeviceBuf<f32> = DeviceBuf::from_slice(gate)?;
    let du: DeviceBuf<f32> = DeviceBuf::from_slice(up)?;
    let dy: DeviceBuf<f32> = DeviceBuf::new(n)?;
    let block: u32 = 256;
    let grid: u32 = (n as u32 + block - 1) / block;
    let mut g = dg.raw_ptr(); let mut u = du.raw_ptr(); let mut y = dy.raw_ptr();
    let mut na = n as u32;
    let mut args: [*mut c_void; 4] = [
        &mut g as *mut _ as *mut c_void, &mut u as *mut _ as *mut c_void,
        &mut y as *mut _ as *mut c_void, &mut na as *mut _ as *mut c_void,
    ];
    unsafe { f.launch((grid, 1, 1), (block, 1, 1), 0, None, &mut args)?; }
    hip::Device(0).synchronize()?;
    let mut out = vec![0.0f32; n];
    dy.copy_to_host(&mut out)?;
    Ok(out)
}

/// In-place final-logit soft-cap: `y[i] = cap·tanh(y[i]/cap)`.
pub fn logit_softcap_f32(cache: &KernelCache, y: &[f32], cap: f32) -> Result<Vec<f32>, String> {
    let n = y.len();
    let hsaco = cache.compile("logit_softcap", include_str!("../../kernels/logit_softcap.cpp"))?;
    let module = Module::load(&hsaco)?;
    let f = module.function("logit_softcap_f32")?;
    let dy: DeviceBuf<f32> = DeviceBuf::from_slice(y)?;
    let block: u32 = 256;
    let grid: u32 = (n as u32 + block - 1) / block;
    let mut yp = dy.raw_ptr(); let mut na = n as u32; let mut c = cap;
    let mut args: [*mut c_void; 3] = [
        &mut yp as *mut _ as *mut c_void, &mut na as *mut _ as *mut c_void,
        &mut c  as *mut _ as *mut c_void,
    ];
    unsafe { f.launch((grid, 1, 1), (block, 1, 1), 0, None, &mut args)?; }
    hip::Device(0).synchronize()?;
    let mut out = vec![0.0f32; n];
    dy.copy_to_host(&mut out)?;
    Ok(out)
}

/// Batched causal GQA attention with a sliding window (`window == 0`
/// means full causal). Same as `attn_step_f32` plus the window bound.
pub fn attn_step_window_f32(cache: &KernelCache, q: &[f32], k_cache: &[f32], v_cache: &[f32],
                            n_heads: usize, n_kv_heads: usize, head_dim: usize,
                            total_len: usize, window: usize, scaling: f32)
    -> Result<Vec<f32>, String>
{
    assert_eq!(q.len(), n_heads * head_dim);
    let hsaco = cache.compile("attn_step_window",
                              include_str!("../../kernels/attn_step_window.cpp"))?;
    let module = Module::load(&hsaco)?;
    let f = module.function("attn_step_window_f32")?;
    let dq: DeviceBuf<f32> = DeviceBuf::from_slice(q)?;
    let dk: DeviceBuf<f32> = DeviceBuf::from_slice(k_cache)?;
    let dv: DeviceBuf<f32> = DeviceBuf::from_slice(v_cache)?;
    let dy: DeviceBuf<f32> = DeviceBuf::new(n_heads * head_dim)?;
    let block: u32 = 256;
    let win_len = if window > 0 && total_len > window { window } else { total_len };
    let smem = ((head_dim + win_len) as u32 + block) * std::mem::size_of::<f32>() as u32;
    let mut q_ = dq.raw_ptr(); let mut k_ = dk.raw_ptr(); let mut v_ = dv.raw_ptr();
    let mut o_ = dy.raw_ptr();
    let mut nh = n_heads as u32; let mut nkv = n_kv_heads as u32; let mut hd = head_dim as u32;
    let mut tl = total_len as u32; let mut wn = window as u32; let mut sc = scaling;
    let mut args: [*mut c_void; 10] = [
        &mut q_ as *mut _ as *mut c_void, &mut k_ as *mut _ as *mut c_void,
        &mut v_ as *mut _ as *mut c_void, &mut o_ as *mut _ as *mut c_void,
        &mut nh as *mut _ as *mut c_void, &mut nkv as *mut _ as *mut c_void,
        &mut hd as *mut _ as *mut c_void, &mut tl as *mut _ as *mut c_void,
        &mut wn as *mut _ as *mut c_void, &mut sc as *mut _ as *mut c_void,
    ];
    unsafe { f.launch((n_heads as u32, 1, 1), (block, 1, 1), smem, None, &mut args)?; }
    hip::Device(0).synchronize()?;
    let mut out = vec![0.0f32; n_heads * head_dim];
    dy.copy_to_host(&mut out)?;
    Ok(out)
}

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
    fn attn_step_matches_cpu_oracle() {
        let Some(cache) = skip_if_no_gpu() else { return };
        // Qwen 3.5 0.8B full-attention shape: n_heads=8, n_kv_heads=2,
        // head_dim=256. We don't need a real model for the kernel test —
        // just verify per-head softmax(QKᵀ/√d)·V matches a CPU oracle.
        let n_heads = 8usize;
        let n_kv_heads = 2usize;
        let head_dim = 256usize;
        let groups = n_heads / n_kv_heads;
        let scaling = (head_dim as f32).powf(-0.5);

        for &total_len in &[1usize, 4, 17, 64] {
            let mut s: u64 = 0xA77E_C0DE ^ total_len as u64;
            let mut rng = || { s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                               ((s >> 33) as u32 as f32 / u32::MAX as f32) - 0.5 };
            let q: Vec<f32>       = (0..n_heads * head_dim).map(|_| rng()).collect();
            let k_cache: Vec<f32> = (0..total_len * n_kv_heads * head_dim).map(|_| rng()).collect();
            let v_cache: Vec<f32> = (0..total_len * n_kv_heads * head_dim).map(|_| rng()).collect();

            // CPU oracle.
            let mut cpu = vec![0.0f32; n_heads * head_dim];
            let mut scores = vec![0.0f32; total_len];
            for h in 0..n_heads {
                let kv_h = h / groups;
                let q_h = &q[h * head_dim..(h + 1) * head_dim];
                for t in 0..total_len {
                    let off = (t * n_kv_heads + kv_h) * head_dim;
                    let k_t = &k_cache[off..off + head_dim];
                    let mut acc = 0.0f32;
                    for d in 0..head_dim { acc += q_h[d] * k_t[d]; }
                    scores[t] = acc * scaling;
                }
                crate::cpu::ops::softmax(&mut scores);
                let head_out = &mut cpu[h * head_dim..(h + 1) * head_dim];
                head_out.fill(0.0);
                for t in 0..total_len {
                    let off = (t * n_kv_heads + kv_h) * head_dim;
                    let v_t = &v_cache[off..off + head_dim];
                    let s = scores[t];
                    for d in 0..head_dim { head_out[d] += s * v_t[d]; }
                }
            }

            let gpu = attn_step_f32(&cache, &q, &k_cache, &v_cache,
                n_heads, n_kv_heads, head_dim, total_len, scaling).expect("gpu attn");

            const ABS_TOL: f32 = 5.0e-5;
            const REL_TOL: f32 = 5.0e-4;
            let mut max_abs = 0.0f32;
            let mut worst_violation = 0.0f32;
            let mut worst_at = 0usize;
            for i in 0..gpu.len() {
                let d = (gpu[i] - cpu[i]).abs();
                if d > max_abs { max_abs = d; }
                let allowed = ABS_TOL.max(REL_TOL * cpu[i].abs());
                let v = d - allowed;
                if v > worst_violation { worst_violation = v; worst_at = i; }
            }
            eprintln!("attn total_len={total_len}: max_abs={max_abs:.3e}, worst_violation={:.3e} at {worst_at}",
                worst_violation);
            assert!(worst_violation <= 0.0,
                "attn total_len={total_len}: idx {worst_at} gpu={} cpu={} exceeds tol",
                gpu[worst_at], cpu[worst_at]);
        }
    }

    #[test]
    fn rope_batched_matches_per_position() {
        let Some(cache) = skip_if_no_gpu() else { return };
        use crate::cpu::rope::{RopeCache, apply_rope};
        let head_dim = 256usize;
        let rotary_dim = 64usize;
        let n_heads = 8usize;
        let n_rows = 5usize;
        let base_pos = 3usize;
        let max_seq = 32usize;
        let rope = RopeCache::new(rotary_dim, max_seq, 10000.0);
        let mut cos = vec![0.0f32; max_seq * rotary_dim];
        let mut sin = vec![0.0f32; max_seq * rotary_dim];
        for pos in 0..max_seq {
            let (c, s) = rope.get(pos);
            cos[pos*rotary_dim..(pos+1)*rotary_dim].copy_from_slice(c);
            sin[pos*rotary_dim..(pos+1)*rotary_dim].copy_from_slice(s);
        }
        let mut s: u64 = 0x5EED_C0DE;
        let mut rng = || { s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                           ((s >> 33) as u32 as f32 / u32::MAX as f32) - 0.5 };
        let x: Vec<f32> = (0..n_rows*n_heads*head_dim).map(|_| rng()).collect();

        let gpu = rope_apply_batched_f32(&cache, &x, &cos, &sin,
            head_dim, rotary_dim, n_heads, n_rows, base_pos).expect("gpu");

        let mut cpu = x.clone();
        for r in 0..n_rows {
            for h in 0..n_heads {
                let off = (r*n_heads + h) * head_dim;
                apply_rope(&mut cpu[off..off+head_dim], &rope, base_pos + r);
            }
        }
        let mut max_abs = 0.0f32;
        for i in 0..gpu.len() {
            let d = (gpu[i]-cpu[i]).abs();
            if d > max_abs { max_abs = d; }
        }
        eprintln!("rope_batched: max_abs={max_abs:.3e}");
        assert!(max_abs < 1e-6);
    }

    #[test]
    fn attn_step_batched_matches_per_row() {
        let Some(cache) = skip_if_no_gpu() else { return };
        let n_heads = 8usize;
        let n_kv_heads = 2usize;
        let head_dim = 256usize;
        let groups = n_heads / n_kv_heads;
        let scaling = (head_dim as f32).powf(-0.5);
        let base_pos = 2usize;
        let n_rows = 5usize;
        let total = base_pos + n_rows;

        let mut s: u64 = 0xA77E_BA7C;
        let mut rng = || { s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                           ((s >> 33) as u32 as f32 / u32::MAX as f32) - 0.5 };
        let q: Vec<f32> = (0..n_rows*n_heads*head_dim).map(|_| rng()).collect();
        let k_cache: Vec<f32> = (0..total*n_kv_heads*head_dim).map(|_| rng()).collect();
        let v_cache: Vec<f32> = (0..total*n_kv_heads*head_dim).map(|_| rng()).collect();

        let gpu = attn_step_batched_f32(&cache, &q, &k_cache, &v_cache,
            n_heads, n_kv_heads, head_dim, base_pos, n_rows, scaling).expect("gpu");

        // CPU oracle: per row r, attend over total_len = base_pos + r + 1.
        let mut cpu = vec![0.0f32; n_rows*n_heads*head_dim];
        let mut scores = vec![0.0f32; total];
        for r in 0..n_rows {
            let tl = base_pos + r + 1;
            for h in 0..n_heads {
                let kv_h = h / groups;
                let q_h = &q[(r*n_heads+h)*head_dim..(r*n_heads+h+1)*head_dim];
                for t in 0..tl {
                    let off = (t*n_kv_heads+kv_h)*head_dim;
                    let mut acc = 0.0f32;
                    for d in 0..head_dim { acc += q_h[d]*k_cache[off+d]; }
                    scores[t] = acc*scaling;
                }
                crate::cpu::ops::softmax(&mut scores[..tl]);
                let ho = &mut cpu[(r*n_heads+h)*head_dim..(r*n_heads+h+1)*head_dim];
                ho.fill(0.0);
                for t in 0..tl {
                    let off = (t*n_kv_heads+kv_h)*head_dim;
                    let sc = scores[t];
                    for d in 0..head_dim { ho[d] += sc*v_cache[off+d]; }
                }
            }
        }
        let mut max_abs = 0.0f32;
        for i in 0..gpu.len() {
            let d = (gpu[i]-cpu[i]).abs();
            if d > max_abs { max_abs = d; }
        }
        eprintln!("attn_step_batched: max_abs={max_abs:.3e}");
        assert!(max_abs < 5e-5, "attn_step_batched max_abs {max_abs:.3e}");
    }

    #[test]
    fn rope_matches_cpu_per_head() {
        let Some(cache) = skip_if_no_gpu() else { return };
        use crate::cpu::rope::{RopeCache, apply_rope};

        // Qwen 3.5 0.8B uses head_dim=256, rotary_dim=64, freq_base=10000.
        let head_dim = 256usize;
        let rotary_dim = 64usize;
        let n_heads = 8usize;
        let max_seq = 32usize;
        let rope = RopeCache::new(rotary_dim, max_seq, 10000.0);

        // Synthesize multi-head input.
        let mut s: u64 = 0x515E_C0DE;
        let mut rng = || { s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                           ((s >> 33) as u32 as f32 / u32::MAX as f32) - 0.5 };
        let x: Vec<f32> = (0..n_heads * head_dim).map(|_| rng()).collect();

        // Pull cos/sin tables out of the cache. Need to re-derive since
        // RopeCache keeps them private; rebuild via the public get().
        let mut cos = vec![0.0f32; max_seq * rotary_dim];
        let mut sin = vec![0.0f32; max_seq * rotary_dim];
        for pos in 0..max_seq {
            let (c, s) = rope.get(pos);
            cos[pos * rotary_dim..(pos + 1) * rotary_dim].copy_from_slice(c);
            sin[pos * rotary_dim..(pos + 1) * rotary_dim].copy_from_slice(s);
        }

        for &pos in &[0usize, 1, 5, 17, 31] {
            let mut cpu = x.clone();
            for h in 0..n_heads {
                apply_rope(&mut cpu[h * head_dim..(h + 1) * head_dim], &rope, pos);
            }
            let gpu = rope_apply_f32(&cache, &x, &cos, &sin, head_dim, rotary_dim, n_heads, pos)
                .expect("gpu rope");

            let mut max_abs = 0.0_f32;
            for i in 0..gpu.len() {
                let d = (gpu[i] - cpu[i]).abs();
                if d > max_abs { max_abs = d; }
            }
            eprintln!("rope pos={pos}: max_abs={max_abs:.3e}");
            assert!(max_abs < 1e-6, "rope pos={pos}: max_abs {max_abs:.3e}");
        }
    }

    #[test]
    fn geglu_matches_cpu() {
        let Some(cache) = skip_if_no_gpu() else { return };
        let n = 4096;
        let mut s: u64 = 0x6E61_CAFE;
        let mut rng = || { s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                           ((s >> 33) as u32 as f32 / u32::MAX as f32) - 0.5 };
        let gate: Vec<f32> = (0..n).map(|_| rng() * 4.0).collect();
        let up:   Vec<f32> = (0..n).map(|_| rng() * 4.0).collect();
        let gpu = geglu_mul_f32(&cache, &gate, &up).expect("gpu geglu");
        for i in 0..n {
            let want = crate::cpu::ops::gelu(gate[i]) * up[i];
            let d = (gpu[i] - want).abs();
            assert!(d < 1e-4, "geglu[{i}]: gpu {} cpu {} d {d:.3e}", gpu[i], want);
        }
    }

    #[test]
    fn logit_softcap_matches_cpu() {
        let Some(cache) = skip_if_no_gpu() else { return };
        let cap = 30.0f32;
        let y: Vec<f32> = (0..2048).map(|i| (i as f32 - 1024.0) * 0.13).collect();
        let gpu = logit_softcap_f32(&cache, &y, cap).expect("gpu softcap");
        for i in 0..y.len() {
            let want = cap * (y[i] / cap).tanh();
            let d = (gpu[i] - want).abs();
            assert!(d < 1e-4, "softcap[{i}]: gpu {} cpu {} d {d:.3e}", gpu[i], want);
        }
    }

    #[test]
    fn attn_step_window_matches_cpu() {
        let Some(cache) = skip_if_no_gpu() else { return };
        let n_heads = 8usize; let n_kv = 2usize; let head_dim = 128usize;
        let groups = n_heads / n_kv;
        let total_len = 50usize;
        let window = 16usize;
        let scaling = 1.0f32;
        let mut s: u64 = 0x5117_DEAD;
        let mut rng = || { s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                           ((s >> 33) as u32 as f32 / u32::MAX as f32) - 0.5 };
        let q: Vec<f32> = (0..n_heads*head_dim).map(|_| rng()).collect();
        let kc: Vec<f32> = (0..total_len*n_kv*head_dim).map(|_| rng()).collect();
        let vc: Vec<f32> = (0..total_len*n_kv*head_dim).map(|_| rng()).collect();
        let gpu = attn_step_window_f32(&cache, &q, &kc, &vc,
            n_heads, n_kv, head_dim, total_len, window, scaling).expect("gpu");
        // CPU: each head attends only to [total_len-window, total_len).
        let lo = total_len - window;
        let mut cpu = vec![0.0f32; n_heads*head_dim];
        let mut sc = vec![0.0f32; window];
        for h in 0..n_heads {
            let kvh = h / groups;
            let qh = &q[h*head_dim..(h+1)*head_dim];
            for s in 0..window {
                let t = lo + s;
                let off = (t*n_kv+kvh)*head_dim;
                let mut a = 0.0f32;
                for d in 0..head_dim { a += qh[d]*kc[off+d]; }
                sc[s] = a*scaling;
            }
            crate::cpu::ops::softmax(&mut sc);
            let ho = &mut cpu[h*head_dim..(h+1)*head_dim];
            for s in 0..window {
                let off = ((lo+s)*n_kv+kvh)*head_dim;
                for d in 0..head_dim { ho[d] += sc[s]*vc[off+d]; }
            }
        }
        let mut max_abs = 0.0f32;
        for i in 0..gpu.len() { max_abs = max_abs.max((gpu[i]-cpu[i]).abs()); }
        eprintln!("attn_step_window: max_abs={max_abs:.3e}");
        assert!(max_abs < 5e-5, "attn_step_window max_abs {max_abs:.3e}");
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

    /// Relative L2 error ||a-b|| / ||b|| — robust where individual
    /// outputs cross zero (max_rel is not).
    fn rel_l2(a: &[f32], b: &[f32]) -> f32 {
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for (x, y) in a.iter().zip(b) {
            num += ((x - y) as f64).powi(2);
            den += (*y as f64).powi(2);
        }
        (num.sqrt() / den.sqrt().max(1e-12)) as f32
    }

    // The dp4a matvecs quantize the activation to int8, so they are
    // lossier than the f32 path — the bound reflects q8 activation
    // quantization, not a kernel-correctness epsilon.
    const DP4A_REL_L2_MAX: f32 = 1.5e-2;

    #[test]
    fn matvec_q4_k_dp4a_matches_dequant_path() {
        let Some(cache) = skip_if_no_gpu() else { return };
        use crate::quant::q4_k::{BLOCK_SIZE, BYTES_PER_BLOCK};
        use crate::quant::half::f32_to_f16;

        let in_dim = 2048usize;
        let out_dim = 384usize;
        let total_blocks = out_dim * (in_dim / BLOCK_SIZE);
        let mut w_bytes = vec![0u8; total_blocks * BYTES_PER_BLOCK];
        let mut s: u64 = 0xD4A4_0001;
        let mut rng_u8 = || -> u8 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 56) as u8
        };
        for blk in 0..total_blocks {
            let off = blk * BYTES_PER_BLOCK;
            let d    = ((blk % 29) as f32 - 14.0) * 0.003;
            let dmin = ((blk % 17) as f32 -  8.0) * 0.0015;
            w_bytes[off..off+2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
            w_bytes[off+2..off+4].copy_from_slice(&f32_to_f16(dmin).to_le_bytes());
            for i in 0..12  { w_bytes[off + 4  + i] = rng_u8(); }
            for i in 0..128 { w_bytes[off + 16 + i] = rng_u8(); }
        }
        let mut xs: u64 = 0x1357_9BDF;
        let mut x_rng = || { xs = xs.wrapping_mul(6364136223846793005)
                                    .wrapping_add(1442695040888963407);
                             ((xs >> 33) as u32 as f32 / u32::MAX as f32) - 0.5 };
        let x: Vec<f32> = (0..in_dim).map(|_| x_rng()).collect();

        let mut w_fp32 = vec![0.0f32; out_dim * in_dim];
        crate::quant::q4_k::dequantize_to_f32(&w_bytes, &mut w_fp32);
        let mut cpu = vec![0.0f32; out_dim];
        crate::cpu::ops::matvec(&x, &w_fp32, in_dim, out_dim, &mut cpu);

        let gpu = matvec_kquant_dp4a(&cache, "matvec_q4_k_dp4a", MATVEC_Q4_K_DP4A_SRC,
            "matvec_q4_k_dp4a_f32", &w_bytes, &x, in_dim, out_dim).expect("q4_k dp4a");
        let e = rel_l2(&gpu, &cpu);
        eprintln!("matvec_q4_k_dp4a {out_dim}x{in_dim}: rel_l2={e:.3e}");
        assert!(e < DP4A_REL_L2_MAX, "q4_k dp4a rel_l2 {e:.3e} exceeds {DP4A_REL_L2_MAX:.1e}");
    }

    #[test]
    fn matvec_q4k_repacked_matches_dequant_path() {
        let Some(cache) = skip_if_no_gpu() else { return };
        use crate::quant::q4_k::{BLOCK_SIZE, BYTES_PER_BLOCK};
        use crate::quant::half::f32_to_f16;

        let in_dim = 2048usize;
        let out_dim = 384usize;
        let total_blocks = out_dim * (in_dim / BLOCK_SIZE);
        let mut w_bytes = vec![0u8; total_blocks * BYTES_PER_BLOCK];
        let mut s: u64 = 0xD4A4_0001;
        let mut rng_u8 = || -> u8 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 56) as u8
        };
        for blk in 0..total_blocks {
            let off = blk * BYTES_PER_BLOCK;
            let d    = ((blk % 29) as f32 - 14.0) * 0.003;
            let dmin = ((blk % 17) as f32 -  8.0) * 0.0015;
            w_bytes[off..off+2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
            w_bytes[off+2..off+4].copy_from_slice(&f32_to_f16(dmin).to_le_bytes());
            for i in 0..12  { w_bytes[off + 4  + i] = rng_u8(); }
            for i in 0..128 { w_bytes[off + 16 + i] = rng_u8(); }
        }
        let mut xs: u64 = 0x1357_9BDF;
        let mut x_rng = || { xs = xs.wrapping_mul(6364136223846793005)
                                    .wrapping_add(1442695040888963407);
                             ((xs >> 33) as u32 as f32 / u32::MAX as f32) - 0.5 };
        let x: Vec<f32> = (0..in_dim).map(|_| x_rng()).collect();

        let mut w_fp32 = vec![0.0f32; out_dim * in_dim];
        crate::quant::q4_k::dequantize_to_f32(&w_bytes, &mut w_fp32);
        let mut cpu = vec![0.0f32; out_dim];
        crate::cpu::ops::matvec(&x, &w_fp32, in_dim, out_dim, &mut cpu);

        let packed = crate::quant::q4_k::repack_for_matvec(&w_bytes, in_dim, out_dim);
        let gpu = run_repacked_matvec(&cache, "matvec_q4k_repacked", MATVEC_Q4K_REPACKED_SRC,
            "matvec_q4k_repacked_f32", &packed, &x, in_dim, out_dim).expect("q4k repacked");
        let e = rel_l2(&gpu, &cpu);
        eprintln!("matvec_q4k_repacked {out_dim}x{in_dim}: rel_l2={e:.3e}");
        assert!(e < DP4A_REL_L2_MAX,
            "q4k repacked rel_l2 {e:.3e} exceeds {DP4A_REL_L2_MAX:.1e}");

        // --- Q5_K repacked, same harness ---
        {
            use crate::quant::q5_k::{BLOCK_SIZE as Q5_BS, BYTES_PER_BLOCK as Q5_BPB};
            let tb = out_dim * (in_dim / Q5_BS);
            let mut wb = vec![0u8; tb * Q5_BPB];
            let mut s: u64 = 0xD5A5_1234;
            let mut r = || { s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                             (s >> 56) as u8 };
            for blk in 0..tb {
                let o = blk * Q5_BPB;
                let d    = ((blk % 29) as f32 - 14.0) * 0.003;
                let dmin = ((blk % 17) as f32 -  8.0) * 0.0015;
                wb[o..o+2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
                wb[o+2..o+4].copy_from_slice(&f32_to_f16(dmin).to_le_bytes());
                for i in 0..172 { wb[o+4+i] = r(); }
            }
            let mut wf = vec![0.0f32; out_dim * in_dim];
            crate::quant::q5_k::dequantize_to_f32(&wb, &mut wf);
            let mut cpu5 = vec![0.0f32; out_dim];
            crate::cpu::ops::matvec(&x, &wf, in_dim, out_dim, &mut cpu5);
            let p5 = crate::quant::q5_k::repack_for_matvec(&wb, in_dim, out_dim);
            let g5 = run_repacked_matvec(&cache, "matvec_q5k_repacked", MATVEC_Q5K_REPACKED_SRC,
                "matvec_q5k_repacked_f32", &p5, &x, in_dim, out_dim).expect("q5k repacked");
            let e5 = rel_l2(&g5, &cpu5);
            eprintln!("matvec_q5k_repacked {out_dim}x{in_dim}: rel_l2={e5:.3e}");
            assert!(e5 < DP4A_REL_L2_MAX,
                "q5k repacked rel_l2 {e5:.3e} exceeds {DP4A_REL_L2_MAX:.1e}");
        }

        // --- Q6_K repacked, same harness ---
        {
            use crate::quant::q6_k::{BLOCK_SIZE as Q6_BS, BYTES_PER_BLOCK as Q6_BPB};
            let tb = out_dim * (in_dim / Q6_BS);
            let mut wb = vec![0u8; tb * Q6_BPB];
            let mut s: u64 = 0xD6A6_5678;
            let mut r = || { s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                             (s >> 56) as u8 };
            for blk in 0..tb {
                let o = blk * Q6_BPB;
                for i in 0..208 { wb[o + i] = r(); }      // ql + qh + scales
                let d = ((blk % 23) as f32 - 11.0) * 0.004;
                wb[o+208..o+210].copy_from_slice(&f32_to_f16(d).to_le_bytes());
            }
            let mut wf = vec![0.0f32; out_dim * in_dim];
            crate::quant::q6_k::dequantize_to_f32(&wb, &mut wf);
            let mut cpu6 = vec![0.0f32; out_dim];
            crate::cpu::ops::matvec(&x, &wf, in_dim, out_dim, &mut cpu6);
            let p6 = crate::quant::q6_k::repack_for_matvec(&wb, in_dim, out_dim);
            let g6 = run_repacked_matvec(&cache, "matvec_q6k_repacked", MATVEC_Q6K_REPACKED_SRC,
                "matvec_q6k_repacked_f32", &p6, &x, in_dim, out_dim).expect("q6k repacked");
            let e6 = rel_l2(&g6, &cpu6);
            eprintln!("matvec_q6k_repacked {out_dim}x{in_dim}: rel_l2={e6:.3e}");
            assert!(e6 < DP4A_REL_L2_MAX,
                "q6k repacked rel_l2 {e6:.3e} exceeds {DP4A_REL_L2_MAX:.1e}");
        }
    }

    #[test]
    fn mmq_gemm_q4k_repacked_matches_matvec() {
        let Some(cache) = skip_if_no_gpu() else { return };
        use crate::quant::q4_k::{BLOCK_SIZE, BYTES_PER_BLOCK};
        use crate::quant::half::f32_to_f16;

        let in_dim  = 2048usize;
        let out_dim = 384usize;   // 1.5·256 — exercises the out_dim tail
        let p_rows  = 70usize;    // not a multiple of TN=32 — exercises the tail
        let total_blocks = out_dim * (in_dim / BLOCK_SIZE);
        let mut w_bytes = vec![0u8; total_blocks * BYTES_PER_BLOCK];
        let mut s: u64 = 0x9494_0001;
        let mut rng_u8 = || -> u8 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 56) as u8
        };
        for blk in 0..total_blocks {
            let off = blk * BYTES_PER_BLOCK;
            let d    = ((blk % 29) as f32 - 14.0) * 0.003;
            let dmin = ((blk % 17) as f32 -  8.0) * 0.0015;
            w_bytes[off..off+2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
            w_bytes[off+2..off+4].copy_from_slice(&f32_to_f16(dmin).to_le_bytes());
            for i in 0..12  { w_bytes[off + 4  + i] = rng_u8(); }
            for i in 0..128 { w_bytes[off + 16 + i] = rng_u8(); }
        }
        let mut xs: u64 = 0x2468_ACE0;
        let mut x_rng = || { xs = xs.wrapping_mul(6364136223846793005)
                                    .wrapping_add(1442695040888963407);
                             ((xs >> 33) as u32 as f32 / u32::MAX as f32) - 0.5 };
        let x: Vec<f32> = (0..p_rows * in_dim).map(|_| x_rng()).collect();

        // fp32 reference — per-row matvec on the dequantised weight.
        let mut w_fp32 = vec![0.0f32; out_dim * in_dim];
        crate::quant::q4_k::dequantize_to_f32(&w_bytes, &mut w_fp32);
        let mut cpu = vec![0.0f32; p_rows * out_dim];
        for p in 0..p_rows {
            let mut row = vec![0.0f32; out_dim];
            crate::cpu::ops::matvec(&x[p*in_dim..(p+1)*in_dim], &w_fp32,
                                    in_dim, out_dim, &mut row);
            cpu[p*out_dim..(p+1)*out_dim].copy_from_slice(&row);
        }

        let packed = crate::quant::q4_k::repack_for_matvec(&w_bytes, in_dim, out_dim);
        let gpu = run_mmq_gemm(&cache, "mmq_gemm_q4k_repacked", MMQ_GEMM_Q4K_REPACKED_SRC,
            "mmq_gemm_q4k_repacked_f32", &packed, &x, p_rows, in_dim, out_dim)
            .expect("mmq gemm q4k");
        let e = rel_l2(&gpu, &cpu);
        eprintln!("mmq_gemm_q4k_repacked {p_rows}x{out_dim}x{in_dim}: rel_l2={e:.3e}");
        assert!(e < DP4A_REL_L2_MAX,
            "mmq gemm q4k rel_l2 {e:.3e} exceeds {DP4A_REL_L2_MAX:.1e}");
    }

    /// Random `[p_rows, in_dim]` activations for the MMQ tests.
    fn mmq_test_x(p_rows: usize, in_dim: usize) -> Vec<f32> {
        let mut xs: u64 = 0x2468_ACE0;
        let mut x_rng = || { xs = xs.wrapping_mul(6364136223846793005)
                                    .wrapping_add(1442695040888963407);
                             ((xs >> 33) as u32 as f32 / u32::MAX as f32) - 0.5 };
        (0..p_rows * in_dim).map(|_| x_rng()).collect()
    }

    #[test]
    fn attn_prefill_flash_matches_plain() {
        let Some(cache) = skip_if_no_gpu() else { return };
        let (p, n_heads, n_kv, head_dim) = (70usize, 8usize, 2usize, 128usize);
        let mut s: u64 = 0xA11E_0F1A;
        let mut rng = || { s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                           ((s >> 40) as u32 as f32 / (1u32 << 24) as f32) - 0.5 };
        let q: Vec<f32> = (0..p*n_heads*head_dim).map(|_| rng()).collect();
        let k: Vec<f32> = (0..p*n_kv*head_dim).map(|_| rng()).collect();
        let v: Vec<f32> = (0..p*n_kv*head_dim).map(|_| rng()).collect();

        // window 0 (full causal) and a sliding window narrower than p.
        for window in [0u32, 24] {
            let plain = attn_prefill_f32(&cache, &q, &k, &v, p, n_heads, n_kv, head_dim, window)
                .expect("plain attn_prefill");
            let flash = attn_prefill_flash_f32(&cache, &q, &k, &v, p, n_heads, n_kv, head_dim, window)
                .expect("flash attn_prefill");
            let e = rel_l2(&flash, &plain);
            eprintln!("attn_prefill_flash window={window}: rel_l2={e:.3e}");
            assert!(e < 1.0e-3,
                "flash vs plain attn_prefill window={window}: rel_l2 {e:.3e} too large");
        }
    }

    #[test]
    fn mmq_gemm_q5k_repacked_matches_matvec() {
        let Some(cache) = skip_if_no_gpu() else { return };
        use crate::quant::q5_k::{BLOCK_SIZE, BYTES_PER_BLOCK};
        use crate::quant::half::f32_to_f16;

        let in_dim = 2048usize;
        let out_dim = 384usize;
        let p_rows = 70usize;
        let total_blocks = out_dim * (in_dim / BLOCK_SIZE);
        let mut w = vec![0u8; total_blocks * BYTES_PER_BLOCK];
        let mut s: u64 = 0xD5A5_2222;
        let mut r = || { s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                         (s >> 56) as u8 };
        for blk in 0..total_blocks {
            let o = blk * BYTES_PER_BLOCK;
            let d    = ((blk % 29) as f32 - 14.0) * 0.003;
            let dmin = ((blk % 17) as f32 -  8.0) * 0.0015;
            w[o..o+2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
            w[o+2..o+4].copy_from_slice(&f32_to_f16(dmin).to_le_bytes());
            for i in 0..172 { w[o+4+i] = r(); }
        }
        let x = mmq_test_x(p_rows, in_dim);
        let mut w_fp32 = vec![0.0f32; out_dim * in_dim];
        crate::quant::q5_k::dequantize_to_f32(&w, &mut w_fp32);
        let mut cpu = vec![0.0f32; p_rows * out_dim];
        for p in 0..p_rows {
            let mut row = vec![0.0f32; out_dim];
            crate::cpu::ops::matvec(&x[p*in_dim..(p+1)*in_dim], &w_fp32,
                                    in_dim, out_dim, &mut row);
            cpu[p*out_dim..(p+1)*out_dim].copy_from_slice(&row);
        }
        let packed = crate::quant::q5_k::repack_for_matvec(&w, in_dim, out_dim);
        let gpu = run_mmq_gemm(&cache, "mmq_gemm_q5k_repacked", MMQ_GEMM_Q5K_REPACKED_SRC,
            "mmq_gemm_q5k_repacked_f32", &packed, &x, p_rows, in_dim, out_dim)
            .expect("mmq gemm q5k");
        let e = rel_l2(&gpu, &cpu);
        eprintln!("mmq_gemm_q5k_repacked {p_rows}x{out_dim}x{in_dim}: rel_l2={e:.3e}");
        assert!(e < DP4A_REL_L2_MAX,
            "mmq gemm q5k rel_l2 {e:.3e} exceeds {DP4A_REL_L2_MAX:.1e}");
    }

    #[test]
    fn mmq_gemm_q6k_repacked_matches_matvec() {
        let Some(cache) = skip_if_no_gpu() else { return };
        use crate::quant::q6_k::{BLOCK_SIZE, BYTES_PER_BLOCK};
        use crate::quant::half::f32_to_f16;

        let in_dim = 2048usize;
        let out_dim = 384usize;
        let p_rows = 70usize;
        let total_blocks = out_dim * (in_dim / BLOCK_SIZE);
        let mut w = vec![0u8; total_blocks * BYTES_PER_BLOCK];
        let mut s: u64 = 0xD6A6_3333;
        let mut r = || { s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                         (s >> 56) as u8 };
        for blk in 0..total_blocks {
            let o = blk * BYTES_PER_BLOCK;
            for i in 0..208 { w[o+i] = r(); }          // ql + qh + scales
            let d = ((blk % 23) as f32 - 11.0) * 0.004;
            w[o+208..o+210].copy_from_slice(&f32_to_f16(d).to_le_bytes());
        }
        let x = mmq_test_x(p_rows, in_dim);
        let mut w_fp32 = vec![0.0f32; out_dim * in_dim];
        crate::quant::q6_k::dequantize_to_f32(&w, &mut w_fp32);
        let mut cpu = vec![0.0f32; p_rows * out_dim];
        for p in 0..p_rows {
            let mut row = vec![0.0f32; out_dim];
            crate::cpu::ops::matvec(&x[p*in_dim..(p+1)*in_dim], &w_fp32,
                                    in_dim, out_dim, &mut row);
            cpu[p*out_dim..(p+1)*out_dim].copy_from_slice(&row);
        }
        let packed = crate::quant::q6_k::repack_for_matvec(&w, in_dim, out_dim);
        let gpu = run_mmq_gemm(&cache, "mmq_gemm_q6k_repacked", MMQ_GEMM_Q6K_REPACKED_SRC,
            "mmq_gemm_q6k_repacked_f32", &packed, &x, p_rows, in_dim, out_dim)
            .expect("mmq gemm q6k");
        let e = rel_l2(&gpu, &cpu);
        eprintln!("mmq_gemm_q6k_repacked {p_rows}x{out_dim}x{in_dim}: rel_l2={e:.3e}");
        assert!(e < DP4A_REL_L2_MAX,
            "mmq gemm q6k rel_l2 {e:.3e} exceeds {DP4A_REL_L2_MAX:.1e}");
    }

    #[test]
    #[ignore = "benchmark — run explicitly with --ignored"]
    fn bench_q4k_repacked_vs_dp4a() {
        let Some(cache) = skip_if_no_gpu() else { return };
        use crate::quant::q4_k::{BLOCK_SIZE, BYTES_PER_BLOCK};
        use crate::quant::half::f32_to_f16;

      for &(in_dim, out_dim) in &[(5376usize, 21504usize), (5376, 8192),
                                  (5376, 4096), (8192, 5376), (4096, 5376)] {
        let total_blocks = out_dim * (in_dim / BLOCK_SIZE);
        let mut w = vec![0u8; total_blocks * BYTES_PER_BLOCK];
        let mut s: u64 = 0xBEEF_0001;
        let mut r = || { s = s.wrapping_mul(6364136223846793005).wrapping_add(1); (s >> 56) as u8 };
        for blk in 0..total_blocks {
            let o = blk * BYTES_PER_BLOCK;
            w[o..o+2].copy_from_slice(&f32_to_f16(0.01).to_le_bytes());
            w[o+2..o+4].copy_from_slice(&f32_to_f16(0.005).to_le_bytes());
            for i in 0..140 { w[o+4+i] = r(); }
        }
        let x: Vec<f32> = (0..in_dim).map(|i| ((i * 7 % 13) as f32 - 6.0) * 0.1).collect();
        let packed = crate::quant::q4_k::repack_for_matvec(&w, in_dim, out_dim);

        let qmod = Module::load(&cache.compile("quantize_q8", QUANTIZE_Q8_SOURCE).unwrap()).unwrap();
        let dp4a = Module::load(&cache.compile("matvec_q4_k_dp4a", MATVEC_Q4_K_DP4A_SRC).unwrap()).unwrap();
        let repk = Module::load(&cache.compile("matvec_q4k_repacked", MATVEC_Q4K_REPACKED_SRC).unwrap()).unwrap();

        let dx: DeviceBuf<f32> = DeviceBuf::from_slice(&x).unwrap();
        let dxq: DeviceBuf<u8> = DeviceBuf::new((in_dim / 32) * 40).unwrap();
        let dw_q: DeviceBuf<u8> = DeviceBuf::from_slice(&w).unwrap();
        let dw_r: DeviceBuf<u8> = DeviceBuf::from_slice(&packed).unwrap();
        let dy: DeviceBuf<f32> = DeviceBuf::new(out_dim).unwrap();
        let stream = hip::Stream::new().unwrap();

        // quantize the activation once
        {
            let qf = qmod.function("quantize_q8_f32").unwrap();
            let mut xp = dx.raw_ptr(); let mut qp = dxq.raw_ptr(); let mut id = in_dim as u32;
            let mut a: [*mut c_void; 3] = [&mut xp as *mut _ as *mut c_void,
                &mut qp as *mut _ as *mut c_void, &mut id as *mut _ as *mut c_void];
            unsafe { qf.launch(((in_dim as u32 + 255)/256,1,1),(256,1,1),0,Some(&stream),&mut a).unwrap(); }
        }

        let bench = |module: &Module, kname: &str, wptr: *mut c_void, bytes: f64| -> (f64, f64) {
            let f = module.function(kname).unwrap();
            let grid = (out_dim as u32 + 7) / 8;
            let launch = || {
                let mut wp = wptr; let mut qp = dxq.raw_ptr(); let mut yp = dy.raw_ptr();
                let mut id = in_dim as u32; let mut od = out_dim as u32;
                let mut a: [*mut c_void; 5] = [&mut wp as *mut _ as *mut c_void,
                    &mut qp as *mut _ as *mut c_void, &mut yp as *mut _ as *mut c_void,
                    &mut id as *mut _ as *mut c_void, &mut od as *mut _ as *mut c_void];
                unsafe { f.launch((grid,1,1),(256,1,1),0,Some(&stream),&mut a).unwrap(); }
            };
            for _ in 0..20 { launch(); }
            stream.synchronize().unwrap();
            let (a, b) = (hip::Event::new().unwrap(), hip::Event::new().unwrap());
            a.record(&stream).unwrap();
            const N: usize = 300;
            for _ in 0..N { launch(); }
            b.record(&stream).unwrap();
            stream.synchronize().unwrap();
            let ms = hip::Event::elapsed_time(&a, &b).unwrap() as f64 / N as f64;
            (ms, bytes / (ms / 1000.0) / 1e9)
        };

        let q_bytes = (out_dim * (in_dim / BLOCK_SIZE) * BYTES_PER_BLOCK) as f64;
        let r_bytes = (out_dim * (in_dim / 32) * 20) as f64;
        let (dm, dgb) = bench(&dp4a, "matvec_q4_k_dp4a_f32", dw_q.raw_ptr(), q_bytes);
        let (rm, rgb) = bench(&repk, "matvec_q4k_repacked_f32", dw_r.raw_ptr(), r_bytes);
        eprintln!("q4_k matvec {out_dim}x{in_dim}:  dp4a {dm:.4}ms {dgb:.0}GB/s  \
                   repacked {rm:.4}ms {rgb:.0}GB/s  ({:.2}x)", dm / rm);
      }
    }

    #[test]
    fn matvec_q5_k_dp4a_matches_dequant_path() {
        let Some(cache) = skip_if_no_gpu() else { return };
        use crate::quant::q5_k::{BLOCK_SIZE, BYTES_PER_BLOCK};
        use crate::quant::half::f32_to_f16;

        let in_dim = 2048usize;
        let out_dim = 384usize;
        let total_blocks = out_dim * (in_dim / BLOCK_SIZE);
        let mut w_bytes = vec![0u8; total_blocks * BYTES_PER_BLOCK];
        let mut s: u64 = 0xD5A5_0002;
        let mut rng_u8 = || -> u8 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 56) as u8
        };
        for blk in 0..total_blocks {
            let off = blk * BYTES_PER_BLOCK;
            let d    = ((blk % 29) as f32 - 14.0) * 0.003;
            let dmin = ((blk % 17) as f32 -  8.0) * 0.0015;
            w_bytes[off..off+2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
            w_bytes[off+2..off+4].copy_from_slice(&f32_to_f16(dmin).to_le_bytes());
            for i in 0..12  { w_bytes[off + 4  + i] = rng_u8(); }
            for i in 0..32  { w_bytes[off + 16 + i] = rng_u8(); }
            for i in 0..128 { w_bytes[off + 48 + i] = rng_u8(); }
        }
        let mut xs: u64 = 0x2468_ACE0;
        let mut x_rng = || { xs = xs.wrapping_mul(6364136223846793005)
                                    .wrapping_add(1442695040888963407);
                             ((xs >> 33) as u32 as f32 / u32::MAX as f32) - 0.5 };
        let x: Vec<f32> = (0..in_dim).map(|_| x_rng()).collect();

        let mut w_fp32 = vec![0.0f32; out_dim * in_dim];
        crate::quant::q5_k::dequantize_to_f32(&w_bytes, &mut w_fp32);
        let mut cpu = vec![0.0f32; out_dim];
        crate::cpu::ops::matvec(&x, &w_fp32, in_dim, out_dim, &mut cpu);

        let gpu = matvec_kquant_dp4a(&cache, "matvec_q5_k_dp4a", MATVEC_Q5_K_DP4A_SRC,
            "matvec_q5_k_dp4a_f32", &w_bytes, &x, in_dim, out_dim).expect("q5_k dp4a");
        let e = rel_l2(&gpu, &cpu);
        eprintln!("matvec_q5_k_dp4a {out_dim}x{in_dim}: rel_l2={e:.3e}");
        assert!(e < DP4A_REL_L2_MAX, "q5_k dp4a rel_l2 {e:.3e} exceeds {DP4A_REL_L2_MAX:.1e}");
    }

    #[test]
    fn matvec_q6_k_dp4a_matches_dequant_path() {
        let Some(cache) = skip_if_no_gpu() else { return };
        use crate::quant::q6_k::{BLOCK_SIZE, BYTES_PER_BLOCK};
        use crate::quant::half::f32_to_f16;

        let in_dim = 2048usize;
        let out_dim = 384usize;
        let total_blocks = out_dim * (in_dim / BLOCK_SIZE);
        let mut w_bytes = vec![0u8; total_blocks * BYTES_PER_BLOCK];
        let mut s: u64 = 0xD6A6_0003;
        let mut rng_u8 = || -> u8 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 56) as u8
        };
        for blk in 0..total_blocks {
            let off = blk * BYTES_PER_BLOCK;
            for i in 0..128 { w_bytes[off + i]       = rng_u8(); }
            for i in 0..64  { w_bytes[off + 128 + i] = rng_u8(); }
            for i in 0..16  {
                let v = ((blk + i) % 11) as i8 - 5;
                w_bytes[off + 192 + i] = v as u8;
            }
            let d = ((blk % 23) as f32 - 11.0) * 0.005;
            w_bytes[off + 208..off + 210].copy_from_slice(&f32_to_f16(d).to_le_bytes());
        }
        let mut xs: u64 = 0x3690_CF12;
        let mut x_rng = || { xs = xs.wrapping_mul(6364136223846793005)
                                    .wrapping_add(1442695040888963407);
                             ((xs >> 33) as u32 as f32 / u32::MAX as f32) - 0.5 };
        let x: Vec<f32> = (0..in_dim).map(|_| x_rng()).collect();

        let mut w_fp32 = vec![0.0f32; out_dim * in_dim];
        crate::quant::q6_k::dequantize_to_f32(&w_bytes, &mut w_fp32);
        let mut cpu = vec![0.0f32; out_dim];
        crate::cpu::ops::matvec(&x, &w_fp32, in_dim, out_dim, &mut cpu);

        let gpu = matvec_kquant_dp4a(&cache, "matvec_q6_k_dp4a", MATVEC_Q6_K_DP4A_SRC,
            "matvec_q6_k_dp4a_f32", &w_bytes, &x, in_dim, out_dim).expect("q6_k dp4a");
        let e = rel_l2(&gpu, &cpu);
        eprintln!("matvec_q6_k_dp4a {out_dim}x{in_dim}: rel_l2={e:.3e}");
        assert!(e < DP4A_REL_L2_MAX, "q6_k dp4a rel_l2 {e:.3e} exceeds {DP4A_REL_L2_MAX:.1e}");
    }

    #[test]
    fn attn_prefill_matches_reference() {
        let Some(cache) = skip_if_no_gpu() else { return };
        let (p, n_heads, n_kv, hd) = (12usize, 4usize, 2usize, 64usize);
        let window = 0u32;          // full causal
        let mut s: u64 = 0x5EED_A77;
        let mut rng = || { s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                           ((s >> 40) as u32 as f32 / (1u32<<24) as f32) - 0.5 };
        let q: Vec<f32> = (0..p*n_heads*hd).map(|_| rng()).collect();
        let k: Vec<f32> = (0..p*n_kv*hd).map(|_| rng()).collect();
        let v: Vec<f32> = (0..p*n_kv*hd).map(|_| rng()).collect();

        let gpu = attn_prefill_f32(&cache, &q, &k, &v, p, n_heads, n_kv, hd, window)
            .expect("attn_prefill");

        // CPU reference: per (query, head) causal attention.
        let groups = n_heads / n_kv;
        let mut cpu = vec![0.0f32; p * n_heads * hd];
        for qp in 0..p {
            for h in 0..n_heads {
                let kv_h = h / groups;
                let qh = &q[(qp*n_heads + h)*hd..][..hd];
                let mut sc = vec![0.0f32; qp + 1];
                for t in 0..=qp {
                    let kt = &k[(t*n_kv + kv_h)*hd..][..hd];
                    sc[t] = (0..hd).map(|d| qh[d]*kt[d]).sum();
                }
                let m = sc.iter().cloned().fold(f32::MIN, f32::max);
                let exp: Vec<f32> = sc.iter().map(|&x| (x-m).exp()).collect();
                let sum: f32 = exp.iter().sum();
                let o = &mut cpu[(qp*n_heads + h)*hd..][..hd];
                for t in 0..=qp {
                    let vt = &v[(t*n_kv + kv_h)*hd..][..hd];
                    let w = exp[t] / sum;
                    for d in 0..hd { o[d] += w * vt[d]; }
                }
            }
        }
        let e = rel_l2(&gpu, &cpu);
        eprintln!("attn_prefill {p}q {n_heads}h: rel_l2={e:.3e}");
        assert!(e < 1e-4, "attn_prefill rel_l2 {e:.3e}");
    }

    #[test]
    fn matvec_q8_0_dp4a_matches_dequant_path() {
        let Some(cache) = skip_if_no_gpu() else { return };
        use crate::quant::q8_0::{BLOCK_SIZE, BYTES_PER_BLOCK};
        use crate::quant::half::f32_to_f16;

        let in_dim = 2048usize;
        let out_dim = 384usize;
        let total_blocks = out_dim * (in_dim / BLOCK_SIZE);
        let mut w_bytes = vec![0u8; total_blocks * BYTES_PER_BLOCK];
        let mut s: u64 = 0xD8A8_0004;
        let mut rng_u8 = || -> u8 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 56) as u8
        };
        for blk in 0..total_blocks {
            let off = blk * BYTES_PER_BLOCK;
            let d = ((blk % 19) as f32 - 9.0) * 0.004;
            w_bytes[off..off+2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
            for i in 0..32 { w_bytes[off + 2 + i] = rng_u8(); }
        }
        let mut xs: u64 = 0x4812_C0DE;
        let mut x_rng = || { xs = xs.wrapping_mul(6364136223846793005)
                                    .wrapping_add(1442695040888963407);
                             ((xs >> 33) as u32 as f32 / u32::MAX as f32) - 0.5 };
        let x: Vec<f32> = (0..in_dim).map(|_| x_rng()).collect();

        let mut w_fp32 = vec![0.0f32; out_dim * in_dim];
        crate::quant::q8_0::dequantize_to_f32(&w_bytes, &mut w_fp32);
        let mut cpu = vec![0.0f32; out_dim];
        crate::cpu::ops::matvec(&x, &w_fp32, in_dim, out_dim, &mut cpu);

        let gpu = matvec_kquant_dp4a(&cache, "matvec_q8_0_dp4a", MATVEC_Q8_0_DP4A_SRC,
            "matvec_q8_0_dp4a_f32", &w_bytes, &x, in_dim, out_dim).expect("q8_0 dp4a");
        let e = rel_l2(&gpu, &cpu);
        eprintln!("matvec_q8_0_dp4a {out_dim}x{in_dim}: rel_l2={e:.3e}");
        assert!(e < DP4A_REL_L2_MAX, "q8_0 dp4a rel_l2 {e:.3e} exceeds {DP4A_REL_L2_MAX:.1e}");
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

    fn check_bulk_dequant(cache: &KernelCache, name: &str, source: &str, kernel: &str,
                          w_bytes: &[u8], weights_per_block: usize, block_threads: u32,
                          cpu_f32: &[f32]) {
        use crate::quant::half::f32_to_f16;
        let n_blocks = cpu_f32.len() / weights_per_block;
        let hsaco = cache.compile(name, source).expect("compile");
        let module = Module::load(&hsaco).expect("load");
        let f = module.function(kernel).expect("function");

        let dw: DeviceBuf<u8>  = DeviceBuf::from_slice(w_bytes).unwrap();
        let dout: DeviceBuf<u16> = DeviceBuf::new(cpu_f32.len()).unwrap();
        let mut w_ptr = dw.raw_ptr();
        let mut o_ptr = dout.raw_ptr();
        let mut nb = n_blocks as u32;
        let mut args: [*mut c_void; 3] = [
            &mut w_ptr as *mut _ as *mut c_void,
            &mut o_ptr as *mut _ as *mut c_void,
            &mut nb    as *mut _ as *mut c_void,
        ];
        unsafe { f.launch((n_blocks as u32, 1, 1), (block_threads, 1, 1), 0, None, &mut args)
            .expect("launch"); }
        hip::Device(0).synchronize().expect("sync");

        let mut got = vec![0u16; cpu_f32.len()];
        dout.copy_to_host(&mut got).unwrap();
        // GPU emits fp16; reference is f32→f16 of the CPU dequant.
        // Compare numerically (decoding both back to f32) rather than by
        // raw bits — signed zero (±0.0) is numerically equal but differs
        // in the sign bit, and fp16 rounding can land 1 ulp apart.
        use crate::quant::half::f16_to_f32;
        let mut max_abs = 0.0f32;
        for i in 0..cpu_f32.len() {
            let want = f16_to_f32(f32_to_f16(cpu_f32[i]));
            let got_f = f16_to_f32(got[i]);
            let d = (got_f - want).abs();
            if d > max_abs { max_abs = d; }
        }
        eprintln!("{name}: max abs diff vs f32→f16(cpu) = {max_abs:.3e}");
        // fp16 has ~3 decimal digits; allow a couple of ulp at the
        // largest magnitudes seen here (≤ ~40).
        assert!(max_abs < 5e-2, "{name}: abs diff {max_abs:.3e} too large");
    }

    #[test]
    fn bulk_dequant_matches_cpu() {
        let Some(cache) = skip_if_no_gpu() else { return };
        // Synthesise random bytes for each quant type, dequant on CPU to
        // f32, dequant on GPU to f16, and check the f16 results agree
        // with f32→f16 of the CPU output.
        let mut s: u64 = 0xDE2AA47;
        let mut rng_u8 = || -> u8 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 56) as u8
        };

        // Q4_K — 256 w / 144 B.
        {
            let n_blocks = 64;
            let mut w = vec![0u8; n_blocks * crate::quant::q4_k::BYTES_PER_BLOCK];
            for b in w.iter_mut() { *b = rng_u8(); }
            // Tame the fp16 scales so dequant values stay in fp16 range.
            for blk in 0..n_blocks {
                let off = blk * crate::quant::q4_k::BYTES_PER_BLOCK;
                w[off..off+2].copy_from_slice(
                    &crate::quant::half::f32_to_f16(0.01).to_le_bytes());
                w[off+2..off+4].copy_from_slice(
                    &crate::quant::half::f32_to_f16(0.005).to_le_bytes());
            }
            let mut cpu = vec![0.0f32; n_blocks * 256];
            crate::quant::q4_k::dequantize_to_f32(&w, &mut cpu);
            check_bulk_dequant(&cache, "dequant_q4_k_f16",
                include_str!("../../kernels/dequant_q4_k_f16.cpp"), "dequant_q4_k_f16",
                &w, 256, 256, &cpu);
        }
        // Q6_K — 256 w / 210 B.
        {
            let n_blocks = 64;
            let bpb = crate::quant::q6_k::BYTES_PER_BLOCK;
            let mut w = vec![0u8; n_blocks * bpb];
            for b in w.iter_mut() { *b = rng_u8(); }
            for blk in 0..n_blocks {
                let off = blk * bpb;
                w[off+208..off+210].copy_from_slice(
                    &crate::quant::half::f32_to_f16(0.01).to_le_bytes());
            }
            let mut cpu = vec![0.0f32; n_blocks * 256];
            crate::quant::q6_k::dequantize_to_f32(&w, &mut cpu);
            check_bulk_dequant(&cache, "dequant_q6_k_f16",
                include_str!("../../kernels/dequant_q6_k_f16.cpp"), "dequant_q6_k_f16",
                &w, 256, 256, &cpu);
        }
        // Q5_K — 256 w / 176 B.
        {
            let n_blocks = 64;
            let bpb = crate::quant::q5_k::BYTES_PER_BLOCK;
            let mut w = vec![0u8; n_blocks * bpb];
            for b in w.iter_mut() { *b = rng_u8(); }
            for blk in 0..n_blocks {
                let off = blk * bpb;
                w[off..off+2].copy_from_slice(
                    &crate::quant::half::f32_to_f16(0.01).to_le_bytes());
                w[off+2..off+4].copy_from_slice(
                    &crate::quant::half::f32_to_f16(0.005).to_le_bytes());
            }
            let mut cpu = vec![0.0f32; n_blocks * 256];
            crate::quant::q5_k::dequantize_to_f32(&w, &mut cpu);
            check_bulk_dequant(&cache, "dequant_q5_k_f16",
                include_str!("../../kernels/dequant_q5_k_f16.cpp"), "dequant_q5_k_f16",
                &w, 256, 256, &cpu);
        }
        // Q8_0 — 32 w / 34 B.
        {
            let n_blocks = 256;
            let bpb = crate::quant::q8_0::BYTES_PER_BLOCK;
            let mut w = vec![0u8; n_blocks * bpb];
            for b in w.iter_mut() { *b = rng_u8(); }
            for blk in 0..n_blocks {
                let off = blk * bpb;
                w[off..off+2].copy_from_slice(
                    &crate::quant::half::f32_to_f16(0.02).to_le_bytes());
            }
            let mut cpu = vec![0.0f32; n_blocks * 32];
            crate::quant::q8_0::dequantize_to_f32(&w, &mut cpu);
            check_bulk_dequant(&cache, "dequant_q8_0_f16",
                include_str!("../../kernels/dequant_q8_0_f16.cpp"), "dequant_q8_0_f16",
                &w, 32, 32, &cpu);
        }
        // IQ4_XS — 256 w / 136 B.
        {
            let n_blocks = 64;
            let bpb = crate::quant::iq4_xs::BYTES_PER_BLOCK;
            let mut w = vec![0u8; n_blocks * bpb];
            for b in w.iter_mut() { *b = rng_u8(); }
            for blk in 0..n_blocks {
                let off = blk * bpb;
                w[off..off+2].copy_from_slice(
                    &crate::quant::half::f32_to_f16(0.005).to_le_bytes());
            }
            let mut cpu = vec![0.0f32; n_blocks * 256];
            crate::quant::iq4_xs::dequantize_to_f32(&w, &mut cpu);
            check_bulk_dequant(&cache, "dequant_iq4_xs_f16",
                include_str!("../../kernels/dequant_iq4_xs_f16.cpp"), "dequant_iq4_xs_f16",
                &w, 256, 256, &cpu);
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
