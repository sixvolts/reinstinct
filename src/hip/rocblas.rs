//! Runtime `dlopen` of `librocblas.so` for prefill HGEMM.
//!
//! gfx906 still has tuned Tensile HGEMM kernels in the rocBLAS 5.x
//! library installed under `/usr/lib/x86_64-linux-gnu/rocblas/5.1.0/`,
//! so on this system we can use rocBLAS as the prefill workhorse.
//!
//! This module only handles the FFI + a thin RAII handle wrapper +
//! a single typed `hgemm` call. Integration into a multi-token prefill
//! path lives in `runtime/qwen35.rs` (and needs bulk dequant kernels
//! to materialise fp16 weights from the on-disk Q4_K bytes).

use std::ffi::c_void;
use std::ptr::null_mut;
use std::sync::OnceLock;

use libloading::Library;

use super::sys::HipStream;
use super::Stream;

#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct RocblasStatus(pub i32);

impl RocblasStatus {
    pub const SUCCESS: RocblasStatus = RocblasStatus(0);
    pub fn is_ok(self) -> bool { self.0 == 0 }
}

#[repr(C)] pub struct RocblasHandleRaw { _private: [u8; 0] }
pub type RocblasHandle = *mut RocblasHandleRaw;

#[repr(i32)]
#[derive(Copy, Clone, Debug)]
pub enum RocblasOp {
    None      = 111,
    Transpose = 112,
}

#[repr(i32)]
#[derive(Copy, Clone, Debug)]
pub enum RocblasDatatype {
    F16R = 150,
    F32R = 151,
}

pub struct Rocblas {
    _lib: Library,
    pub create_handle:  unsafe extern "C" fn(*mut RocblasHandle) -> RocblasStatus,
    pub destroy_handle: unsafe extern "C" fn(RocblasHandle) -> RocblasStatus,
    pub set_stream:     unsafe extern "C" fn(RocblasHandle, HipStream) -> RocblasStatus,
    /// `rocblas_hgemm` — fp16 matrix multiply. `alpha` and `beta` are
    /// pointers to *fp16* scalars (passed via raw u16 bits here so we
    /// don't need a half-precision Rust type). All matrices are
    /// column-major following BLAS convention.
    pub hgemm: unsafe extern "C" fn(
        RocblasHandle, RocblasOp, RocblasOp,
        i32, i32, i32,                        // m, n, k
        *const u16,                            // alpha (fp16 bits)
        *const u16, i32,                       // A, lda
        *const u16, i32,                       // B, ldb
        *const u16,                            // beta (fp16 bits)
        *mut u16, i32) -> RocblasStatus,       // C, ldc
    /// `rocblas_gemm_ex` — typed GEMM. We use it for fp16-storage /
    /// fp32-compute (HPA): fp16 A/B/C/D, fp32 alpha/beta/accumulate.
    pub gemm_ex: unsafe extern "C" fn(
        RocblasHandle, RocblasOp, RocblasOp,
        i32, i32, i32,                          // m, n, k
        *const c_void,                          // alpha
        *const c_void, RocblasDatatype, i32,    // A, a_type, lda
        *const c_void, RocblasDatatype, i32,    // B, b_type, ldb
        *const c_void,                          // beta
        *const c_void, RocblasDatatype, i32,    // C, c_type, ldc
        *mut c_void,   RocblasDatatype, i32,    // D, d_type, ldd
        RocblasDatatype,                        // compute_type
        i32, i32, u32) -> RocblasStatus,        // algo, solution_index, flags
}

unsafe impl Send for Rocblas {}
unsafe impl Sync for Rocblas {}

static ROCBLAS: OnceLock<Result<Rocblas, String>> = OnceLock::new();

/// Resolve `librocblas.so` and bind the entry points we need. First call
/// dlopens; subsequent calls return the cached table.
pub fn rocblas() -> Result<&'static Rocblas, &'static str> {
    let cell = ROCBLAS.get_or_init(|| unsafe {
        let candidates = [
            "librocblas.so",
            "librocblas.so.5",
            "librocblas.so.4",
            "/usr/lib/x86_64-linux-gnu/librocblas.so",
            "/opt/rocm/lib/librocblas.so",
        ];
        let mut last_err = String::from("no candidate paths");
        let lib = candidates.iter().find_map(|p| match Library::new(p) {
            Ok(l) => Some(l),
            Err(e) => { last_err = format!("{p}: {e}"); None }
        }).ok_or_else(|| format!("could not load librocblas.so ({last_err})"))?;

        macro_rules! sym {
            ($name:literal) => {{
                let s: libloading::Symbol<unsafe extern "C" fn()> = lib.get($name)
                    .map_err(|e| format!("missing {}: {}", String::from_utf8_lossy($name), e))?;
                std::mem::transmute(*s)
            }};
        }

        Ok(Rocblas {
            create_handle:  sym!(b"rocblas_create_handle"),
            destroy_handle: sym!(b"rocblas_destroy_handle"),
            set_stream:     sym!(b"rocblas_set_stream"),
            hgemm:          sym!(b"rocblas_hgemm"),
            gemm_ex:        sym!(b"rocblas_gemm_ex"),
            _lib: lib,
        })
    });
    cell.as_ref().map_err(|s| s.as_str())
}

/// RAII handle. Dropping it issues `rocblas_destroy_handle`. Bind a
/// stream once via `set_stream` before issuing GEMMs.
pub struct Handle { raw: RocblasHandle }

impl Handle {
    pub fn new() -> Result<Self, String> {
        let api = rocblas().map_err(|s| s.to_string())?;
        let mut h: RocblasHandle = null_mut();
        let s = unsafe { (api.create_handle)(&mut h) };
        if !s.is_ok() { return Err(format!("rocblas_create_handle: status {}", s.0)); }
        Ok(Handle { raw: h })
    }

    pub fn set_stream(&self, stream: &Stream) -> Result<(), String> {
        let api = rocblas().map_err(|s| s.to_string())?;
        let s = unsafe { (api.set_stream)(self.raw, stream.raw()) };
        if !s.is_ok() { Err(format!("rocblas_set_stream: status {}", s.0)) } else { Ok(()) }
    }

    /// `C ← α · op(A) · op(B) + β · C`, all matrices fp16 column-major.
    /// Sizes follow BLAS conventions: `op(A)` is m×k, `op(B)` is k×n,
    /// `C` is m×n. Strides are leading dimensions (column count of the
    /// untransposed matrix). Pointers reference *device* memory.
    ///
    /// # Safety
    /// Caller is responsible for sizing the device allocations correctly
    /// and for keeping them alive for the duration of the call (returns
    /// once the launch is enqueued on the bound stream, not when it
    /// completes — sync the stream before reading C).
    pub unsafe fn hgemm(&self,
        trans_a: RocblasOp, trans_b: RocblasOp,
        m: i32, n: i32, k: i32,
        alpha_bits: u16,
        a: *const c_void, lda: i32,
        b: *const c_void, ldb: i32,
        beta_bits: u16,
        c: *mut c_void, ldc: i32,
    ) -> Result<(), String>
    {
        let api = rocblas().map_err(|s| s.to_string())?;
        // alpha/beta are passed by pointer — keep them on the stack for
        // the duration of the call.
        let alpha = alpha_bits;
        let beta  = beta_bits;
        let s = unsafe {
            (api.hgemm)(
                self.raw, trans_a, trans_b, m, n, k,
                &alpha,
                a as *const u16, lda,
                b as *const u16, ldb,
                &beta,
                c as *mut u16, ldc,
            )
        };
        if !s.is_ok() { Err(format!("rocblas_hgemm: status {}", s.0)) } else { Ok(()) }
    }

    /// fp16-storage / fp32-compute GEMM via `rocblas_gemm_ex`:
    /// `D ← α · op(A) · op(B) + β · C`, A/B/C/D all fp16, but alpha,
    /// beta, and the inner accumulation are fp32 — essential for long
    /// reductions where plain `hgemm`'s fp16 accumulate loses precision.
    ///
    /// C and D point at the same buffer here (in-place); β = 0.
    ///
    /// # Safety
    /// Caller sizes the device buffers and keeps them alive across the
    /// (async) launch.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_f16_f32acc(&self,
        trans_a: RocblasOp, trans_b: RocblasOp,
        m: i32, n: i32, k: i32,
        alpha: f32,
        a: *const c_void, lda: i32,
        b: *const c_void, ldb: i32,
        beta: f32,
        c: *mut c_void, ldc: i32,
    ) -> Result<(), String>
    {
        let api = rocblas().map_err(|s| s.to_string())?;
        let alpha = alpha;  // fp32 scalars, passed by pointer
        let beta  = beta;
        let s = unsafe {
            (api.gemm_ex)(
                self.raw, trans_a, trans_b, m, n, k,
                &alpha as *const f32 as *const c_void,
                a, RocblasDatatype::F16R, lda,
                b, RocblasDatatype::F16R, ldb,
                &beta as *const f32 as *const c_void,
                c as *const c_void, RocblasDatatype::F16R, ldc,
                c, RocblasDatatype::F16R, ldc,
                RocblasDatatype::F32R,   // fp32 accumulate
                0,  // rocblas_gemm_algo_standard
                0,  // solution_index
                0,  // flags
            )
        };
        if !s.is_ok() { Err(format!("rocblas_gemm_ex: status {}", s.0)) } else { Ok(()) }
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            if let Ok(api) = rocblas() { unsafe { let _ = (api.destroy_handle)(self.raw); } }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hip::{self, DeviceBuf, Stream};
    use crate::quant::half::{f32_to_f16, f16_to_f32};

    /// Tiny C = A · B HGEMM end-to-end on real hardware.
    #[test]
    fn hgemm_matches_cpu_reference() {
        if hip::device_count().ok().unwrap_or(0) < 1 { eprintln!("skip"); return; }
        let _dev = hip::Device::set(0).unwrap();
        let stream = Stream::new().expect("stream");
        let h = match Handle::new() {
            Ok(h) => h,
            Err(e) => { eprintln!("skip: {e}"); return; }
        };
        h.set_stream(&stream).expect("set_stream");

        // C = A * B with A: 4×3, B: 3×5, C: 4×5 (BLAS column-major).
        let m: i32 = 4;
        let n: i32 = 5;
        let k: i32 = 3;

        // Generate column-major A and B as fp32 first, then convert to fp16.
        // A[i,j] = i + 0.1 * j  → stored at A_col[j*m + i]
        // B[i,j] = j - 0.2 * i  → stored at B_col[j*k + i]
        let mut a_f32 = vec![0.0f32; (m*k) as usize];
        let mut b_f32 = vec![0.0f32; (k*n) as usize];
        for i in 0..m { for j in 0..k {
            a_f32[(j*m + i) as usize] = i as f32 + 0.1 * j as f32;
        }}
        for i in 0..k { for j in 0..n {
            b_f32[(j*k + i) as usize] = j as f32 - 0.2 * i as f32;
        }}

        let a_f16: Vec<u16> = a_f32.iter().map(|&v| f32_to_f16(v)).collect();
        let b_f16: Vec<u16> = b_f32.iter().map(|&v| f32_to_f16(v)).collect();

        let da = DeviceBuf::<u16>::from_slice(&a_f16).unwrap();
        let db = DeviceBuf::<u16>::from_slice(&b_f16).unwrap();
        let dc = DeviceBuf::<u16>::new((m*n) as usize).unwrap();

        unsafe {
            h.hgemm(
                RocblasOp::None, RocblasOp::None,
                m, n, k,
                f32_to_f16(1.0),
                da.as_ptr() as *const std::ffi::c_void, m,
                db.as_ptr() as *const std::ffi::c_void, k,
                f32_to_f16(0.0),
                dc.as_ptr() as *mut std::ffi::c_void, m,
            ).expect("hgemm");
        }
        stream.synchronize().expect("sync");

        let mut c_f16 = vec![0u16; (m*n) as usize];
        dc.copy_to_host(&mut c_f16).unwrap();

        // CPU reference: C[i,j] = Σ_l A[i,l] * B[l,j]. Indices into the
        // column-major storage are A[l*m + i] and B[j*k + l]; result
        // also column-major, C[j*m + i].
        let mut max_abs = 0.0f32;
        for i in 0..m { for j in 0..n {
            let mut acc = 0.0f32;
            for l in 0..k {
                acc += a_f32[(l*m + i) as usize] * b_f32[(j*k + l) as usize];
            }
            let got = f16_to_f32(c_f16[(j*m + i) as usize]);
            let d = (got - acc).abs();
            if d > max_abs { max_abs = d; }
        }}
        eprintln!("rocblas hgemm 4×3 · 3×5: max_abs vs fp32 ref = {max_abs:.3e}");
        // fp16 product accumulation can drift by O(1e-2) on small shapes
        // — the test exists to prove the call goes through, not to bound
        // numerical error tightly.
        assert!(max_abs < 5e-2, "hgemm result diverges by {max_abs:.3e}");
    }
}
