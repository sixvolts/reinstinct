//! End-to-end HIP smoke test: compile a trivial kernel through
//! [`KernelCache`], load it, launch it, and read results back.
//! Not part of the inference path — exists to prove the chain works.

use std::ffi::c_void;

use super::KernelCache;
use crate::hip::{self, DeviceBuf, Module};

const VECTOR_ADD_SOURCE: &str = include_str!("../../kernels/vector_add.cpp");
const VECTOR_ADD_KERNEL: &str = "vector_add_f32";

/// Compute `y = a + b` on the GPU. Allocates per call — testing only.
pub fn vector_add_smoke(cache: &KernelCache, a: &[f32], b: &[f32]) -> Result<Vec<f32>, String> {
    assert_eq!(a.len(), b.len(), "vector_add length mismatch");
    let n = a.len();

    let hsaco = cache.compile("vector_add", VECTOR_ADD_SOURCE)?;
    let module = Module::load(&hsaco)?;
    let f = module.function(VECTOR_ADD_KERNEL)?;

    let da: DeviceBuf<f32> = DeviceBuf::from_slice(a)?;
    let db: DeviceBuf<f32> = DeviceBuf::from_slice(b)?;
    let dy: DeviceBuf<f32> = DeviceBuf::new(n)?;

    let block = 256u32;
    let grid = ((n as u32) + block - 1) / block;

    // Args: kernelParams is an array of pointers, each pointing at the
    // memory holding one kernel argument's value. Pointer args carry a
    // *mut c_void (the device pointer); scalar args carry their value.
    let mut a_ptr = da.raw_ptr();
    let mut b_ptr = db.raw_ptr();
    let mut y_ptr = dy.raw_ptr();
    let mut n_arg = n as u32;
    let mut args: [*mut c_void; 4] = [
        &mut a_ptr  as *mut _ as *mut c_void,
        &mut b_ptr  as *mut _ as *mut c_void,
        &mut y_ptr  as *mut _ as *mut c_void,
        &mut n_arg  as *mut _ as *mut c_void,
    ];

    // SAFETY: signature matches the kernel, all args live until sync below.
    unsafe { f.launch((grid, 1, 1), (block, 1, 1), 0, None, &mut args)?; }
    hip::Device(0).synchronize()?;  // any device is fine; ensures the launch finished

    let mut out = vec![0.0f32; n];
    dy.copy_to_host(&mut out)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_add_matches_cpu() {
        if hip::device_count().ok().unwrap_or(0) < 1 {
            eprintln!("skip: no HIP device"); return;
        }
        let _dev = hip::Device::set(0).expect("set device 0");
        let cache = match KernelCache::new() {
            Ok(c) => c,
            Err(e) => { eprintln!("skip: kernel cache unavailable: {e}"); return; }
        };
        let n = 10_000usize;
        let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
        let b: Vec<f32> = (0..n).map(|i| (i as f32) * -0.25 + 1.0).collect();

        let y = vector_add_smoke(&cache, &a, &b).expect("smoke launch");
        assert_eq!(y.len(), n);
        for i in 0..n {
            let expect = a[i] + b[i];
            assert_eq!(y[i].to_bits(), expect.to_bits(),
                       "mismatch at {i}: {} vs {}", y[i], expect);
        }
    }
}
