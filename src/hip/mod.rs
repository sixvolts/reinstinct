//! Runtime `dlopen` of `libamdhip64.so` and safe Rust wrappers.
//!
//! No link-time ROCm dependency — the engine binary runs against any
//! ROCm 5.7 / 6.x / 7.x runtime that exposes `libamdhip64.so`.

pub mod sys;

use std::ffi::{CString, c_char, c_void};
use std::marker::PhantomData;
use std::ptr::null_mut;

use sys::{Hip, HipDevice, HipError, HipFunction, HipGraph, HipGraphExec, HipMemcpyKind,
          HipModule, HipStream, HipStreamCaptureMode, hip};

/// Result type for the safe HIP API. The error message has already been
/// rendered via `hipGetErrorString` (or describes a load failure).
pub type Result<T> = std::result::Result<T, String>;

#[inline]
fn ck(api: &Hip, e: HipError, ctx: &str) -> Result<()> {
    if e.is_ok() { Ok(()) } else { Err(format!("{ctx}: {} (code {})", api.err_str(e), e.0)) }
}

/// Number of HIP devices the runtime sees.
pub fn device_count() -> Result<i32> {
    let api = hip().map_err(|s| s.to_string())?;
    let mut n = 0i32;
    unsafe { ck(api, (api.get_device_count)(&mut n), "hipGetDeviceCount")?; }
    Ok(n)
}

/// Marketing name of `device` (e.g. "AMD Instinct MI60 / MI50").
pub fn device_name(device: HipDevice) -> Result<String> {
    let api = hip().map_err(|s| s.to_string())?;
    let mut buf = vec![0u8; 256];
    unsafe {
        ck(api, (api.device_get_name)(buf.as_mut_ptr() as *mut c_char, buf.len() as i32, device),
           "hipDeviceGetName")?;
    }
    let nul = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    Ok(String::from_utf8_lossy(&buf[..nul]).into_owned())
}

/// Total VRAM in bytes for `device`.
pub fn device_total_mem(device: HipDevice) -> Result<usize> {
    let api = hip().map_err(|s| s.to_string())?;
    let mut bytes = 0usize;
    unsafe { ck(api, (api.device_total_mem)(&mut bytes, device), "hipDeviceTotalMem")?; }
    Ok(bytes)
}

/// (free, total) VRAM for the *currently-selected* device.
pub fn mem_info() -> Result<(usize, usize)> {
    let api = hip().map_err(|s| s.to_string())?;
    let (mut free, mut total) = (0usize, 0usize);
    unsafe { ck(api, (api.mem_get_info)(&mut free, &mut total), "hipMemGetInfo")?; }
    Ok((free, total))
}

/// RAII handle to a HIP device. Setting it switches the runtime's active
/// device for this thread.
pub struct Device(pub HipDevice);

impl Device {
    pub fn set(id: HipDevice) -> Result<Self> {
        let api = hip().map_err(|s| s.to_string())?;
        unsafe { ck(api, (api.set_device)(id), "hipSetDevice")?; }
        Ok(Device(id))
    }

    pub fn synchronize(&self) -> Result<()> {
        let api = hip().map_err(|s| s.to_string())?;
        unsafe { ck(api, (api.device_synchronize)(), "hipDeviceSynchronize") }
    }
}

/// RAII HIP stream. Drops issue `hipStreamDestroy`.
pub struct Stream { raw: HipStream }

impl Stream {
    pub fn new() -> Result<Self> {
        let api = hip().map_err(|s| s.to_string())?;
        let mut s: HipStream = null_mut();
        unsafe { ck(api, (api.stream_create)(&mut s), "hipStreamCreate")?; }
        Ok(Stream { raw: s })
    }

    pub fn synchronize(&self) -> Result<()> {
        let api = hip().map_err(|s| s.to_string())?;
        unsafe { ck(api, (api.stream_synchronize)(self.raw), "hipStreamSynchronize") }
    }

    pub fn raw(&self) -> HipStream { self.raw }
}

impl Drop for Stream {
    fn drop(&mut self) {
        if let Ok(api) = hip() {
            unsafe { let _ = (api.stream_destroy)(self.raw); }
        }
    }
}

/// Owned device-side allocation of `len` Ts. Drops issue `hipFree`.
pub struct DeviceBuf<T> {
    ptr: *mut T,
    len: usize,
    _phantom: PhantomData<T>,
}

unsafe impl<T: Send> Send for DeviceBuf<T> {}
unsafe impl<T: Sync> Sync for DeviceBuf<T> {}

impl<T: Copy> DeviceBuf<T> {
    /// Allocate `len` Ts on the device. Contents are uninitialised.
    pub fn new(len: usize) -> Result<Self> {
        let api = hip().map_err(|s| s.to_string())?;
        let mut p: *mut c_void = null_mut();
        let bytes = len * std::mem::size_of::<T>();
        unsafe { ck(api, (api.malloc)(&mut p, bytes), "hipMalloc")?; }
        Ok(DeviceBuf { ptr: p as *mut T, len, _phantom: PhantomData })
    }

    /// Allocate and copy `src` H2D in one shot.
    pub fn from_slice(src: &[T]) -> Result<Self> {
        let buf = Self::new(src.len())?;
        buf.copy_from_host(src)?;
        Ok(buf)
    }

    pub fn len(&self) -> usize { self.len }
    pub fn is_empty(&self) -> bool { self.len == 0 }
    pub fn as_ptr(&self) -> *const T { self.ptr }
    pub fn as_mut_ptr(&mut self) -> *mut T { self.ptr }
    /// Raw device pointer as `*mut c_void`, suitable for kernel-arg packing.
    pub fn raw_ptr(&self) -> *mut c_void { self.ptr as *mut c_void }
    pub fn byte_len(&self) -> usize { self.len * std::mem::size_of::<T>() }

    pub fn copy_from_host(&self, src: &[T]) -> Result<()> {
        assert_eq!(src.len(), self.len, "copy_from_host length mismatch");
        let api = hip().map_err(|s| s.to_string())?;
        unsafe {
            ck(api, (api.memcpy)(self.ptr as *mut c_void, src.as_ptr() as *const c_void,
                                  self.byte_len(), HipMemcpyKind::HostToDevice),
               "hipMemcpy H2D")
        }
    }

    pub fn copy_to_host(&self, dst: &mut [T]) -> Result<()> {
        assert_eq!(dst.len(), self.len, "copy_to_host length mismatch");
        let api = hip().map_err(|s| s.to_string())?;
        unsafe {
            ck(api, (api.memcpy)(dst.as_mut_ptr() as *mut c_void, self.ptr as *const c_void,
                                  self.byte_len(), HipMemcpyKind::DeviceToHost),
               "hipMemcpy D2H")
        }
    }

    /// D2D copy: write all of `src` into `self` starting at element index
    /// `dst_offset`. `dst_offset + src.len()` must be within `self.len()`.
    pub fn copy_from_device_at(&self, src: &DeviceBuf<T>, dst_offset: usize) -> Result<()> {
        assert!(dst_offset + src.len <= self.len,
                "copy_from_device_at: dst_offset+src.len ({}) exceeds self.len ({})",
                dst_offset + src.len, self.len);
        let api = hip().map_err(|s| s.to_string())?;
        unsafe {
            let dst_ptr = self.ptr.add(dst_offset) as *mut c_void;
            ck(api, (api.memcpy)(dst_ptr, src.ptr as *const c_void,
                                  src.byte_len(), HipMemcpyKind::DeviceToDevice),
               "hipMemcpy D2D")
        }
    }

    /// Async D2D copy on `stream` — required for HIP graph capture, where
    /// blocking memcpy on the null stream cannot be captured. Ordering
    /// against preceding kernel launches on the same stream is preserved
    /// by stream semantics.
    pub fn copy_from_device_at_async(&self, src: &DeviceBuf<T>, dst_offset: usize, stream: &Stream)
        -> Result<()>
    {
        assert!(dst_offset + src.len <= self.len,
                "copy_from_device_at_async: dst_offset+src.len ({}) exceeds self.len ({})",
                dst_offset + src.len, self.len);
        let api = hip().map_err(|s| s.to_string())?;
        unsafe {
            let dst_ptr = self.ptr.add(dst_offset) as *mut c_void;
            ck(api, (api.memcpy_async)(dst_ptr, src.ptr as *const c_void,
                                        src.byte_len(), HipMemcpyKind::DeviceToDevice, stream.raw),
               "hipMemcpyAsync D2D")
        }
    }
}

impl<T> Drop for DeviceBuf<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            if let Ok(api) = hip() {
                unsafe { let _ = (api.free)(self.ptr as *mut c_void); }
            }
        }
    }
}

/// Captured HIP graph (RAII). Created by ending a stream capture.
pub struct Graph { raw: HipGraph }

impl Graph {
    /// Capture all HIP operations issued on `stream` between
    /// `Graph::begin_capture(...)` and `Graph::end_capture(...)`.
    /// Mode `Global` is the standard choice for our use case.
    pub fn begin_capture(stream: &Stream, mode: HipStreamCaptureMode) -> Result<()> {
        let api = hip().map_err(|s| s.to_string())?;
        unsafe { ck(api, (api.stream_begin_capture)(stream.raw, mode), "hipStreamBeginCapture") }
    }

    /// End capture on `stream` and return the captured graph.
    pub fn end_capture(stream: &Stream) -> Result<Self> {
        let api = hip().map_err(|s| s.to_string())?;
        let mut g: HipGraph = null_mut();
        unsafe { ck(api, (api.stream_end_capture)(stream.raw, &mut g), "hipStreamEndCapture")?; }
        Ok(Graph { raw: g })
    }

    pub fn instantiate(&self) -> Result<GraphExec> {
        let api = hip().map_err(|s| s.to_string())?;
        let mut e: HipGraphExec = null_mut();
        unsafe { ck(api, (api.graph_instantiate)(&mut e, self.raw, null_mut(), null_mut(), 0),
                   "hipGraphInstantiate")?; }
        Ok(GraphExec { raw: e })
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            if let Ok(api) = hip() { unsafe { let _ = (api.graph_destroy)(self.raw); } }
        }
    }
}

/// Instantiated executable HIP graph (RAII).
pub struct GraphExec { raw: HipGraphExec }

impl GraphExec {
    /// Submit the captured chain to `stream`. The submission is async —
    /// caller must sync the stream (or the device) before reading results.
    pub fn launch(&self, stream: &Stream) -> Result<()> {
        let api = hip().map_err(|s| s.to_string())?;
        unsafe { ck(api, (api.graph_launch)(self.raw, stream.raw), "hipGraphLaunch") }
    }
}

impl Drop for GraphExec {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            if let Ok(api) = hip() { unsafe { let _ = (api.graph_exec_destroy)(self.raw); } }
        }
    }
}

/// RAII handle to a loaded `.hsaco` module.
pub struct Module { raw: HipModule }

impl Module {
    /// Load a `.hsaco` from disk. Path must be valid UTF-8.
    pub fn load(path: &std::path::Path) -> Result<Self> {
        let api = hip().map_err(|s| s.to_string())?;
        let cpath = CString::new(path.to_str().ok_or("module path is not UTF-8")?)
            .map_err(|e| format!("path contains NUL: {e}"))?;
        let mut m: HipModule = null_mut();
        unsafe { ck(api, (api.module_load)(&mut m, cpath.as_ptr()), "hipModuleLoad")?; }
        Ok(Module { raw: m })
    }

    /// Look up a kernel symbol by name. The returned function borrows `self`.
    pub fn function<'m>(&'m self, name: &str) -> Result<Function<'m>> {
        let api = hip().map_err(|s| s.to_string())?;
        let cname = CString::new(name).map_err(|e| format!("kernel name has NUL: {e}"))?;
        let mut f: HipFunction = null_mut();
        unsafe { ck(api, (api.module_get_function)(&mut f, self.raw, cname.as_ptr()),
                    "hipModuleGetFunction")?; }
        Ok(Function { raw: f, _phantom: PhantomData })
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            if let Ok(api) = hip() {
                unsafe { let _ = (api.module_unload)(self.raw); }
            }
        }
    }
}

/// Borrowed reference to a kernel function inside a [`Module`].
pub struct Function<'m> {
    raw: HipFunction,
    _phantom: PhantomData<&'m Module>,
}

impl<'m> Function<'m> {
    /// Launch with explicit grid/block dims and an array of pointers to args.
    ///
    /// `args` is an array of pointers — each pointing at a kernel-arg value
    /// living in caller-owned memory (typically stack locals). The pointer
    /// array is passed via `kernelParams`; we never use the legacy `extra`.
    ///
    /// # Safety
    /// Caller must ensure each `args[i]` points at a value matching the
    /// kernel's `i`th parameter type and lifetime spans the launch.
    pub unsafe fn launch(
        &self,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem: u32,
        stream: Option<&Stream>,
        args: &mut [*mut c_void],
    ) -> Result<()> {
        let api = hip().map_err(|s| s.to_string())?;
        let s = stream.map(|s| s.raw()).unwrap_or(null_mut());
        unsafe {
            ck(api, (api.module_launch_kernel)(
                self.raw, grid.0, grid.1, grid.2, block.0, block.1, block.2,
                shared_mem, s, args.as_mut_ptr(), null_mut()), "hipModuleLaunchKernel")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Basic load + device query. Skips silently if no HIP runtime is present.
    #[test]
    fn load_and_query() {
        let _api = match hip() { Ok(a) => a, Err(e) => { eprintln!("skip: {e}"); return; } };
        let n = match device_count() { Ok(n) => n, Err(e) => { eprintln!("skip: {e}"); return; } };
        eprintln!("hip device count = {n}");
        if n == 0 { eprintln!("no devices, skipping"); return; }
        let _ = Device::set(0).expect("set device 0");
        let name = device_name(0).expect("device name");
        let total = device_total_mem(0).expect("total mem");
        let (free, total2) = mem_info().expect("mem info");
        eprintln!("dev0: {name}, total {} GB, free {} / {} GB",
                  total / (1<<30), free / (1<<30), total2 / (1<<30));
    }

    /// H2D + D2H roundtrip on real device 0.
    #[test]
    fn roundtrip_buffer() {
        if device_count().ok().unwrap_or(0) < 1 { eprintln!("skip: no HIP device"); return; }
        let _dev = Device::set(0).unwrap();
        let src: Vec<f32> = (0..4096).map(|i| (i as f32) * 0.5).collect();
        let buf = DeviceBuf::from_slice(&src).expect("alloc + h2d");
        let mut back = vec![0.0f32; src.len()];
        buf.copy_to_host(&mut back).expect("d2h");
        for i in 0..src.len() {
            assert_eq!(src[i].to_bits(), back[i].to_bits(),
                       "round-trip mismatch at {i}: {} vs {}", src[i], back[i]);
        }
    }
}
