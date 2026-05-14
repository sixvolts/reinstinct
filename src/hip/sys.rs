//! Raw FFI bindings to `libamdhip64.so`, loaded at runtime via `dlopen`.
//!
//! We deliberately link nothing at build time — the only ROCm dependency is
//! that `libamdhip64.so` is resolvable by the dynamic linker at startup.

use std::ffi::{CStr, c_char, c_void};
use std::sync::OnceLock;

use libloading::Library;

#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct HipError(pub i32);

impl HipError {
    pub const SUCCESS: HipError = HipError(0);
    pub fn is_ok(self) -> bool { self.0 == 0 }
}

pub type HipDevice = i32;

#[repr(C)] pub struct HipStreamRaw   { _private: [u8; 0] }
#[repr(C)] pub struct HipModuleRaw   { _private: [u8; 0] }
#[repr(C)] pub struct HipFunctionRaw { _private: [u8; 0] }
pub type HipStream   = *mut HipStreamRaw;
pub type HipModule   = *mut HipModuleRaw;
pub type HipFunction = *mut HipFunctionRaw;

#[repr(i32)]
#[derive(Copy, Clone, Debug)]
pub enum HipMemcpyKind {
    HostToHost     = 0,
    HostToDevice   = 1,
    DeviceToHost   = 2,
    DeviceToDevice = 3,
    Default        = 4,
}

/// Bag of resolved function pointers. One of these is constructed once per
/// process by [`hip()`]; thereafter it's a pure plain-old-data table.
pub struct Hip {
    _lib: Library,
    pub get_device_count:    unsafe extern "C" fn(*mut i32) -> HipError,
    pub set_device:          unsafe extern "C" fn(i32) -> HipError,
    pub device_get_name:     unsafe extern "C" fn(*mut c_char, i32, HipDevice) -> HipError,
    pub device_total_mem:    unsafe extern "C" fn(*mut usize, HipDevice) -> HipError,
    pub mem_get_info:        unsafe extern "C" fn(*mut usize, *mut usize) -> HipError,
    pub get_error_string:    unsafe extern "C" fn(HipError) -> *const c_char,

    pub malloc:              unsafe extern "C" fn(*mut *mut c_void, usize) -> HipError,
    pub free:                unsafe extern "C" fn(*mut c_void) -> HipError,
    pub memcpy:              unsafe extern "C" fn(*mut c_void, *const c_void, usize, HipMemcpyKind) -> HipError,

    pub stream_create:       unsafe extern "C" fn(*mut HipStream) -> HipError,
    pub stream_destroy:      unsafe extern "C" fn(HipStream) -> HipError,
    pub stream_synchronize:  unsafe extern "C" fn(HipStream) -> HipError,
    pub device_synchronize:  unsafe extern "C" fn() -> HipError,

    pub module_load:         unsafe extern "C" fn(*mut HipModule, *const c_char) -> HipError,
    pub module_unload:       unsafe extern "C" fn(HipModule) -> HipError,
    pub module_get_function: unsafe extern "C" fn(*mut HipFunction, HipModule, *const c_char) -> HipError,
    pub module_launch_kernel: unsafe extern "C" fn(
        HipFunction, u32, u32, u32, u32, u32, u32, u32, HipStream,
        *mut *mut c_void, *mut *mut c_void) -> HipError,
}

// SAFETY: `Hip` is immutable after construction; all entry points are reentrant.
unsafe impl Send for Hip {}
unsafe impl Sync for Hip {}

impl Hip {
    /// Render a HIP error code into a human-readable message.
    pub fn err_str(&self, e: HipError) -> String {
        // SAFETY: hipGetErrorString returns a static C string, valid for any code.
        unsafe {
            let p = (self.get_error_string)(e);
            if p.is_null() {
                format!("unknown HIP error {}", e.0)
            } else {
                CStr::from_ptr(p).to_string_lossy().into_owned()
            }
        }
    }
}

static HIP: OnceLock<Result<Hip, String>> = OnceLock::new();

/// Resolve `libamdhip64.so` and bind every entry point we use. The first call
/// performs the dlopen; subsequent calls return the cached table.
pub fn hip() -> Result<&'static Hip, &'static str> {
    let cell = HIP.get_or_init(|| unsafe {
        let candidates = [
            "libamdhip64.so",
            "libamdhip64.so.7",
            "libamdhip64.so.6",
            "libamdhip64.so.5",
            "/opt/rocm/lib/libamdhip64.so",
            "/usr/lib/x86_64-linux-gnu/libamdhip64.so",
        ];
        let mut last_err = String::from("no candidate paths");
        let lib = candidates.iter().find_map(|p| match Library::new(p) {
            Ok(l) => Some(l),
            Err(e) => { last_err = format!("{p}: {e}"); None }
        }).ok_or_else(|| format!("could not load libamdhip64.so ({last_err})"))?;

        macro_rules! sym {
            ($name:literal) => {{
                let s: libloading::Symbol<unsafe extern "C" fn()> = lib.get($name)
                    .map_err(|e| format!("missing {}: {}", String::from_utf8_lossy($name), e))?;
                std::mem::transmute(*s)
            }};
        }

        Ok(Hip {
            get_device_count:     sym!(b"hipGetDeviceCount"),
            set_device:           sym!(b"hipSetDevice"),
            device_get_name:      sym!(b"hipDeviceGetName"),
            device_total_mem:     sym!(b"hipDeviceTotalMem"),
            mem_get_info:         sym!(b"hipMemGetInfo"),
            get_error_string:     sym!(b"hipGetErrorString"),
            malloc:               sym!(b"hipMalloc"),
            free:                 sym!(b"hipFree"),
            memcpy:               sym!(b"hipMemcpy"),
            stream_create:        sym!(b"hipStreamCreate"),
            stream_destroy:       sym!(b"hipStreamDestroy"),
            stream_synchronize:   sym!(b"hipStreamSynchronize"),
            device_synchronize:   sym!(b"hipDeviceSynchronize"),
            module_load:          sym!(b"hipModuleLoad"),
            module_unload:        sym!(b"hipModuleUnload"),
            module_get_function:  sym!(b"hipModuleGetFunction"),
            module_launch_kernel: sym!(b"hipModuleLaunchKernel"),
            _lib: lib,
        })
    });
    cell.as_ref().map_err(|s| s.as_str())
}
