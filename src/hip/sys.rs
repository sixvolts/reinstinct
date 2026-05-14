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

#[repr(C)] pub struct HipStreamRaw    { _private: [u8; 0] }
#[repr(C)] pub struct HipModuleRaw    { _private: [u8; 0] }
#[repr(C)] pub struct HipFunctionRaw  { _private: [u8; 0] }
#[repr(C)] pub struct HipGraphRaw     { _private: [u8; 0] }
#[repr(C)] pub struct HipGraphExecRaw { _private: [u8; 0] }
#[repr(C)] pub struct HipGraphNodeRaw { _private: [u8; 0] }
#[repr(C)] pub struct HipEventRaw     { _private: [u8; 0] }
pub type HipStream    = *mut HipStreamRaw;
pub type HipModule    = *mut HipModuleRaw;
pub type HipFunction  = *mut HipFunctionRaw;
pub type HipGraph     = *mut HipGraphRaw;
pub type HipGraphExec = *mut HipGraphExecRaw;
pub type HipGraphNode = *mut HipGraphNodeRaw;
pub type HipEvent     = *mut HipEventRaw;

#[repr(i32)]
#[derive(Copy, Clone, Debug)]
pub enum HipStreamCaptureMode {
    Global      = 0,
    ThreadLocal = 1,
    Relaxed     = 2,
}

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
    pub memcpy_async:        unsafe extern "C" fn(*mut c_void, *const c_void, usize, HipMemcpyKind, HipStream) -> HipError,

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

    pub stream_begin_capture: unsafe extern "C" fn(HipStream, HipStreamCaptureMode) -> HipError,
    pub stream_end_capture:   unsafe extern "C" fn(HipStream, *mut HipGraph) -> HipError,
    pub graph_instantiate:    unsafe extern "C" fn(*mut HipGraphExec, HipGraph,
                                                    *mut HipGraphNode, *mut c_char, usize) -> HipError,
    pub graph_launch:         unsafe extern "C" fn(HipGraphExec, HipStream) -> HipError,
    pub graph_destroy:        unsafe extern "C" fn(HipGraph) -> HipError,
    pub graph_exec_destroy:   unsafe extern "C" fn(HipGraphExec) -> HipError,

    pub event_create:         unsafe extern "C" fn(*mut HipEvent) -> HipError,
    pub event_record:         unsafe extern "C" fn(HipEvent, HipStream) -> HipError,
    pub event_synchronize:    unsafe extern "C" fn(HipEvent) -> HipError,
    pub event_elapsed_time:   unsafe extern "C" fn(*mut f32, HipEvent, HipEvent) -> HipError,
    pub event_destroy:        unsafe extern "C" fn(HipEvent) -> HipError,
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
            memcpy_async:         sym!(b"hipMemcpyAsync"),
            stream_create:        sym!(b"hipStreamCreate"),
            stream_destroy:       sym!(b"hipStreamDestroy"),
            stream_synchronize:   sym!(b"hipStreamSynchronize"),
            device_synchronize:   sym!(b"hipDeviceSynchronize"),
            module_load:          sym!(b"hipModuleLoad"),
            module_unload:        sym!(b"hipModuleUnload"),
            module_get_function:  sym!(b"hipModuleGetFunction"),
            module_launch_kernel: sym!(b"hipModuleLaunchKernel"),
            stream_begin_capture: sym!(b"hipStreamBeginCapture"),
            stream_end_capture:   sym!(b"hipStreamEndCapture"),
            graph_instantiate:    sym!(b"hipGraphInstantiate"),
            graph_launch:         sym!(b"hipGraphLaunch"),
            graph_destroy:        sym!(b"hipGraphDestroy"),
            graph_exec_destroy:   sym!(b"hipGraphExecDestroy"),
            event_create:         sym!(b"hipEventCreate"),
            event_record:         sym!(b"hipEventRecord"),
            event_synchronize:    sym!(b"hipEventSynchronize"),
            event_elapsed_time:   sym!(b"hipEventElapsedTime"),
            event_destroy:        sym!(b"hipEventDestroy"),
            _lib: lib,
        })
    });
    cell.as_ref().map_err(|s| s.as_str())
}
