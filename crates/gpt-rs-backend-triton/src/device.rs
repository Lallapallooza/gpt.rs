use std::ffi::c_void;
use std::fmt;
use std::sync::{Arc, OnceLock};

use gpt_rs::backend::spec::{BackendError, BackendResult};
use libloading::Library;

type CUresult = i32;
type CUdevice = i32;
type CUcontext = *mut c_void;
type CUdeviceptr = u64;

const CUDA_SUCCESS: CUresult = 0;

type CuInitFn = unsafe extern "C" fn(flags: u32) -> CUresult;
type CuDeviceGetFn = unsafe extern "C" fn(device: *mut CUdevice, ordinal: i32) -> CUresult;
type CuCtxCreateV2Fn =
    unsafe extern "C" fn(ctx: *mut CUcontext, flags: u32, dev: CUdevice) -> CUresult;
type CuCtxDestroyV2Fn = unsafe extern "C" fn(ctx: CUcontext) -> CUresult;
type CuCtxSetCurrentFn = unsafe extern "C" fn(ctx: CUcontext) -> CUresult;
type CuMemAllocV2Fn = unsafe extern "C" fn(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult;
type CuMemFreeV2Fn = unsafe extern "C" fn(dptr: CUdeviceptr) -> CUresult;
type CuMemcpyHtoDV2Fn = unsafe extern "C" fn(
    dst_device: CUdeviceptr,
    src_host: *const c_void,
    byte_count: usize,
) -> CUresult;
type CuMemcpyDtoHV2Fn = unsafe extern "C" fn(
    dst_host: *mut c_void,
    src_device: CUdeviceptr,
    byte_count: usize,
) -> CUresult;
type CuMemsetD8V2Fn =
    unsafe extern "C" fn(dst_device: CUdeviceptr, value: u8, count: usize) -> CUresult;

struct DriverFns {
    cu_init: CuInitFn,
    cu_device_get: CuDeviceGetFn,
    cu_ctx_create_v2: CuCtxCreateV2Fn,
    cu_ctx_destroy_v2: CuCtxDestroyV2Fn,
    cu_ctx_set_current: CuCtxSetCurrentFn,
    cu_mem_alloc_v2: CuMemAllocV2Fn,
    cu_mem_free_v2: CuMemFreeV2Fn,
    cu_memcpy_hto_d_v2: CuMemcpyHtoDV2Fn,
    cu_memcpy_dto_h_v2: CuMemcpyDtoHV2Fn,
    cu_memset_d8_v2: CuMemsetD8V2Fn,
}

pub struct CudaDriver {
    _lib: Library,
    fns: DriverFns,
    // Stored as usize so CudaDriver can satisfy Send/Sync requirements for backend traits.
    ctx: usize,
}

impl Drop for CudaDriver {
    fn drop(&mut self) {
        if self.ctx != 0 {
            // SAFETY: Context is owned by this driver instance and destroyed once on drop.
            let _ = unsafe { (self.fns.cu_ctx_destroy_v2)(self.ctx_ptr()) };
            self.ctx = 0;
        }
    }
}

pub struct DeviceBuffer {
    driver: Arc<CudaDriver>,
    ptr: CUdeviceptr,
    bytes: usize,
}

impl fmt::Debug for DeviceBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeviceBuffer")
            .field("ptr", &self.ptr)
            .field("bytes", &self.bytes)
            .finish()
    }
}

impl DeviceBuffer {
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn read_to_vec(&self) -> BackendResult<Vec<u8>> {
        self.driver.download(self.ptr, self.bytes)
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        // SAFETY: Device pointer was allocated by this driver and is released once on drop.
        let _ = unsafe { (self.driver.fns.cu_mem_free_v2)(self.ptr) };
    }
}

static CUDA_DRIVER: OnceLock<Result<Arc<CudaDriver>, String>> = OnceLock::new();

pub fn is_available() -> bool {
    driver().is_ok()
}

pub fn driver() -> BackendResult<Arc<CudaDriver>> {
    let init = CUDA_DRIVER.get_or_init(|| match CudaDriver::new() {
        Ok(driver) => Ok(Arc::new(driver)),
        Err(err) => Err(err.to_string()),
    });
    match init {
        Ok(driver) => Ok(Arc::clone(driver)),
        Err(msg) => Err(BackendError::execution(format!(
            "CUDA driver unavailable for Triton backend: {msg}"
        ))),
    }
}

impl CudaDriver {
    fn new() -> BackendResult<Self> {
        let lib = load_cuda_library()?;
        let fns = DriverFns {
            cu_init: load_symbol(&lib, b"cuInit\0")?,
            cu_device_get: load_symbol(&lib, b"cuDeviceGet\0")?,
            cu_ctx_create_v2: load_symbol(&lib, b"cuCtxCreate_v2\0")?,
            cu_ctx_destroy_v2: load_symbol(&lib, b"cuCtxDestroy_v2\0")?,
            cu_ctx_set_current: load_symbol(&lib, b"cuCtxSetCurrent\0")?,
            cu_mem_alloc_v2: load_symbol(&lib, b"cuMemAlloc_v2\0")?,
            cu_mem_free_v2: load_symbol(&lib, b"cuMemFree_v2\0")?,
            cu_memcpy_hto_d_v2: load_symbol(&lib, b"cuMemcpyHtoD_v2\0")?,
            cu_memcpy_dto_h_v2: load_symbol(&lib, b"cuMemcpyDtoH_v2\0")?,
            cu_memset_d8_v2: load_symbol(&lib, b"cuMemsetD8_v2\0")?,
        };

        // SAFETY: Calls are made with valid pointers and follow CUDA driver API contract.
        unsafe {
            check_cuda((fns.cu_init)(0), "cuInit")?;
            let mut dev: CUdevice = 0;
            check_cuda(
                (fns.cu_device_get)(&mut dev as *mut CUdevice, 0),
                "cuDeviceGet",
            )?;
            let mut ctx: CUcontext = std::ptr::null_mut();
            check_cuda(
                (fns.cu_ctx_create_v2)(&mut ctx as *mut CUcontext, 0, dev),
                "cuCtxCreate_v2",
            )?;
            check_cuda((fns.cu_ctx_set_current)(ctx), "cuCtxSetCurrent")?;
            Ok(Self {
                _lib: lib,
                fns,
                ctx: ctx as usize,
            })
        }
    }

    pub fn alloc_and_upload(self: &Arc<Self>, bytes: &[u8]) -> BackendResult<Arc<DeviceBuffer>> {
        let buffer = self.alloc(bytes.len())?;
        if !bytes.is_empty() {
            self.ensure_current()?;
            // SAFETY: Destination is a valid allocated device pointer and source host slice is valid.
            unsafe {
                check_cuda(
                    (self.fns.cu_memcpy_hto_d_v2)(
                        buffer.ptr,
                        bytes.as_ptr() as *const c_void,
                        bytes.len(),
                    ),
                    "cuMemcpyHtoD_v2",
                )?;
            }
        }
        Ok(buffer)
    }

    pub fn alloc_zeroed(self: &Arc<Self>, bytes: usize) -> BackendResult<Arc<DeviceBuffer>> {
        let buffer = self.alloc(bytes)?;
        if bytes != 0 {
            self.ensure_current()?;
            // SAFETY: Destination is a valid allocated device pointer; memset count is bounded by allocation size.
            unsafe {
                check_cuda(
                    (self.fns.cu_memset_d8_v2)(buffer.ptr, 0, bytes),
                    "cuMemsetD8_v2",
                )?;
            }
        }
        Ok(buffer)
    }

    pub fn download(&self, ptr: CUdeviceptr, bytes: usize) -> BackendResult<Vec<u8>> {
        self.ensure_current()?;
        let mut out = vec![0u8; bytes];
        if bytes != 0 {
            // SAFETY: Source device pointer is valid for `bytes`; destination host buffer is valid and writable.
            unsafe {
                check_cuda(
                    (self.fns.cu_memcpy_dto_h_v2)(out.as_mut_ptr() as *mut c_void, ptr, bytes),
                    "cuMemcpyDtoH_v2",
                )?;
            }
        }
        Ok(out)
    }

    fn alloc(self: &Arc<Self>, bytes: usize) -> BackendResult<Arc<DeviceBuffer>> {
        self.ensure_current()?;
        let mut ptr: CUdeviceptr = 0;
        // SAFETY: `ptr` is a valid out pointer for CUDA allocation.
        unsafe {
            check_cuda(
                (self.fns.cu_mem_alloc_v2)(&mut ptr as *mut CUdeviceptr, bytes),
                "cuMemAlloc_v2",
            )?;
        }
        Ok(Arc::new(DeviceBuffer {
            driver: Arc::clone(self),
            ptr,
            bytes,
        }))
    }

    fn ensure_current(&self) -> BackendResult<()> {
        // SAFETY: Context was created by this driver and remains valid until drop.
        unsafe {
            check_cuda(
                (self.fns.cu_ctx_set_current)(self.ctx_ptr()),
                "cuCtxSetCurrent",
            )
        }
    }

    fn ctx_ptr(&self) -> CUcontext {
        self.ctx as CUcontext
    }
}

fn load_cuda_library() -> BackendResult<Library> {
    let candidates = ["libcuda.so.1", "libcuda.so", "nvcuda.dll", "libcuda.dylib"];

    for candidate in candidates {
        // SAFETY: Dynamic library probe only; no symbols are invoked at this stage.
        if let Ok(lib) = unsafe { Library::new(candidate) } {
            return Ok(lib);
        }
    }

    Err(BackendError::execution(
        "failed to load CUDA driver library (tried libcuda.so.1, libcuda.so, nvcuda.dll, libcuda.dylib)",
    ))
}

fn load_symbol<T: Copy>(lib: &Library, name: &'static [u8]) -> BackendResult<T> {
    // SAFETY: Caller provides expected symbol type from CUDA driver API.
    let sym = unsafe { lib.get::<T>(name) }.map_err(|err| {
        BackendError::execution(format!(
            "failed to resolve CUDA symbol {}: {err}",
            String::from_utf8_lossy(name)
        ))
    })?;
    Ok(*sym)
}

fn check_cuda(code: CUresult, op: &str) -> BackendResult<()> {
    if code == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(BackendError::execution(format!(
            "CUDA driver call {op} failed with code {code}"
        )))
    }
}
