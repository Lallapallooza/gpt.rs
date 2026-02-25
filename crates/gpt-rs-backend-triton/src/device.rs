use std::ffi::{c_void, CString};
use std::fmt;
use std::sync::{Arc, Mutex, OnceLock};

use gpt_rs::backend::spec::{BackendError, BackendResult};
use libloading::Library;

type CUresult = i32;
type CUdevice = i32;
type CUcontext = *mut c_void;
type CUdeviceptr = u64;
type CUmodule = *mut c_void;
type CUfunction = *mut c_void;
type CUstream = *mut c_void;
type CUevent = *mut c_void;

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
type CuMemcpyDtoDV2Fn = unsafe extern "C" fn(
    dst_device: CUdeviceptr,
    src_device: CUdeviceptr,
    byte_count: usize,
) -> CUresult;
type CuMemsetD8V2Fn =
    unsafe extern "C" fn(dst_device: CUdeviceptr, value: u8, count: usize) -> CUresult;
type CuModuleLoadDataExFn = unsafe extern "C" fn(
    module: *mut CUmodule,
    image: *const c_void,
    num_options: u32,
    options: *mut u32,
    option_values: *mut *mut c_void,
) -> CUresult;
type CuModuleUnloadFn = unsafe extern "C" fn(module: CUmodule) -> CUresult;
type CuModuleGetFunctionFn =
    unsafe extern "C" fn(hfunc: *mut CUfunction, hmod: CUmodule, name: *const i8) -> CUresult;
type CuLaunchKernelFn = unsafe extern "C" fn(
    f: CUfunction,
    grid_dim_x: u32,
    grid_dim_y: u32,
    grid_dim_z: u32,
    block_dim_x: u32,
    block_dim_y: u32,
    block_dim_z: u32,
    shared_mem_bytes: u32,
    h_stream: CUstream,
    kernel_params: *mut *mut c_void,
    extra: *mut *mut c_void,
) -> CUresult;
type CuEventCreateFn = unsafe extern "C" fn(event: *mut CUevent, flags: u32) -> CUresult;
type CuEventDestroyV2Fn = unsafe extern "C" fn(event: CUevent) -> CUresult;
type CuEventRecordFn = unsafe extern "C" fn(event: CUevent, stream: CUstream) -> CUresult;
type CuEventSynchronizeFn = unsafe extern "C" fn(event: CUevent) -> CUresult;
type CuEventElapsedTimeFn =
    unsafe extern "C" fn(ms: *mut f32, start: CUevent, end: CUevent) -> CUresult;

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
    cu_memcpy_dto_d_v2: CuMemcpyDtoDV2Fn,
    cu_memset_d8_v2: CuMemsetD8V2Fn,
    cu_module_load_data_ex: CuModuleLoadDataExFn,
    cu_module_unload: CuModuleUnloadFn,
    cu_module_get_function: CuModuleGetFunctionFn,
    cu_launch_kernel: CuLaunchKernelFn,
    cu_event_create: CuEventCreateFn,
    cu_event_destroy_v2: CuEventDestroyV2Fn,
    cu_event_record: CuEventRecordFn,
    cu_event_synchronize: CuEventSynchronizeFn,
    cu_event_elapsed_time: CuEventElapsedTimeFn,
}

pub struct CudaDriver {
    _lib: Library,
    fns: DriverFns,
    // Stored as usize so CudaDriver can satisfy Send/Sync requirements for backend traits.
    ctx: usize,
    alloc_cache: Mutex<AllocCache>,
}

impl Drop for CudaDriver {
    fn drop(&mut self) {
        self.release_cached_allocations();
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

    pub fn device_ptr(&self) -> u64 {
        self.ptr
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        self.driver.recycle(self.ptr, self.bytes);
    }
}

#[derive(Clone)]
pub struct CudaFunction {
    #[allow(dead_code)]
    module: Arc<CudaModule>,
    func: usize,
}

pub struct CudaModule {
    driver: Arc<CudaDriver>,
    module: usize,
}

impl Drop for CudaModule {
    fn drop(&mut self) {
        if self.module != 0 {
            // SAFETY: Module belongs to this driver and is unloaded once.
            let _ = unsafe { (self.driver.fns.cu_module_unload)(self.module_ptr()) };
            self.module = 0;
        }
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
            cu_memcpy_dto_d_v2: load_symbol(&lib, b"cuMemcpyDtoD_v2\0")?,
            cu_memset_d8_v2: load_symbol(&lib, b"cuMemsetD8_v2\0")?,
            cu_module_load_data_ex: load_symbol(&lib, b"cuModuleLoadDataEx\0")?,
            cu_module_unload: load_symbol(&lib, b"cuModuleUnload\0")?,
            cu_module_get_function: load_symbol(&lib, b"cuModuleGetFunction\0")?,
            cu_launch_kernel: load_symbol(&lib, b"cuLaunchKernel\0")?,
            cu_event_create: load_symbol(&lib, b"cuEventCreate\0")?,
            cu_event_destroy_v2: load_symbol_any(
                &lib,
                &[b"cuEventDestroy_v2\0", b"cuEventDestroy\0"],
            )?,
            cu_event_record: load_symbol(&lib, b"cuEventRecord\0")?,
            cu_event_synchronize: load_symbol(&lib, b"cuEventSynchronize\0")?,
            cu_event_elapsed_time: load_symbol(&lib, b"cuEventElapsedTime\0")?,
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
                alloc_cache: Mutex::new(AllocCache::default()),
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

    pub fn upload_to_device(&self, dst: CUdeviceptr, bytes: &[u8]) -> BackendResult<()> {
        if bytes.is_empty() {
            return Ok(());
        }
        self.ensure_current()?;
        // SAFETY: Destination device pointer is valid and caller provides exact source buffer.
        unsafe {
            check_cuda(
                (self.fns.cu_memcpy_hto_d_v2)(dst, bytes.as_ptr() as *const c_void, bytes.len()),
                "cuMemcpyHtoD_v2",
            )?;
        }
        Ok(())
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

    pub fn copy_device_to_device(
        &self,
        dst: CUdeviceptr,
        src: CUdeviceptr,
        bytes: usize,
    ) -> BackendResult<()> {
        if bytes == 0 {
            return Ok(());
        }
        self.ensure_current()?;
        // SAFETY: src/dst pointers are valid CUDA allocations and byte range is provided by caller.
        unsafe {
            check_cuda(
                (self.fns.cu_memcpy_dto_d_v2)(dst, src, bytes),
                "cuMemcpyDtoD_v2",
            )?;
        }
        Ok(())
    }

    fn alloc(self: &Arc<Self>, bytes: usize) -> BackendResult<Arc<DeviceBuffer>> {
        if bytes != 0 {
            let cached_ptr = {
                let mut cache = self
                    .alloc_cache
                    .lock()
                    .expect("triton allocator cache mutex poisoned");
                cache.take(bytes)
            };
            if let Some(ptr) = cached_ptr {
                return Ok(Arc::new(DeviceBuffer {
                    driver: Arc::clone(self),
                    ptr,
                    bytes,
                }));
            }
        }

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

    fn recycle(&self, ptr: CUdeviceptr, bytes: usize) {
        if ptr == 0 {
            return;
        }
        if bytes == 0 {
            let _ = self.free_ptr(ptr);
            return;
        }
        let should_free = {
            let mut cache = self
                .alloc_cache
                .lock()
                .expect("triton allocator cache mutex poisoned");
            !cache.put(ptr, bytes)
        };
        if should_free {
            let _ = self.free_ptr(ptr);
        }
    }

    fn free_ptr(&self, ptr: CUdeviceptr) -> BackendResult<()> {
        self.ensure_current()?;
        // SAFETY: `ptr` is a CUDA allocation associated with this context.
        unsafe { check_cuda((self.fns.cu_mem_free_v2)(ptr), "cuMemFree_v2") }
    }

    fn release_cached_allocations(&mut self) {
        let pointers = {
            let mut cache = self
                .alloc_cache
                .lock()
                .expect("triton allocator cache mutex poisoned");
            cache.drain()
        };
        if pointers.is_empty() {
            return;
        }
        if self.ctx != 0 {
            // SAFETY: Context is owned by this driver and valid until drop returns.
            let _ = unsafe { (self.fns.cu_ctx_set_current)(self.ctx_ptr()) };
        }
        for ptr in pointers {
            // SAFETY: pointers were cached from successful allocations and are released once.
            let _ = unsafe { (self.fns.cu_mem_free_v2)(ptr) };
        }
    }

    pub fn load_ptx_module(self: &Arc<Self>, ptx: &str) -> BackendResult<Arc<CudaModule>> {
        self.ensure_current()?;
        let c_ptx = CString::new(ptx)
            .map_err(|_| BackendError::execution("ptx source contains NUL byte"))?;
        let mut module: CUmodule = std::ptr::null_mut();
        // SAFETY: pointer arguments are valid for cuModuleLoadDataEx.
        unsafe {
            check_cuda(
                (self.fns.cu_module_load_data_ex)(
                    &mut module as *mut CUmodule,
                    c_ptx.as_ptr() as *const c_void,
                    0,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                ),
                "cuModuleLoadDataEx",
            )?;
        }
        Ok(Arc::new(CudaModule {
            driver: Arc::clone(self),
            module: module as usize,
        }))
    }

    pub fn get_function(
        &self,
        module: &Arc<CudaModule>,
        symbol: &str,
    ) -> BackendResult<CudaFunction> {
        self.ensure_current()?;
        let c_symbol = CString::new(symbol)
            .map_err(|_| BackendError::execution("kernel symbol contains NUL byte"))?;
        let mut function: CUfunction = std::ptr::null_mut();
        // SAFETY: module and output pointers are valid.
        unsafe {
            check_cuda(
                (self.fns.cu_module_get_function)(
                    &mut function as *mut CUfunction,
                    module.module_ptr(),
                    c_symbol.as_ptr(),
                ),
                "cuModuleGetFunction",
            )?;
        }

        Ok(CudaFunction {
            module: Arc::clone(module),
            func: function as usize,
        })
    }

    pub fn launch_kernel(
        &self,
        function: &CudaFunction,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem_bytes: u32,
        params: &mut [*mut c_void],
    ) -> BackendResult<()> {
        self.ensure_current()?;
        // SAFETY: function and parameter pointers are valid for kernel launch.
        unsafe {
            check_cuda(
                (self.fns.cu_launch_kernel)(
                    function.func_ptr(),
                    grid.0,
                    grid.1,
                    grid.2,
                    block.0,
                    block.1,
                    block.2,
                    shared_mem_bytes,
                    std::ptr::null_mut(),
                    params.as_mut_ptr(),
                    std::ptr::null_mut(),
                ),
                "cuLaunchKernel",
            )?;
        }
        Ok(())
    }

    pub fn time_with_events<F>(&self, op_name: &str, op: F) -> BackendResult<f32>
    where
        F: FnOnce() -> BackendResult<()>,
    {
        const CU_EVENT_DEFAULT: u32 = 0;

        self.ensure_current()?;
        let mut start: CUevent = std::ptr::null_mut();
        let mut end: CUevent = std::ptr::null_mut();

        // SAFETY: start/end are valid out-pointers for CUDA event creation.
        unsafe {
            check_cuda(
                (self.fns.cu_event_create)(&mut start as *mut CUevent, CU_EVENT_DEFAULT),
                "cuEventCreate(start)",
            )?;
            check_cuda(
                (self.fns.cu_event_create)(&mut end as *mut CUevent, CU_EVENT_DEFAULT),
                "cuEventCreate(end)",
            )?;
        }

        let start_guard = CudaEventGuard {
            driver: self,
            event: start,
        };
        let end_guard = CudaEventGuard {
            driver: self,
            event: end,
        };

        // SAFETY: Events are valid handles, and null stream means default stream.
        unsafe {
            check_cuda(
                (self.fns.cu_event_record)(start_guard.event, std::ptr::null_mut()),
                "cuEventRecord(start)",
            )?;
        }

        op()?;

        // SAFETY: Event record/sync operations use valid event handles.
        unsafe {
            check_cuda(
                (self.fns.cu_event_record)(end_guard.event, std::ptr::null_mut()),
                "cuEventRecord(end)",
            )?;
            check_cuda(
                (self.fns.cu_event_synchronize)(end_guard.event),
                "cuEventSynchronize(end)",
            )?;
        }

        let mut elapsed_ms = 0.0f32;
        // SAFETY: elapsed output pointer and event handles are valid.
        unsafe {
            check_cuda(
                (self.fns.cu_event_elapsed_time)(
                    &mut elapsed_ms as *mut f32,
                    start_guard.event,
                    end_guard.event,
                ),
                "cuEventElapsedTime",
            )?;
        }
        if !elapsed_ms.is_finite() || elapsed_ms < 0.0 {
            return Err(BackendError::execution(format!(
                "invalid CUDA event elapsed time for {op_name}: {elapsed_ms}"
            )));
        }

        Ok(elapsed_ms)
    }

    pub fn ensure_current(&self) -> BackendResult<()> {
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

struct CudaEventGuard<'a> {
    driver: &'a CudaDriver,
    event: CUevent,
}

impl Drop for CudaEventGuard<'_> {
    fn drop(&mut self) {
        if self.event.is_null() {
            return;
        }
        // SAFETY: Event was created by this driver and is destroyed once here.
        let _ = unsafe { (self.driver.fns.cu_event_destroy_v2)(self.event) };
        self.event = std::ptr::null_mut();
    }
}

impl CudaFunction {
    fn func_ptr(&self) -> CUfunction {
        self.func as CUfunction
    }
}

impl CudaModule {
    fn module_ptr(&self) -> CUmodule {
        self.module as CUmodule
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

fn load_symbol_any<T: Copy>(lib: &Library, names: &[&'static [u8]]) -> BackendResult<T> {
    for name in names {
        // SAFETY: Caller provides expected symbol type from CUDA driver API.
        if let Ok(sym) = unsafe { lib.get::<T>(name) } {
            return Ok(*sym);
        }
    }

    let tried = names
        .iter()
        .map(|name| {
            String::from_utf8_lossy(name)
                .trim_end_matches('\0')
                .to_string()
        })
        .collect::<Vec<_>>()
        .join(", ");
    Err(BackendError::execution(format!(
        "failed to resolve any CUDA symbol variant: {tried}"
    )))
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

#[derive(Default)]
struct AllocCache {
    bins: std::collections::HashMap<usize, Vec<CUdeviceptr>>,
    cached_bytes: usize,
}

impl AllocCache {
    const MAX_CACHED_BYTES: usize = 512 * 1024 * 1024;

    fn take(&mut self, bytes: usize) -> Option<CUdeviceptr> {
        let ptr = self.bins.get_mut(&bytes).and_then(Vec::pop);
        if ptr.is_some() {
            self.cached_bytes = self.cached_bytes.saturating_sub(bytes);
            if self.bins.get(&bytes).is_some_and(Vec::is_empty) {
                self.bins.remove(&bytes);
            }
        }
        ptr
    }

    fn put(&mut self, ptr: CUdeviceptr, bytes: usize) -> bool {
        if self
            .cached_bytes
            .checked_add(bytes)
            .is_none_or(|sum| sum > Self::MAX_CACHED_BYTES)
        {
            return false;
        }
        self.cached_bytes += bytes;
        self.bins.entry(bytes).or_default().push(ptr);
        true
    }

    fn drain(&mut self) -> Vec<CUdeviceptr> {
        self.cached_bytes = 0;
        let mut out = Vec::new();
        for (_, mut pointers) in std::mem::take(&mut self.bins) {
            out.append(&mut pointers);
        }
        out
    }
}
