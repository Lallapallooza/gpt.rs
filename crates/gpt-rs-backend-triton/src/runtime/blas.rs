use std::ffi::c_void;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use gpt_rs::backend::spec::{BackendError, BackendResult, TensorSpec};
use gpt_rs::profiling::{self, ScopeMeta, WorkStats};
use libloading::Library;

use super::{
    fused_dot_epilogue, CudaDriver, DeviceBuffer, KernelSpec, TritonExecutor, TritonTensor,
    GPU_EVENT_TIMING_ENABLED,
};

impl TritonExecutor {
    pub(super) fn cublas(&self, driver: &Arc<CudaDriver>) -> BackendResult<Arc<CublasContext>> {
        let state = self
            .cublas
            .get_or_init(|| match CublasContext::new(Arc::clone(driver)) {
                Ok(ctx) => Ok(Arc::new(ctx)),
                Err(err) => Err(err.to_string()),
            });

        match state {
            Ok(ctx) => Ok(Arc::clone(ctx)),
            Err(message) => Err(BackendError::execution(format!(
                "cublas runtime unavailable: {message}"
            ))),
        }
    }
}

pub(super) fn cublas_profile_scope(
    transposed_rhs: bool,
    m: usize,
    n: usize,
    k: usize,
) -> profiling::ScopeGuard {
    let signature = format!("sgemm.m{m}.n{n}.k{k}.rhs_t{}", u8::from(transposed_rhs));
    profiling::backend_scope_with_meta("backend.triton.cublas_sgemm", || {
        let meta = profiling::signature_id(&signature)
            .map(ScopeMeta::signature)
            .unwrap_or_default();
        let m_u64 = m as u64;
        let n_u64 = n as u64;
        let k_u64 = k as u64;
        let work = WorkStats {
            elements: m_u64.saturating_mul(n_u64),
            flops: m_u64
                .saturating_mul(n_u64)
                .saturating_mul(k_u64)
                .saturating_mul(2),
            ..WorkStats::default()
        };
        meta.with_work(work)
    })
}

fn cublas_strided_batched_profile_scope(
    transposed_rhs: bool,
    m: usize,
    n: usize,
    k: usize,
    batches: usize,
) -> profiling::ScopeGuard {
    let signature = format!(
        "sgemm_strided_batched.b{batches}.m{m}.n{n}.k{k}.rhs_t{}",
        u8::from(transposed_rhs)
    );
    profiling::backend_scope_with_meta("backend.triton.cublas_sgemm_strided_batched", || {
        let meta = profiling::signature_id(&signature)
            .map(ScopeMeta::signature)
            .unwrap_or_default();
        let m_u64 = m as u64;
        let n_u64 = n as u64;
        let k_u64 = k as u64;
        let b_u64 = batches as u64;
        let work = WorkStats {
            elements: b_u64.saturating_mul(m_u64.saturating_mul(n_u64)),
            flops: b_u64.saturating_mul(
                m_u64
                    .saturating_mul(n_u64)
                    .saturating_mul(k_u64)
                    .saturating_mul(2),
            ),
            ..WorkStats::default()
        };
        meta.with_work(work)
    })
}

pub(super) struct DotGeneralArgs<'a> {
    pub(super) spec: &'a gpt_rs::backend::spec::DotGeneralSpec,
    pub(super) lhs_spec: &'a TensorSpec,
    pub(super) rhs_spec: &'a TensorSpec,
    pub(super) out_spec: &'a TensorSpec,
}

#[derive(Clone)]
pub(super) struct OutputBinding<'a> {
    pub(super) spec: &'a TensorSpec,
    pub(super) tensor: Option<TritonTensor>,
}

impl<'a> OutputBinding<'a> {
    pub(super) fn new(spec: &'a TensorSpec, tensor: Option<TritonTensor>) -> Self {
        Self { spec, tensor }
    }
}

pub(super) struct DotBiasRank2Args<'a> {
    pub(super) driver: &'a Arc<CudaDriver>,
    pub(super) kernel: &'a KernelSpec,
    pub(super) plan: &'a fused_dot_epilogue::FusedDotBiasPlan,
    pub(super) lhs: &'a TritonTensor,
    pub(super) rhs: &'a TritonTensor,
    pub(super) bias: &'a TritonTensor,
    pub(super) out_spec: &'a TensorSpec,
    pub(super) output: Option<TritonTensor>,
}

#[derive(Copy, Clone)]
pub(super) struct StridedBatchedGemmConfig {
    pub(super) m: usize,
    pub(super) n: usize,
    pub(super) k: usize,
    pub(super) lhs_stride: usize,
    pub(super) rhs_stride: usize,
    pub(super) out_stride: usize,
    pub(super) batches: usize,
}

type CublasStatus = i32;
type CublasHandle = *mut c_void;

const CUBLAS_STATUS_SUCCESS: CublasStatus = 0;
const CUBLAS_OP_N: i32 = 0;
const CUBLAS_OP_T: i32 = 1;
const CUBLAS_POINTER_MODE_DEVICE: i32 = 1;

type CublasCreateFn = unsafe extern "C" fn(handle: *mut CublasHandle) -> CublasStatus;
type CublasDestroyFn = unsafe extern "C" fn(handle: CublasHandle) -> CublasStatus;
type CublasSetStreamFn =
    unsafe extern "C" fn(handle: CublasHandle, stream: *mut c_void) -> CublasStatus;
type CublasSetPointerModeFn = unsafe extern "C" fn(handle: CublasHandle, mode: i32) -> CublasStatus;
type CublasSgemmFn = unsafe extern "C" fn(
    handle: CublasHandle,
    transa: i32,
    transb: i32,
    m: i32,
    n: i32,
    k: i32,
    alpha: *const f32,
    a: *const f32,
    lda: i32,
    b: *const f32,
    ldb: i32,
    beta: *const f32,
    c: *mut f32,
    ldc: i32,
) -> CublasStatus;
type CublasSgemmStridedBatchedFn = unsafe extern "C" fn(
    handle: CublasHandle,
    transa: i32,
    transb: i32,
    m: i32,
    n: i32,
    k: i32,
    alpha: *const f32,
    a: *const f32,
    lda: i32,
    stride_a: i64,
    b: *const f32,
    ldb: i32,
    stride_b: i64,
    beta: *const f32,
    c: *mut f32,
    ldc: i32,
    stride_c: i64,
    batch_count: i32,
) -> CublasStatus;

struct CublasFns {
    create: CublasCreateFn,
    destroy: CublasDestroyFn,
    set_stream: CublasSetStreamFn,
    set_pointer_mode: CublasSetPointerModeFn,
    sgemm: CublasSgemmFn,
    sgemm_strided_batched: CublasSgemmStridedBatchedFn,
}

pub(super) struct CublasContext {
    _lib: Library,
    fns: CublasFns,
    handle: usize,
    driver: Arc<CudaDriver>,
    alpha: Arc<DeviceBuffer>,
    beta: Arc<DeviceBuffer>,
}

impl Drop for CublasContext {
    fn drop(&mut self) {
        // SAFETY: Handle is created once and destroyed once on drop.
        let _ = unsafe { (self.fns.destroy)(self.handle as CublasHandle) };
        self.handle = 0;
    }
}

impl CublasContext {
    fn run_cublas_timed<F>(&self, op_name: &str, op: F) -> BackendResult<()>
    where
        F: FnOnce() -> BackendResult<()>,
    {
        if GPU_EVENT_TIMING_ENABLED.load(Ordering::Relaxed) {
            let _elapsed_ms = self.driver.time_with_events(op_name, op)?;
            Ok(())
        } else {
            op()
        }
    }

    pub(super) fn new(driver: Arc<CudaDriver>) -> BackendResult<Self> {
        let lib = load_cublas_library()?;
        let fns = CublasFns {
            create: load_cublas_symbol(&lib, b"cublasCreate_v2\0")?,
            destroy: load_cublas_symbol(&lib, b"cublasDestroy_v2\0")?,
            set_stream: load_cublas_symbol(&lib, b"cublasSetStream_v2\0")?,
            set_pointer_mode: load_cublas_symbol(&lib, b"cublasSetPointerMode_v2\0")?,
            sgemm: load_cublas_symbol(&lib, b"cublasSgemm_v2\0")?,
            sgemm_strided_batched: load_cublas_symbol(&lib, b"cublasSgemmStridedBatched\0")?,
        };

        driver.ensure_current()?;
        let mut handle: CublasHandle = std::ptr::null_mut();
        // SAFETY: cublasCreate_v2 initializes the output handle pointer.
        unsafe {
            check_cublas(
                (fns.create)(&mut handle as *mut CublasHandle),
                "cublasCreate_v2",
            )?;
            check_cublas(
                (fns.set_stream)(handle, driver.stream_handle()),
                "cublasSetStream_v2",
            )?;
            check_cublas(
                (fns.set_pointer_mode)(handle, CUBLAS_POINTER_MODE_DEVICE),
                "cublasSetPointerMode_v2",
            )?;
        }

        let alpha = driver.alloc_and_upload(&1.0f32.to_ne_bytes())?;
        let beta = driver.alloc_and_upload(&0.0f32.to_ne_bytes())?;

        Ok(Self {
            _lib: lib,
            fns,
            handle: handle as usize,
            driver,
            alpha,
            beta,
        })
    }

    pub(super) fn sgemm_row_major(
        &self,
        lhs: &DeviceBuffer,
        rhs: &DeviceBuffer,
        out: &DeviceBuffer,
        m: usize,
        n: usize,
        k: usize,
    ) -> BackendResult<()> {
        self.sgemm_row_major_raw(
            lhs.device_ptr(),
            rhs.device_ptr(),
            out.device_ptr(),
            m,
            n,
            k,
        )
    }

    pub(super) fn sgemm_row_major_strided_batched(
        &self,
        lhs: &DeviceBuffer,
        rhs: &DeviceBuffer,
        out: &DeviceBuffer,
        cfg: StridedBatchedGemmConfig,
    ) -> BackendResult<()> {
        self.sgemm_row_major_strided_batched_raw(
            lhs.device_ptr(),
            rhs.device_ptr(),
            out.device_ptr(),
            cfg,
        )
    }

    pub(super) fn sgemm_row_major_strided_batched_rhs_transposed(
        &self,
        lhs: &DeviceBuffer,
        rhs: &DeviceBuffer,
        out: &DeviceBuffer,
        cfg: StridedBatchedGemmConfig,
    ) -> BackendResult<()> {
        self.sgemm_row_major_strided_batched_raw_rhs_transposed(
            lhs.device_ptr(),
            rhs.device_ptr(),
            out.device_ptr(),
            cfg,
        )
    }

    pub(super) fn sgemm_row_major_raw(
        &self,
        lhs_ptr: u64,
        rhs_ptr: u64,
        out_ptr: u64,
        m: usize,
        n: usize,
        k: usize,
    ) -> BackendResult<()> {
        let _scope = cublas_profile_scope(false, m, n, k);
        let m_i32 = i32::try_from(m)
            .map_err(|_| BackendError::execution("matrix dimension m exceeds i32"))?;
        let n_i32 = i32::try_from(n)
            .map_err(|_| BackendError::execution("matrix dimension n exceeds i32"))?;
        let k_i32 = i32::try_from(k)
            .map_err(|_| BackendError::execution("matrix dimension k exceeds i32"))?;

        self.driver.ensure_current()?;
        let alpha_ptr = self.alpha.device_ptr() as usize as *const f32;
        let beta_ptr = self.beta.device_ptr() as usize as *const f32;
        // Row-major C = A * B using column-major GEMM: C^T = B^T * A^T.
        self.run_cublas_timed("cublasSgemm_v2", || {
            // SAFETY: pointers are valid CUDA device pointers for buffers sized according to m,n,k.
            unsafe {
                check_cublas(
                    (self.fns.sgemm)(
                        self.handle as CublasHandle,
                        CUBLAS_OP_N,
                        CUBLAS_OP_N,
                        n_i32,
                        m_i32,
                        k_i32,
                        alpha_ptr,
                        rhs_ptr as usize as *const f32,
                        n_i32,
                        lhs_ptr as usize as *const f32,
                        k_i32,
                        beta_ptr,
                        out_ptr as usize as *mut f32,
                        n_i32,
                    ),
                    "cublasSgemm_v2",
                )?;
            }
            Ok(())
        })?;
        Ok(())
    }

    fn sgemm_row_major_strided_batched_raw(
        &self,
        lhs_ptr: u64,
        rhs_ptr: u64,
        out_ptr: u64,
        cfg: StridedBatchedGemmConfig,
    ) -> BackendResult<()> {
        let StridedBatchedGemmConfig {
            m,
            n,
            k,
            lhs_stride,
            rhs_stride,
            out_stride,
            batches,
        } = cfg;
        let _scope = cublas_strided_batched_profile_scope(false, m, n, k, batches);
        let m_i32 = i32::try_from(m)
            .map_err(|_| BackendError::execution("matrix dimension m exceeds i32"))?;
        let n_i32 = i32::try_from(n)
            .map_err(|_| BackendError::execution("matrix dimension n exceeds i32"))?;
        let k_i32 = i32::try_from(k)
            .map_err(|_| BackendError::execution("matrix dimension k exceeds i32"))?;
        let batch_i32 = i32::try_from(batches)
            .map_err(|_| BackendError::execution("batch count exceeds i32"))?;
        let lhs_stride_i64 = i64::try_from(lhs_stride)
            .map_err(|_| BackendError::execution("lhs stride exceeds i64"))?;
        let rhs_stride_i64 = i64::try_from(rhs_stride)
            .map_err(|_| BackendError::execution("rhs stride exceeds i64"))?;
        let out_stride_i64 = i64::try_from(out_stride)
            .map_err(|_| BackendError::execution("out stride exceeds i64"))?;

        self.driver.ensure_current()?;
        let alpha_ptr = self.alpha.device_ptr() as usize as *const f32;
        let beta_ptr = self.beta.device_ptr() as usize as *const f32;
        // Row-major C = A * B using column-major GEMM: C^T = B^T * A^T.
        // Batch-strided variant follows the same transform.
        self.run_cublas_timed("cublasSgemmStridedBatched", || {
            // SAFETY: pointers/strides are valid CUDA device arguments sized for the batched GEMM.
            unsafe {
                check_cublas(
                    (self.fns.sgemm_strided_batched)(
                        self.handle as CublasHandle,
                        CUBLAS_OP_N,
                        CUBLAS_OP_N,
                        n_i32,
                        m_i32,
                        k_i32,
                        alpha_ptr,
                        rhs_ptr as usize as *const f32,
                        n_i32,
                        rhs_stride_i64,
                        lhs_ptr as usize as *const f32,
                        k_i32,
                        lhs_stride_i64,
                        beta_ptr,
                        out_ptr as usize as *mut f32,
                        n_i32,
                        out_stride_i64,
                        batch_i32,
                    ),
                    "cublasSgemmStridedBatched",
                )?;
            }
            Ok(())
        })?;
        Ok(())
    }

    fn sgemm_row_major_strided_batched_raw_rhs_transposed(
        &self,
        lhs_ptr: u64,
        rhs_ptr: u64,
        out_ptr: u64,
        cfg: StridedBatchedGemmConfig,
    ) -> BackendResult<()> {
        let StridedBatchedGemmConfig {
            m,
            n,
            k,
            lhs_stride,
            rhs_stride,
            out_stride,
            batches,
        } = cfg;
        let _scope = cublas_strided_batched_profile_scope(true, m, n, k, batches);
        let m_i32 = i32::try_from(m)
            .map_err(|_| BackendError::execution("matrix dimension m exceeds i32"))?;
        let n_i32 = i32::try_from(n)
            .map_err(|_| BackendError::execution("matrix dimension n exceeds i32"))?;
        let k_i32 = i32::try_from(k)
            .map_err(|_| BackendError::execution("matrix dimension k exceeds i32"))?;
        let batch_i32 = i32::try_from(batches)
            .map_err(|_| BackendError::execution("batch count exceeds i32"))?;
        let lhs_stride_i64 = i64::try_from(lhs_stride)
            .map_err(|_| BackendError::execution("lhs stride exceeds i64"))?;
        let rhs_stride_i64 = i64::try_from(rhs_stride)
            .map_err(|_| BackendError::execution("rhs stride exceeds i64"))?;
        let out_stride_i64 = i64::try_from(out_stride)
            .map_err(|_| BackendError::execution("out stride exceeds i64"))?;

        self.driver.ensure_current()?;
        let alpha_ptr = self.alpha.device_ptr() as usize as *const f32;
        let beta_ptr = self.beta.device_ptr() as usize as *const f32;
        // Row-major C = A(MxK) * B^T where rhs pointer stores row-major B(NxK).
        // Column-major equivalent per batch:
        // C^T(NxM) = B(NxK) * A^T(KxM), with cuBLAS transa=Transpose for rhs operand.
        self.run_cublas_timed("cublasSgemmStridedBatched", || {
            // SAFETY: pointers/strides are valid CUDA device arguments sized for the batched GEMM.
            unsafe {
                check_cublas(
                    (self.fns.sgemm_strided_batched)(
                        self.handle as CublasHandle,
                        CUBLAS_OP_T,
                        CUBLAS_OP_N,
                        n_i32,
                        m_i32,
                        k_i32,
                        alpha_ptr,
                        rhs_ptr as usize as *const f32,
                        k_i32,
                        rhs_stride_i64,
                        lhs_ptr as usize as *const f32,
                        k_i32,
                        lhs_stride_i64,
                        beta_ptr,
                        out_ptr as usize as *mut f32,
                        n_i32,
                        out_stride_i64,
                        batch_i32,
                    ),
                    "cublasSgemmStridedBatched",
                )?;
            }
            Ok(())
        })?;
        Ok(())
    }
}

fn load_cublas_library() -> BackendResult<Library> {
    let candidates = [
        "libcublas.so.12",
        "libcublas.so",
        "cublas64_12.dll",
        "cublas64_11.dll",
    ];

    for candidate in candidates {
        // SAFETY: dynamic library probing only.
        if let Ok(lib) = unsafe { Library::new(candidate) } {
            return Ok(lib);
        }
    }

    Err(BackendError::execution(
        "failed to load cuBLAS library (tried libcublas.so.12, libcublas.so, cublas64_12.dll, cublas64_11.dll)",
    ))
}

fn load_cublas_symbol<T: Copy>(lib: &Library, name: &'static [u8]) -> BackendResult<T> {
    // SAFETY: symbol type is expected to match the cuBLAS API.
    let symbol = unsafe { lib.get::<T>(name) }.map_err(|err| {
        BackendError::execution(format!(
            "failed to resolve cuBLAS symbol {}: {err}",
            String::from_utf8_lossy(name)
        ))
    })?;
    Ok(*symbol)
}

fn check_cublas(status: CublasStatus, call: &str) -> BackendResult<()> {
    if status == CUBLAS_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(BackendError::execution(format!(
            "cuBLAS call {call} failed with status {status}"
        )))
    }
}
