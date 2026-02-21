use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::{Arc, Mutex, OnceLock};

use gpt_rs::backend::spec::{
    BackendError, BackendResult, DType, Dimension, ElementwiseBinaryOp, ReduceKind,
};
use libloading::Library;

use crate::bundle::{BundleStep, KernelKind, KernelSpec, SerializableLiteral, TritonBundle};
use crate::compiler::KernelCompiler;
use crate::device::{self, CudaDriver, CudaFunction, DeviceBuffer};
use crate::tensor::TritonTensor;

pub struct TritonExecutor {
    compiler: KernelCompiler,
    loaded_kernels: Mutex<HashMap<u64, Arc<LoadedKernel>>>,
    cublas: OnceLock<Result<Arc<CublasContext>, String>>,
    reduce_ones_cache: Mutex<HashMap<usize, Arc<DeviceBuffer>>>,
}

impl TritonExecutor {
    pub fn new() -> Self {
        Self {
            compiler: KernelCompiler::new(),
            loaded_kernels: Mutex::new(HashMap::new()),
            cublas: OnceLock::new(),
            reduce_ones_cache: Mutex::new(HashMap::new()),
        }
    }

    pub fn execute_bundle(
        &self,
        bundle: &TritonBundle,
        entry_inputs: &[TritonTensor],
    ) -> BackendResult<Vec<TritonTensor>> {
        if bundle.parameter_ids.len() != entry_inputs.len() {
            return Err(BackendError::execution(format!(
                "triton entry input arity mismatch: expected {}, got {}",
                bundle.parameter_ids.len(),
                entry_inputs.len()
            )));
        }

        let driver = device::driver()?;
        let mut values: HashMap<u32, TritonTensor> = HashMap::new();
        for (value_id, input) in bundle
            .parameter_ids
            .iter()
            .copied()
            .zip(entry_inputs.iter())
        {
            values.insert(value_id, input.clone());
        }

        let kernel_specs = bundle
            .kernels
            .iter()
            .map(|spec| (spec.id.as_str(), spec))
            .collect::<HashMap<_, _>>();

        for step in &bundle.steps {
            match step {
                BundleStep::Constant { value_id, literal } => {
                    let tensor = literal_to_tensor(&driver, literal)?;
                    values.insert(*value_id, tensor);
                }
                BundleStep::Alias {
                    value_id,
                    source_id,
                    spec,
                } => {
                    let source = values.get(source_id).cloned().ok_or_else(|| {
                        BackendError::execution(format!(
                            "alias source value {} missing from runtime state",
                            source_id
                        ))
                    })?;
                    values.insert(
                        *value_id,
                        TritonTensor {
                            spec: spec.clone(),
                            buffer: source.buffer,
                        },
                    );
                }
                BundleStep::ElementwiseBinary {
                    value_id,
                    lhs_id,
                    rhs_id,
                    op,
                    spec,
                    kernel_id,
                } => {
                    let lhs = values.get(lhs_id).cloned().ok_or_else(|| {
                        BackendError::execution(format!(
                            "lhs value {} missing for elementwise step",
                            lhs_id
                        ))
                    })?;
                    let rhs = values.get(rhs_id).cloned().ok_or_else(|| {
                        BackendError::execution(format!(
                            "rhs value {} missing for elementwise step",
                            rhs_id
                        ))
                    })?;
                    let kernel = kernel_specs.get(kernel_id.as_str()).ok_or_else(|| {
                        BackendError::execution(format!(
                            "kernel '{}' missing from bundle",
                            kernel_id
                        ))
                    })?;
                    let out =
                        self.execute_elementwise_binary(&driver, kernel, *op, &lhs, &rhs, spec)?;
                    values.insert(*value_id, out);
                }
                BundleStep::DotGeneral {
                    value_id,
                    lhs_id,
                    rhs_id,
                    lhs_spec,
                    rhs_spec,
                    out_spec,
                    spec,
                } => {
                    let lhs = values.get(lhs_id).cloned().ok_or_else(|| {
                        BackendError::execution(format!(
                            "lhs value {} missing for dot_general step",
                            lhs_id
                        ))
                    })?;
                    let rhs = values.get(rhs_id).cloned().ok_or_else(|| {
                        BackendError::execution(format!(
                            "rhs value {} missing for dot_general step",
                            rhs_id
                        ))
                    })?;
                    let args = DotGeneralArgs {
                        spec,
                        lhs_spec,
                        rhs_spec,
                        out_spec,
                    };
                    let out = self.execute_dot_general(&driver, args, &lhs, &rhs)?;
                    values.insert(*value_id, out);
                }
                BundleStep::Reduce {
                    value_id,
                    input_id,
                    input_spec,
                    out_spec,
                    spec,
                } => {
                    let input = values.get(input_id).cloned().ok_or_else(|| {
                        BackendError::execution(format!(
                            "input value {} missing for reduce step",
                            input_id
                        ))
                    })?;
                    let out = self.execute_reduce_sum_last_axis(
                        &driver, input_spec, out_spec, spec, &input,
                    )?;
                    values.insert(*value_id, out);
                }
            }
        }

        let mut results = Vec::with_capacity(bundle.result_ids.len());
        for result_id in &bundle.result_ids {
            let value = values.get(result_id).cloned().ok_or_else(|| {
                BackendError::execution(format!(
                    "bundle result value {} missing from runtime state",
                    result_id
                ))
            })?;
            results.push(value);
        }
        Ok(results)
    }

    fn execute_elementwise_binary(
        &self,
        driver: &Arc<CudaDriver>,
        kernel: &KernelSpec,
        op: ElementwiseBinaryOp,
        lhs: &TritonTensor,
        rhs: &TritonTensor,
        spec: &gpt_rs::backend::spec::TensorSpec,
    ) -> BackendResult<TritonTensor> {
        if lhs.spec != *spec || rhs.spec != *spec {
            return Err(BackendError::execution(
                "elementwise operands/spec mismatch in triton runtime",
            ));
        }
        if spec.dtype != DType::F32 {
            return Err(BackendError::execution(format!(
                "elementwise runtime currently supports F32 only, got {:?}",
                spec.dtype
            )));
        }
        if !matches!(kernel.kind, KernelKind::ElementwiseBinaryF32) {
            return Err(BackendError::execution(format!(
                "unexpected kernel kind for elementwise binary: {:?}",
                kernel.kind
            )));
        }

        let element_count = static_element_count(&spec.shape)?;
        let out = TritonTensor::new(spec.clone(), driver.alloc_zeroed(byte_len(spec)?)?);

        let loaded = self.load_kernel(driver, kernel)?;
        let opcode = binary_opcode(op);
        let count_u32 = u32::try_from(element_count).map_err(|_| {
            BackendError::execution("elementwise tensor too large for u32 launch size")
        })?;
        let block_x = 256u32;
        let grid_x = count_u32.div_ceil(block_x);

        let mut lhs_ptr = lhs.buffer.device_ptr();
        let mut rhs_ptr = rhs.buffer.device_ptr();
        let mut out_ptr = out.buffer.device_ptr();
        let mut n = count_u32;
        let mut op_u32 = opcode;
        let mut params = [
            (&mut lhs_ptr as *mut u64).cast::<c_void>(),
            (&mut rhs_ptr as *mut u64).cast::<c_void>(),
            (&mut out_ptr as *mut u64).cast::<c_void>(),
            (&mut n as *mut u32).cast::<c_void>(),
            (&mut op_u32 as *mut u32).cast::<c_void>(),
        ];
        driver.launch_kernel(
            &loaded.function,
            (grid_x, 1, 1),
            (block_x, 1, 1),
            0,
            &mut params,
        )?;

        Ok(out)
    }

    fn execute_dot_general(
        &self,
        driver: &Arc<CudaDriver>,
        args: DotGeneralArgs<'_>,
        lhs: &TritonTensor,
        rhs: &TritonTensor,
    ) -> BackendResult<TritonTensor> {
        let DotGeneralArgs {
            spec,
            lhs_spec,
            rhs_spec,
            out_spec,
        } = args;
        if lhs.spec != *lhs_spec || rhs.spec != *rhs_spec {
            return Err(BackendError::execution("dot_general tensor/spec mismatch"));
        }
        if lhs_spec.dtype != DType::F32
            || rhs_spec.dtype != DType::F32
            || out_spec.dtype != DType::F32
        {
            return Err(BackendError::execution(
                "dot_general runtime currently supports F32 only",
            ));
        }
        if !spec.batch_lhs.is_empty()
            || !spec.batch_rhs.is_empty()
            || spec.contract_lhs.as_slice() != [1]
            || spec.contract_rhs.as_slice() != [0]
        {
            return Err(BackendError::execution(
                "dot_general runtime supports rank-2 MxK · KxN only",
            ));
        }

        let lhs_dims = static_dims(&lhs_spec.shape)?;
        let rhs_dims = static_dims(&rhs_spec.shape)?;
        let out_dims = static_dims(&out_spec.shape)?;
        if lhs_dims.len() != 2 || rhs_dims.len() != 2 || out_dims.len() != 2 {
            return Err(BackendError::execution(
                "dot_general runtime supports rank-2 tensors only",
            ));
        }

        let m = lhs_dims[0];
        let k = lhs_dims[1];
        let k_rhs = rhs_dims[0];
        let n = rhs_dims[1];
        if k != k_rhs || out_dims[0] != m || out_dims[1] != n {
            return Err(BackendError::execution(
                "dot_general shape mismatch for matrix multiplication",
            ));
        }

        let out = TritonTensor::new(out_spec.clone(), driver.alloc_zeroed(byte_len(out_spec)?)?);
        let cublas = self.cublas(driver)?;
        cublas.sgemm_row_major(&lhs.buffer, &rhs.buffer, &out.buffer, m, n, k)?;
        Ok(out)
    }

    fn execute_reduce_sum_last_axis(
        &self,
        driver: &Arc<CudaDriver>,
        input_spec: &gpt_rs::backend::spec::TensorSpec,
        out_spec: &gpt_rs::backend::spec::TensorSpec,
        spec: &gpt_rs::backend::spec::ReduceSpec,
        input: &TritonTensor,
    ) -> BackendResult<TritonTensor> {
        if input.spec != *input_spec {
            return Err(BackendError::execution("reduce tensor/spec mismatch"));
        }
        if input_spec.dtype != DType::F32 || out_spec.dtype != DType::F32 {
            return Err(BackendError::execution(
                "reduce runtime currently supports F32 only",
            ));
        }
        if spec.kind != ReduceKind::Sum {
            return Err(BackendError::execution(
                "reduce runtime currently supports sum only",
            ));
        }

        let input_dims = static_dims(&input_spec.shape)?;
        if input_dims.is_empty() {
            return Err(BackendError::execution(
                "reduce runtime does not support scalar inputs",
            ));
        }
        let last_axis = input_dims.len() - 1;
        if spec.axes.as_slice() != [last_axis] {
            return Err(BackendError::execution(
                "reduce runtime supports reducing the last axis only",
            ));
        }

        let rows = input_dims[..last_axis]
            .iter()
            .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
            .ok_or_else(|| BackendError::execution("reduce row dimension overflow"))?;
        let cols = input_dims[last_axis];

        let out = TritonTensor::new(out_spec.clone(), driver.alloc_zeroed(byte_len(out_spec)?)?);
        let ones = self.reduce_ones(driver, cols)?;
        let cublas = self.cublas(driver)?;
        cublas.sgemm_row_major(&input.buffer, &ones, &out.buffer, rows, 1, cols)?;
        Ok(out)
    }

    fn load_kernel(
        &self,
        driver: &Arc<CudaDriver>,
        spec: &KernelSpec,
    ) -> BackendResult<Arc<LoadedKernel>> {
        let compiled = self.compiler.compile(spec)?;
        if let Some(found) = self
            .loaded_kernels
            .lock()
            .expect("triton loaded kernel cache poisoned")
            .get(&compiled.fingerprint)
            .cloned()
        {
            return Ok(found);
        }

        let module = driver.load_ptx_module(compiled.ptx.as_ref())?;
        let function = driver.get_function(&module, compiled.symbol.as_ref())?;
        let loaded = Arc::new(LoadedKernel {
            fingerprint: compiled.fingerprint,
            function,
        });
        self.loaded_kernels
            .lock()
            .expect("triton loaded kernel cache poisoned")
            .insert(compiled.fingerprint, Arc::clone(&loaded));
        Ok(loaded)
    }

    fn cublas(&self, driver: &Arc<CudaDriver>) -> BackendResult<Arc<CublasContext>> {
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

    fn reduce_ones(
        &self,
        driver: &Arc<CudaDriver>,
        cols: usize,
    ) -> BackendResult<Arc<DeviceBuffer>> {
        if let Some(found) = self
            .reduce_ones_cache
            .lock()
            .expect("reduce ones cache poisoned")
            .get(&cols)
            .cloned()
        {
            return Ok(found);
        }

        let ones = vec![1.0f32; cols];
        let bytes = f32_to_bytes(&ones);
        let buffer = driver.alloc_and_upload(&bytes)?;
        self.reduce_ones_cache
            .lock()
            .expect("reduce ones cache poisoned")
            .insert(cols, Arc::clone(&buffer));
        Ok(buffer)
    }
}

struct LoadedKernel {
    #[allow(dead_code)]
    fingerprint: u64,
    function: CudaFunction,
}

fn literal_to_tensor(
    driver: &Arc<CudaDriver>,
    literal: &SerializableLiteral,
) -> BackendResult<TritonTensor> {
    let expected = byte_len(&literal.spec)?;
    if expected != literal.bytes.len() {
        return Err(BackendError::execution(format!(
            "literal byte length mismatch for dtype {:?}: expected {}, got {}",
            literal.spec.dtype,
            expected,
            literal.bytes.len()
        )));
    }

    let buffer = driver.alloc_and_upload(&literal.bytes)?;
    Ok(TritonTensor::new(literal.spec.clone(), buffer))
}

fn byte_len(spec: &gpt_rs::backend::spec::TensorSpec) -> BackendResult<usize> {
    spec.byte_len().ok_or_else(|| {
        BackendError::execution(format!(
            "cannot compute byte length for dtype {:?} and shape {:?}",
            spec.dtype,
            spec.shape.dims()
        ))
    })
}

fn static_dims(shape: &gpt_rs::backend::spec::Shape) -> BackendResult<Vec<usize>> {
    let mut dims = Vec::with_capacity(shape.rank());
    for dim in shape.dims() {
        match dim {
            Dimension::Static(value) => dims.push(*value),
            Dimension::Dynamic(_) => {
                return Err(BackendError::execution(
                    "dynamic dimensions are not supported by triton runtime",
                ))
            }
        }
    }
    Ok(dims)
}

fn static_element_count(shape: &gpt_rs::backend::spec::Shape) -> BackendResult<usize> {
    let dims = static_dims(shape)?;
    let mut count = 1usize;
    for dim in dims {
        count = count
            .checked_mul(dim)
            .ok_or_else(|| BackendError::execution("element count overflow"))?;
    }
    Ok(count)
}

fn binary_opcode(op: ElementwiseBinaryOp) -> u32 {
    match op {
        ElementwiseBinaryOp::Add => 0,
        ElementwiseBinaryOp::Sub => 1,
        ElementwiseBinaryOp::Mul => 2,
        ElementwiseBinaryOp::Div => 3,
        ElementwiseBinaryOp::Maximum => 4,
        ElementwiseBinaryOp::Minimum => 5,
    }
}

fn f32_to_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

struct DotGeneralArgs<'a> {
    spec: &'a gpt_rs::backend::spec::DotGeneralSpec,
    lhs_spec: &'a gpt_rs::backend::spec::TensorSpec,
    rhs_spec: &'a gpt_rs::backend::spec::TensorSpec,
    out_spec: &'a gpt_rs::backend::spec::TensorSpec,
}

type CublasStatus = i32;
type CublasHandle = *mut c_void;

const CUBLAS_STATUS_SUCCESS: CublasStatus = 0;
const CUBLAS_OP_N: i32 = 0;

type CublasCreateFn = unsafe extern "C" fn(handle: *mut CublasHandle) -> CublasStatus;
type CublasDestroyFn = unsafe extern "C" fn(handle: CublasHandle) -> CublasStatus;
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

struct CublasFns {
    create: CublasCreateFn,
    destroy: CublasDestroyFn,
    sgemm: CublasSgemmFn,
}

struct CublasContext {
    _lib: Library,
    fns: CublasFns,
    handle: usize,
    driver: Arc<CudaDriver>,
}

impl Drop for CublasContext {
    fn drop(&mut self) {
        // SAFETY: Handle is created once and destroyed once on drop.
        let _ = unsafe { (self.fns.destroy)(self.handle as CublasHandle) };
        self.handle = 0;
    }
}

impl CublasContext {
    fn new(driver: Arc<CudaDriver>) -> BackendResult<Self> {
        let lib = load_cublas_library()?;
        let fns = CublasFns {
            create: load_cublas_symbol(&lib, b"cublasCreate_v2\0")?,
            destroy: load_cublas_symbol(&lib, b"cublasDestroy_v2\0")?,
            sgemm: load_cublas_symbol(&lib, b"cublasSgemm_v2\0")?,
        };

        driver.ensure_current()?;
        let mut handle: CublasHandle = std::ptr::null_mut();
        // SAFETY: cublasCreate_v2 initializes the output handle pointer.
        unsafe {
            check_cublas(
                (fns.create)(&mut handle as *mut CublasHandle),
                "cublasCreate_v2",
            )?;
        }

        Ok(Self {
            _lib: lib,
            fns,
            handle: handle as usize,
            driver,
        })
    }

    fn sgemm_row_major(
        &self,
        lhs: &DeviceBuffer,
        rhs: &DeviceBuffer,
        out: &DeviceBuffer,
        m: usize,
        n: usize,
        k: usize,
    ) -> BackendResult<()> {
        let m_i32 = i32::try_from(m)
            .map_err(|_| BackendError::execution("matrix dimension m exceeds i32"))?;
        let n_i32 = i32::try_from(n)
            .map_err(|_| BackendError::execution("matrix dimension n exceeds i32"))?;
        let k_i32 = i32::try_from(k)
            .map_err(|_| BackendError::execution("matrix dimension k exceeds i32"))?;

        self.driver.ensure_current()?;
        let alpha = 1.0f32;
        let beta = 0.0f32;
        // Row-major C = A * B using column-major GEMM: C^T = B^T * A^T.
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
                    &alpha as *const f32,
                    rhs.device_ptr() as usize as *const f32,
                    n_i32,
                    lhs.device_ptr() as usize as *const f32,
                    k_i32,
                    &beta as *const f32,
                    out.device_ptr() as usize as *mut f32,
                    n_i32,
                ),
                "cublasSgemm_v2",
            )?;
        }
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

impl Default for TritonExecutor {
    fn default() -> Self {
        Self::new()
    }
}
