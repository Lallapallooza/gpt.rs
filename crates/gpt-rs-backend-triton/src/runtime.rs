mod fused_dot_epilogue;
mod fused_elementwise;

use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use gpt_rs::backend::shape_helpers::{
    checked_element_count_or_error, contiguous_strides_or_error, static_dims_or_error,
};
use gpt_rs::backend::spec::{
    BackendError, BackendResult, ComparisonOp, CustomCallSpec, DType, ElementwiseBinaryOp,
    ElementwiseUnaryOp, Operand, Operation, ReduceKind, TensorLiteral, TensorSpec, ValueId,
    ValueType,
};
use gpt_rs::profiling::{self, ScopeMeta, WorkStats};
use libloading::Library;

use crate::artifact::TritonArtifact;
use crate::compiler::KernelCompiler;
use crate::device::{self, CudaDriver, CudaFunction, DeviceBuffer};
use crate::kernels::{
    KernelKind, KernelSpec, BROADCAST_KERNEL_ID, BROADCAST_SI32_KERNEL_ID,
    COMPARE_SI32_I1_KERNEL_ID, CONCAT_KERNEL_ID, DOT_BIAS_RANK2_KERNEL_ID,
    DYNAMIC_UPDATE_SLICE_F32_KERNEL_ID, EWISE_BINARY_KERNEL_ID, EWISE_UNARY_KERNEL_ID,
    EXTRACT_PATCHES_NHWC_KERNEL_ID, IOTA_SI32_KERNEL_ID, REDUCE_MAX_LAST_AXIS_KERNEL_ID,
    REDUCE_SUM_LAST_AXIS_KERNEL_ID, REDUCE_WINDOW_MAX_NHWC_KERNEL_ID, SELECT_I1_F32_KERNEL_ID,
    SLICE_KERNEL_ID, TAKE_F32_I32_KERNEL_ID, TRANSPOSE_KERNEL_ID,
};
use crate::targets::{TARGET_DOT_BIAS_FUSED_F32_V1, TARGET_ELEMENTWISE_FUSED_F32_V1};
use crate::tensor::TritonTensor;

thread_local! {
    static EXECUTION_KERNEL_STACK: RefCell<Vec<HashMap<String, Arc<LoadedKernel>>>> =
        const { RefCell::new(Vec::new()) };
}

static GPU_EVENT_TIMING_ENABLED: AtomicBool = AtomicBool::new(false);
static KERNEL_LAUNCH_COUNT: AtomicU64 = AtomicU64::new(0);

pub(crate) fn set_gpu_event_timing_enabled(enabled: bool) {
    GPU_EVENT_TIMING_ENABLED.store(enabled, Ordering::Relaxed);
}

pub struct TritonExecutor {
    compiler: KernelCompiler,
    loaded_kernels: Mutex<HashMap<u64, Arc<LoadedKernel>>>,
    fused_kernel_specs: Mutex<HashMap<u64, KernelSpec>>,
    cublas: OnceLock<Result<Arc<CublasContext>, String>>,
}

impl TritonExecutor {
    pub fn new() -> Self {
        Self {
            compiler: KernelCompiler::new(),
            loaded_kernels: Mutex::new(HashMap::new()),
            fused_kernel_specs: Mutex::new(HashMap::new()),
            cublas: OnceLock::new(),
        }
    }

    pub fn execute_artifact(
        &self,
        artifact: &TritonArtifact,
        entry_inputs: &[TritonTensor],
    ) -> BackendResult<Vec<TritonTensor>> {
        let function = artifact
            .program
            .functions
            .iter()
            .find(|function| function.name == artifact.program.entry)
            .ok_or_else(|| BackendError::execution("triton artifact entry function not found"))?;

        if function.parameter_ids.len() != entry_inputs.len() {
            return Err(BackendError::execution(format!(
                "triton entry input arity mismatch: expected {}, got {}",
                function.parameter_ids.len(),
                entry_inputs.len()
            )));
        }

        let driver = device::driver()?;
        let kernels = artifact
            .kernels
            .iter()
            .map(|spec| (spec.id.as_str(), spec))
            .collect::<HashMap<_, _>>();
        let preloaded = self.preload_execution_kernels(&driver, &artifact.kernels)?;
        let _execution_kernel_guard = ExecutionKernelGuard::push(preloaded);

        let mut values: HashMap<ValueId, TritonTensor> = HashMap::new();
        for (value_id, input) in function.parameter_ids.iter().zip(entry_inputs.iter()) {
            values.insert(*value_id, input.clone());
        }
        let launch_start = KERNEL_LAUNCH_COUNT.load(Ordering::Relaxed);
        let dispatch_start = std::time::Instant::now();

        for instruction in &function.body {
            if let Some(missing) = first_missing_operand_value(instruction, &values) {
                return Err(BackendError::execution(format!(
                    "triton runtime operand value {} missing before instruction {} ({:?})",
                    missing.0, instruction.id.0, instruction.op
                )));
            }
            let output = self.execute_instruction(&driver, &kernels, &values, instruction)?;
            values.insert(instruction.id, output);
        }

        let mut results = Vec::with_capacity(function.result_ids.len());
        for result_id in &function.result_ids {
            let value = values.get(result_id).cloned().ok_or_else(|| {
                BackendError::execution(format!(
                    "missing result value {} in triton runtime",
                    result_id.0
                ))
            })?;
            results.push(value);
        }
        let dispatch_elapsed = dispatch_start.elapsed();
        profiling::record_backend_aggregate(
            "backend.triton.dispatch",
            1,
            dispatch_elapsed,
            WorkStats {
                elements: function.body.len() as u64,
                ..WorkStats::default()
            },
        );
        let launch_end = KERNEL_LAUNCH_COUNT.load(Ordering::Relaxed);
        let launched = launch_end.saturating_sub(launch_start);
        if launched != 0 {
            profiling::record_backend_aggregate(
                "backend.triton.launch_count",
                launched,
                Duration::ZERO,
                WorkStats::default(),
            );
        }
        Ok(results)
    }

    fn preload_execution_kernels(
        &self,
        driver: &Arc<CudaDriver>,
        specs: &[KernelSpec],
    ) -> BackendResult<HashMap<String, Arc<LoadedKernel>>> {
        let mut out = HashMap::with_capacity(specs.len());
        for spec in specs {
            let loaded = self.load_kernel(driver, spec)?;
            out.insert(spec.id.clone(), loaded);
        }
        Ok(out)
    }

    fn execute_instruction(
        &self,
        driver: &Arc<CudaDriver>,
        kernels: &HashMap<&str, &KernelSpec>,
        values: &HashMap<ValueId, TritonTensor>,
        instruction: &gpt_rs::backend::spec::Instruction,
    ) -> BackendResult<TritonTensor> {
        match &instruction.op {
            Operation::Constant(literal) => literal_to_tensor(driver, literal),
            Operation::StopGradient | Operation::Reshape(_) => {
                let source =
                    self.resolve_operand_tensor(driver, values, instruction.operands.first())?;
                let out_spec = output_tensor_spec(&instruction.output)?;
                let src_bytes = byte_len(&source.spec)?;
                let out_bytes = byte_len(&out_spec)?;
                if src_bytes != out_bytes {
                    return Err(BackendError::execution(format!(
                        "alias byte-size mismatch: source={} bytes, output={} bytes",
                        src_bytes, out_bytes
                    )));
                }
                Ok(TritonTensor::new(out_spec, source.buffer))
            }
            Operation::ElementwiseBinary(op) => {
                let lhs =
                    self.resolve_operand_tensor(driver, values, instruction.operands.first())?;
                let rhs =
                    self.resolve_operand_tensor(driver, values, instruction.operands.get(1))?;
                let out_spec = output_tensor_spec(&instruction.output)?;
                let kernel = kernels.get(EWISE_BINARY_KERNEL_ID).ok_or_else(|| {
                    BackendError::execution("missing elementwise binary kernel in triton artifact")
                })?;
                self.execute_elementwise_binary(driver, kernel, *op, &lhs, &rhs, &out_spec)
            }
            Operation::ElementwiseUnary(op) => {
                let input =
                    self.resolve_operand_tensor(driver, values, instruction.operands.first())?;
                let out_spec = output_tensor_spec(&instruction.output)?;
                let kernel = kernels.get(EWISE_UNARY_KERNEL_ID).ok_or_else(|| {
                    BackendError::execution("missing elementwise unary kernel in triton artifact")
                })?;
                self.execute_elementwise_unary(driver, kernel, *op, &input, &out_spec)
            }
            Operation::BroadcastTo(_) => {
                let input =
                    self.resolve_operand_tensor(driver, values, instruction.operands.first())?;
                let out_spec = output_tensor_spec(&instruction.output)?;
                self.execute_broadcast(driver, kernels, &input, &out_spec)
            }
            Operation::Slice(spec) => {
                let input =
                    self.resolve_operand_tensor(driver, values, instruction.operands.first())?;
                let out_spec = output_tensor_spec(&instruction.output)?;
                self.execute_slice(driver, kernels, &input, spec, &out_spec)
            }
            Operation::DynamicSlice(spec) => {
                let input =
                    self.resolve_operand_tensor(driver, values, instruction.operands.first())?;
                let starts =
                    self.resolve_operand_tensor(driver, values, instruction.operands.get(1))?;
                let out_spec = output_tensor_spec(&instruction.output)?;
                self.execute_dynamic_slice(driver, kernels, &input, &starts, spec, &out_spec)
            }
            Operation::Transpose(spec) => {
                let input =
                    self.resolve_operand_tensor(driver, values, instruction.operands.first())?;
                let out_spec = output_tensor_spec(&instruction.output)?;
                self.execute_transpose(driver, kernels, &input, spec, &out_spec)
            }
            Operation::Concat(spec) => {
                if instruction.operands.len() != 2 {
                    return Err(BackendError::execution(
                        "triton concat runtime currently supports exactly two operands",
                    ));
                }
                let lhs =
                    self.resolve_operand_tensor(driver, values, instruction.operands.first())?;
                let rhs =
                    self.resolve_operand_tensor(driver, values, instruction.operands.get(1))?;
                let out_spec = output_tensor_spec(&instruction.output)?;
                self.execute_concat(driver, kernels, &lhs, &rhs, spec, &out_spec)
            }
            Operation::Iota(spec) => {
                let out_spec = output_tensor_spec(&instruction.output)?;
                let kernel = kernels.get(IOTA_SI32_KERNEL_ID).ok_or_else(|| {
                    BackendError::execution("missing iota kernel in triton artifact")
                })?;
                self.execute_iota(driver, kernel, spec, &out_spec)
            }
            Operation::Compare(spec) => {
                let lhs =
                    self.resolve_operand_tensor(driver, values, instruction.operands.first())?;
                let rhs =
                    self.resolve_operand_tensor(driver, values, instruction.operands.get(1))?;
                let out_spec = output_tensor_spec(&instruction.output)?;
                let kernel = kernels.get(COMPARE_SI32_I1_KERNEL_ID).ok_or_else(|| {
                    BackendError::execution("missing compare kernel in triton artifact")
                })?;
                self.execute_compare(driver, kernel, spec.op, &lhs, &rhs, &out_spec)
            }
            Operation::Select => {
                let pred =
                    self.resolve_operand_tensor(driver, values, instruction.operands.first())?;
                let when_true =
                    self.resolve_operand_tensor(driver, values, instruction.operands.get(1))?;
                let when_false =
                    self.resolve_operand_tensor(driver, values, instruction.operands.get(2))?;
                let out_spec = output_tensor_spec(&instruction.output)?;
                let kernel = kernels.get(SELECT_I1_F32_KERNEL_ID).ok_or_else(|| {
                    BackendError::execution("missing select kernel in triton artifact")
                })?;
                self.execute_select(driver, kernel, &pred, &when_true, &when_false, &out_spec)
            }
            Operation::Take => {
                let params =
                    self.resolve_operand_tensor(driver, values, instruction.operands.first())?;
                let indices =
                    self.resolve_operand_tensor(driver, values, instruction.operands.get(1))?;
                let out_spec = output_tensor_spec(&instruction.output)?;
                let kernel = kernels.get(TAKE_F32_I32_KERNEL_ID).ok_or_else(|| {
                    BackendError::execution("missing take kernel in triton artifact")
                })?;
                self.execute_take(driver, kernel, &params, &indices, &out_spec)
            }
            Operation::DynamicUpdateSlice(_spec) => {
                let base =
                    self.resolve_operand_tensor(driver, values, instruction.operands.first())?;
                let update =
                    self.resolve_operand_tensor(driver, values, instruction.operands.get(1))?;
                let starts =
                    self.resolve_operand_tensor(driver, values, instruction.operands.get(2))?;
                let out_spec = output_tensor_spec(&instruction.output)?;
                let kernel = kernels
                    .get(DYNAMIC_UPDATE_SLICE_F32_KERNEL_ID)
                    .ok_or_else(|| {
                        BackendError::execution(
                            "missing dynamic_update_slice kernel in triton artifact",
                        )
                    })?;
                self.execute_dynamic_update_slice(
                    driver, kernel, &base, &update, &starts, &out_spec,
                )
            }
            Operation::DotGeneral(spec) => {
                let lhs =
                    self.resolve_operand_tensor(driver, values, instruction.operands.first())?;
                let rhs =
                    self.resolve_operand_tensor(driver, values, instruction.operands.get(1))?;
                let out_spec = output_tensor_spec(&instruction.output)?;
                let args = DotGeneralArgs {
                    spec,
                    lhs_spec: &lhs.spec,
                    rhs_spec: &rhs.spec,
                    out_spec: &out_spec,
                };
                self.execute_dot_general(driver, args, &lhs, &rhs)
            }
            Operation::Reduce(spec) => {
                let input =
                    self.resolve_operand_tensor(driver, values, instruction.operands.first())?;
                let out_spec = output_tensor_spec(&instruction.output)?;
                self.execute_reduce(driver, kernels, &input.spec, &out_spec, spec, &input)
            }
            Operation::ExtractPatches(spec) => {
                let input =
                    self.resolve_operand_tensor(driver, values, instruction.operands.first())?;
                let out_spec = output_tensor_spec(&instruction.output)?;
                let kernel = kernels.get(EXTRACT_PATCHES_NHWC_KERNEL_ID).ok_or_else(|| {
                    BackendError::execution("missing extract_patches kernel in triton artifact")
                })?;
                self.execute_extract_patches(driver, kernel, &input, spec, &out_spec)
            }
            Operation::ReduceWindow(spec) => {
                let input =
                    self.resolve_operand_tensor(driver, values, instruction.operands.first())?;
                let out_spec = output_tensor_spec(&instruction.output)?;
                let kernel = kernels
                    .get(REDUCE_WINDOW_MAX_NHWC_KERNEL_ID)
                    .ok_or_else(|| {
                        BackendError::execution("missing reduce_window kernel in triton artifact")
                    })?;
                self.execute_reduce_window(driver, kernel, &input, spec, &out_spec)
            }
            Operation::CustomCall(spec) => {
                let out_spec = output_tensor_spec(&instruction.output)?;
                self.execute_custom_call(driver, kernels, values, instruction, spec, &out_spec)
            }
            other => Err(BackendError::execution(format!(
                "triton runtime does not support instruction op: {:?}",
                other
            ))),
        }
    }

    fn resolve_operand_tensor(
        &self,
        driver: &Arc<CudaDriver>,
        values: &HashMap<ValueId, TritonTensor>,
        operand: Option<&Operand>,
    ) -> BackendResult<TritonTensor> {
        match operand {
            Some(Operand::Value(id)) => values.get(id).cloned().ok_or_else(|| {
                BackendError::execution(format!(
                    "operand value {} missing from triton runtime state",
                    id.0
                ))
            }),
            Some(Operand::Literal(literal)) => literal_to_tensor(driver, literal),
            Some(Operand::TupleElement { .. }) => Err(BackendError::execution(
                "tuple element operands are not supported by triton runtime",
            )),
            None => Err(BackendError::execution("missing instruction operand")),
        }
    }

    fn execute_custom_call(
        &self,
        driver: &Arc<CudaDriver>,
        kernels: &HashMap<&str, &KernelSpec>,
        values: &HashMap<ValueId, TritonTensor>,
        instruction: &gpt_rs::backend::spec::Instruction,
        spec: &CustomCallSpec,
        out_spec: &TensorSpec,
    ) -> BackendResult<TritonTensor> {
        match spec.target.as_str() {
            TARGET_ELEMENTWISE_FUSED_F32_V1 => self.execute_fused_elementwise_custom_call(
                driver,
                values,
                instruction,
                spec,
                out_spec,
            ),
            TARGET_DOT_BIAS_FUSED_F32_V1 => self.execute_fused_dot_bias_custom_call(
                driver,
                kernels,
                values,
                instruction,
                spec,
                out_spec,
            ),
            _ => Err(BackendError::execution(format!(
                "unsupported triton custom_call target '{}'",
                spec.target
            ))),
        }
    }

    fn execute_fused_elementwise_custom_call(
        &self,
        driver: &Arc<CudaDriver>,
        values: &HashMap<ValueId, TritonTensor>,
        instruction: &gpt_rs::backend::spec::Instruction,
        spec: &CustomCallSpec,
        out_spec: &TensorSpec,
    ) -> BackendResult<TritonTensor> {
        let plan = fused_elementwise::FusedElementwisePlan::parse(spec)?;
        let input_tensors = instruction
            .operands
            .iter()
            .map(|operand| self.resolve_operand_tensor(driver, values, Some(operand)))
            .collect::<BackendResult<Vec<_>>>()?;
        self.execute_fused_elementwise_plan(driver, &plan, input_tensors.as_slice(), out_spec)
    }

    fn execute_fused_elementwise_plan(
        &self,
        driver: &Arc<CudaDriver>,
        plan: &fused_elementwise::FusedElementwisePlan,
        input_tensors: &[TritonTensor],
        out_spec: &TensorSpec,
    ) -> BackendResult<TritonTensor> {
        let input_specs = input_tensors
            .iter()
            .map(|tensor| tensor.spec.clone())
            .collect::<Vec<_>>();
        let kernel_key = plan.cache_fingerprint(out_spec, input_specs.as_slice())?;
        let kernel = {
            let mut cache = self
                .fused_kernel_specs
                .lock()
                .expect("triton fused kernel-spec cache poisoned");
            if let Some(cached) = cache.get(&kernel_key).cloned() {
                profiling::cache_event("triton_backend.fused_kernel_spec_hit");
                cached
            } else {
                profiling::cache_event("triton_backend.fused_kernel_spec_miss");
                let built = plan.build_kernel_spec(out_spec, input_specs.as_slice())?;
                cache.insert(kernel_key, built.clone());
                built
            }
        };
        let out = TritonTensor::new(out_spec.clone(), driver.alloc_zeroed(byte_len(out_spec)?)?);
        let loaded = self.load_kernel(driver, &kernel)?;
        let n = static_element_count(&out_spec.shape)?;
        if n == 0 {
            return Ok(out);
        }
        let n_u32 = u32::try_from(n).map_err(|_| {
            BackendError::execution("fused elementwise element count exceeds u32 range")
        })?;
        let mut params: Vec<*mut c_void> = Vec::with_capacity(input_tensors.len() + 3);
        let mut ptr_args: Vec<u64> = input_tensors
            .iter()
            .map(|tensor| tensor.buffer.device_ptr())
            .collect::<Vec<_>>();
        for ptr in &mut ptr_args {
            params.push((ptr as *mut u64).cast::<c_void>());
        }
        let mut out_ptr = out.buffer.device_ptr();
        params.push((&mut out_ptr as *mut u64).cast::<c_void>());
        let mut n_kernel = n_u32;
        params.push((&mut n_kernel as *mut u32).cast::<c_void>());
        let mut opaque_ptr = 0u64;
        params.push((&mut opaque_ptr as *mut u64).cast::<c_void>());
        launch_1d(driver, &loaded, n_u32, 256, params.as_mut_slice())?;
        Ok(out)
    }

    fn execute_fused_dot_bias_custom_call(
        &self,
        driver: &Arc<CudaDriver>,
        kernels: &HashMap<&str, &KernelSpec>,
        values: &HashMap<ValueId, TritonTensor>,
        instruction: &gpt_rs::backend::spec::Instruction,
        spec: &CustomCallSpec,
        out_spec: &TensorSpec,
    ) -> BackendResult<TritonTensor> {
        let _scope = profiling::backend_scope("backend.triton.fused.dot_epilogue");
        if instruction.operands.len() < 3 {
            return Err(BackendError::execution(
                "fused dot+bias custom_call requires at least three operands",
            ));
        }
        let plan = fused_dot_epilogue::FusedDotBiasPlan::parse(spec, instruction.operands.len())?;
        let lhs = self.resolve_operand_tensor(driver, values, instruction.operands.first())?;
        let rhs = self.resolve_operand_tensor(driver, values, instruction.operands.get(1))?;
        let bias =
            self.resolve_operand_tensor(driver, values, instruction.operands.get(plan.add_input))?;

        if let Some(kernel) = kernels.get(DOT_BIAS_RANK2_KERNEL_ID) {
            let rank2_args = DotBiasRank2Args {
                driver,
                kernel,
                plan: &plan,
                lhs: &lhs,
                rhs: &rhs,
                bias: &bias,
                out_spec,
            };
            if let Some(out) = self.try_execute_dot_bias_rank2(rank2_args)? {
                return Ok(out);
            }
        }

        ensure_static_broadcastable(&out_spec.shape, &bias.spec.shape, "fused dot+bias bias")?;

        let dot = self.execute_dot_general(
            driver,
            DotGeneralArgs {
                spec: &plan.dot,
                lhs_spec: &lhs.spec,
                rhs_spec: &rhs.spec,
                out_spec,
            },
            &lhs,
            &rhs,
        )?;
        let epilogue_plan = fused_elementwise::FusedElementwisePlan::from_components(
            vec![1],
            vec![0],
            vec![0],
            vec![1],
        )?;
        let epilogue_inputs = vec![dot, bias];
        self.execute_fused_elementwise_plan(
            driver,
            &epilogue_plan,
            epilogue_inputs.as_slice(),
            out_spec,
        )
    }

    fn try_execute_dot_bias_rank2(
        &self,
        args: DotBiasRank2Args<'_>,
    ) -> BackendResult<Option<TritonTensor>> {
        let DotBiasRank2Args {
            driver,
            kernel,
            plan,
            lhs,
            rhs,
            bias,
            out_spec,
        } = args;
        if !matches!(kernel.kind, KernelKind::DotBiasRank2F32) {
            return Err(BackendError::execution(format!(
                "unexpected dot+bias kernel kind: {:?}",
                kernel.kind
            )));
        }

        if !plan.dot.batch_lhs.is_empty()
            || !plan.dot.batch_rhs.is_empty()
            || plan.dot.contract_lhs.as_slice() != [1]
            || plan.dot.contract_rhs.as_slice() != [0]
        {
            return Ok(None);
        }

        if lhs.spec.dtype != DType::F32
            || rhs.spec.dtype != DType::F32
            || bias.spec.dtype != DType::F32
            || out_spec.dtype != DType::F32
        {
            return Ok(None);
        }

        let lhs_dims = static_dims_or_error(&lhs.spec.shape, |_| {
            BackendError::execution("dynamic dimensions are not supported by triton runtime")
        })?;
        let rhs_dims = static_dims_or_error(&rhs.spec.shape, |_| {
            BackendError::execution("dynamic dimensions are not supported by triton runtime")
        })?;
        let bias_dims = static_dims_or_error(&bias.spec.shape, |_| {
            BackendError::execution("dynamic dimensions are not supported by triton runtime")
        })?;
        let out_dims = static_dims_or_error(&out_spec.shape, |_| {
            BackendError::execution("dynamic dimensions are not supported by triton runtime")
        })?;

        if lhs_dims.len() != 2 || rhs_dims.len() != 2 || out_dims.len() != 2 {
            return Ok(None);
        }

        let m = lhs_dims[0];
        let k = lhs_dims[1];
        let rhs_k = rhs_dims[0];
        let n = rhs_dims[1];
        if k != rhs_k || out_dims[0] != m || out_dims[1] != n {
            return Ok(None);
        }
        let bias_rank1 = bias_dims.len() == 1 && bias_dims[0] == n;
        let bias_row = bias_dims.len() == 2 && m == 1 && bias_dims[0] == 1 && bias_dims[1] == n;
        if !(bias_rank1 || bias_row) {
            return Ok(None);
        }

        if m == 0 || n == 0 || k == 0 {
            let out =
                TritonTensor::new(out_spec.clone(), driver.alloc_zeroed(byte_len(out_spec)?)?);
            return Ok(Some(out));
        }

        let mut m_i32 = i32::try_from(m)
            .map_err(|_| BackendError::execution("dot+bias m exceeds i32 range"))?;
        let mut n_i32 = i32::try_from(n)
            .map_err(|_| BackendError::execution("dot+bias n exceeds i32 range"))?;
        let mut k_i32 = i32::try_from(k)
            .map_err(|_| BackendError::execution("dot+bias k exceeds i32 range"))?;

        let out = TritonTensor::new(out_spec.clone(), driver.alloc_zeroed(byte_len(out_spec)?)?);
        let loaded = self.load_kernel(driver, kernel)?;

        let mut lhs_ptr = lhs.buffer.device_ptr();
        let mut rhs_ptr = rhs.buffer.device_ptr();
        let mut bias_ptr = bias.buffer.device_ptr();
        let mut out_ptr = out.buffer.device_ptr();
        let mut opaque_ptr = 0u64;

        let mut params = [
            (&mut lhs_ptr as *mut u64).cast::<c_void>(),
            (&mut rhs_ptr as *mut u64).cast::<c_void>(),
            (&mut bias_ptr as *mut u64).cast::<c_void>(),
            (&mut out_ptr as *mut u64).cast::<c_void>(),
            (&mut m_i32 as *mut i32).cast::<c_void>(),
            (&mut n_i32 as *mut i32).cast::<c_void>(),
            (&mut k_i32 as *mut i32).cast::<c_void>(),
            (&mut opaque_ptr as *mut u64).cast::<c_void>(),
        ];

        let block_n = 128u32;
        let m_u32 = u32::try_from(m)
            .map_err(|_| BackendError::execution("dot+bias m exceeds u32 range"))?;
        let n_u32 = u32::try_from(n)
            .map_err(|_| BackendError::execution("dot+bias n exceeds u32 range"))?;
        let n_blocks = n_u32.div_ceil(block_n);
        let work = m_u32
            .checked_mul(n_u32)
            .ok_or_else(|| BackendError::execution("dot+bias work element overflow"))?;
        launch_program_grid_2d(driver, &loaded, m_u32, n_blocks, block_n, work, &mut params)?;
        Ok(Some(out))
    }

    fn execute_elementwise_binary(
        &self,
        driver: &Arc<CudaDriver>,
        kernel: &KernelSpec,
        op: ElementwiseBinaryOp,
        lhs: &TritonTensor,
        rhs: &TritonTensor,
        spec: &TensorSpec,
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

        let mut lhs_ptr = lhs.buffer.device_ptr();
        let mut rhs_ptr = rhs.buffer.device_ptr();
        let mut out_ptr = out.buffer.device_ptr();
        let mut n = count_u32;
        let mut op_u32 = opcode;
        // Python Triton-compiled kernels currently expose one trailing opaque
        // pointer parameter; built-in kernels ignore this extra argument.
        let mut opaque_ptr = 0u64;
        let mut params = [
            (&mut lhs_ptr as *mut u64).cast::<c_void>(),
            (&mut rhs_ptr as *mut u64).cast::<c_void>(),
            (&mut out_ptr as *mut u64).cast::<c_void>(),
            (&mut n as *mut u32).cast::<c_void>(),
            (&mut op_u32 as *mut u32).cast::<c_void>(),
            (&mut opaque_ptr as *mut u64).cast::<c_void>(),
        ];
        launch_1d(driver, &loaded, count_u32, 256, &mut params)?;

        Ok(out)
    }

    fn execute_elementwise_unary(
        &self,
        driver: &Arc<CudaDriver>,
        kernel: &KernelSpec,
        op: ElementwiseUnaryOp,
        input: &TritonTensor,
        spec: &TensorSpec,
    ) -> BackendResult<TritonTensor> {
        if input.spec != *spec {
            return Err(BackendError::execution(
                "elementwise unary operand/spec mismatch in triton runtime",
            ));
        }
        if spec.dtype != DType::F32 {
            return Err(BackendError::execution(format!(
                "elementwise unary runtime currently supports F32 only, got {:?}",
                spec.dtype
            )));
        }
        if !matches!(kernel.kind, KernelKind::ElementwiseUnaryF32) {
            return Err(BackendError::execution(format!(
                "unexpected kernel kind for elementwise unary: {:?}",
                kernel.kind
            )));
        }

        let element_count = static_element_count(&spec.shape)?;
        let out = TritonTensor::new(spec.clone(), driver.alloc_zeroed(byte_len(spec)?)?);

        let loaded = self.load_kernel(driver, kernel)?;
        let opcode = unary_opcode(op)?;
        let count_u32 = u32::try_from(element_count).map_err(|_| {
            BackendError::execution("elementwise unary tensor too large for u32 launch size")
        })?;

        let mut in_ptr = input.buffer.device_ptr();
        let mut out_ptr = out.buffer.device_ptr();
        let mut n = count_u32;
        let mut op_u32 = opcode;
        // Python Triton-compiled kernels currently expose one trailing opaque
        // pointer parameter; built-in kernels ignore this extra argument.
        let mut opaque_ptr = 0u64;
        let mut params = [
            (&mut in_ptr as *mut u64).cast::<c_void>(),
            (&mut out_ptr as *mut u64).cast::<c_void>(),
            (&mut n as *mut u32).cast::<c_void>(),
            (&mut op_u32 as *mut u32).cast::<c_void>(),
            (&mut opaque_ptr as *mut u64).cast::<c_void>(),
        ];
        launch_1d(driver, &loaded, count_u32, 256, &mut params)?;

        Ok(out)
    }

    fn execute_broadcast(
        &self,
        driver: &Arc<CudaDriver>,
        kernels: &HashMap<&str, &KernelSpec>,
        input: &TritonTensor,
        out_spec: &TensorSpec,
    ) -> BackendResult<TritonTensor> {
        if input.spec.dtype != out_spec.dtype {
            return Err(BackendError::execution(
                "broadcast input/output dtype mismatch",
            ));
        }
        if !matches!(out_spec.dtype, DType::F32 | DType::Si32) {
            return Err(BackendError::execution(format!(
                "broadcast runtime supports F32/Si32 only, got {:?}",
                out_spec.dtype
            )));
        }

        let out = TritonTensor::new(out_spec.clone(), driver.alloc_zeroed(byte_len(out_spec)?)?);
        let element_count = static_element_count(&out_spec.shape)?;
        if element_count == 0 {
            return Ok(out);
        }

        let kernel_id = match out_spec.dtype {
            DType::F32 => BROADCAST_KERNEL_ID,
            DType::Si32 => BROADCAST_SI32_KERNEL_ID,
            _ => {
                return Err(BackendError::execution(format!(
                    "unsupported broadcast dtype {:?}",
                    out_spec.dtype
                )))
            }
        };
        let kernel = kernels.get(kernel_id).ok_or_else(|| {
            BackendError::execution(format!("missing broadcast kernel {kernel_id} in artifact"))
        })?;
        let expected_kind = match out_spec.dtype {
            DType::F32 => KernelKind::BroadcastF32Rank4,
            DType::Si32 => KernelKind::BroadcastSi32Rank4,
            _ => unreachable!(),
        };
        if kernel.kind != expected_kind {
            return Err(BackendError::execution(format!(
                "unexpected broadcast kernel kind: {:?}",
                kernel.kind
            )));
        }

        let (out_dims, in_strides) = broadcast_rank4_layout(&input.spec.shape, &out_spec.shape)?;
        let loaded = self.load_kernel(driver, kernel)?;

        let mut in_ptr = input.buffer.device_ptr();
        let mut out_ptr = out.buffer.device_ptr();
        let mut n = u32::try_from(element_count)
            .map_err(|_| BackendError::execution("broadcast element count exceeds u32 range"))?;
        let mut od0 = out_dims[0];
        let mut od1 = out_dims[1];
        let mut od2 = out_dims[2];
        let mut od3 = out_dims[3];
        let mut is0 = in_strides[0];
        let mut is1 = in_strides[1];
        let mut is2 = in_strides[2];
        let mut is3 = in_strides[3];
        let mut opaque_ptr = 0u64;
        let mut params = [
            (&mut in_ptr as *mut u64).cast::<c_void>(),
            (&mut out_ptr as *mut u64).cast::<c_void>(),
            (&mut n as *mut u32).cast::<c_void>(),
            (&mut od0 as *mut i32).cast::<c_void>(),
            (&mut od1 as *mut i32).cast::<c_void>(),
            (&mut od2 as *mut i32).cast::<c_void>(),
            (&mut od3 as *mut i32).cast::<c_void>(),
            (&mut is0 as *mut i32).cast::<c_void>(),
            (&mut is1 as *mut i32).cast::<c_void>(),
            (&mut is2 as *mut i32).cast::<c_void>(),
            (&mut is3 as *mut i32).cast::<c_void>(),
            (&mut opaque_ptr as *mut u64).cast::<c_void>(),
        ];
        launch_1d(driver, &loaded, n, 256, &mut params)?;
        Ok(out)
    }

    fn execute_slice(
        &self,
        driver: &Arc<CudaDriver>,
        kernels: &HashMap<&str, &KernelSpec>,
        input: &TritonTensor,
        spec: &gpt_rs::backend::spec::SliceSpec,
        out_spec: &TensorSpec,
    ) -> BackendResult<TritonTensor> {
        if input.spec.dtype != out_spec.dtype {
            return Err(BackendError::execution("slice input/output dtype mismatch"));
        }
        if spec.starts.len() != spec.sizes.len() {
            return Err(BackendError::execution(
                "slice starts and sizes length mismatch",
            ));
        }

        match out_spec.dtype {
            DType::F32 => self.execute_slice_f32(driver, kernels, input, &spec.starts, out_spec),
            DType::Si32 => {
                // Minimal i32 slice support for attention decode path (rank-1).
                let in_dims = static_dims_or_error(&input.spec.shape, |_| {
                    BackendError::execution(
                        "dynamic dimensions are not supported by triton runtime",
                    )
                })?;
                let out_dims = static_dims_or_error(&out_spec.shape, |_| {
                    BackendError::execution(
                        "dynamic dimensions are not supported by triton runtime",
                    )
                })?;
                if in_dims.len() != 1 || out_dims.len() != 1 || spec.starts.len() != 1 {
                    return Err(BackendError::execution(
                        "slice Si32 path currently supports rank-1 only",
                    ));
                }
                let start = spec.starts[0];
                let len = out_dims[0];
                if match start.checked_add(len) {
                    Some(end) => end > in_dims[0],
                    None => true,
                } {
                    return Err(BackendError::execution("slice Si32 bounds check failed"));
                }
                let out =
                    TritonTensor::new(out_spec.clone(), driver.alloc_zeroed(byte_len(out_spec)?)?);
                let offset_bytes = start
                    .checked_mul(4)
                    .ok_or_else(|| BackendError::execution("slice Si32 offset overflow"))?;
                let src_ptr = input
                    .buffer
                    .device_ptr()
                    .checked_add(u64::try_from(offset_bytes).map_err(|_| {
                        BackendError::execution("slice Si32 offset conversion overflow")
                    })?)
                    .ok_or_else(|| BackendError::execution("slice Si32 pointer overflow"))?;
                driver.copy_device_to_device(out.buffer.device_ptr(), src_ptr, len * 4)?;
                Ok(out)
            }
            _ => Err(BackendError::execution(format!(
                "slice runtime supports F32/Si32 only, got {:?}",
                out_spec.dtype
            ))),
        }
    }

    fn execute_slice_f32(
        &self,
        driver: &Arc<CudaDriver>,
        kernels: &HashMap<&str, &KernelSpec>,
        input: &TritonTensor,
        starts: &[usize],
        out_spec: &TensorSpec,
    ) -> BackendResult<TritonTensor> {
        let kernel = kernels
            .get(SLICE_KERNEL_ID)
            .ok_or_else(|| BackendError::execution("missing slice kernel in triton artifact"))?;
        if !matches!(kernel.kind, KernelKind::SliceF32Rank4) {
            return Err(BackendError::execution(format!(
                "unexpected slice kernel kind: {:?}",
                kernel.kind
            )));
        }
        let (out_dims, in_strides, starts4) =
            slice_rank4_layout(&input.spec.shape, &out_spec.shape, starts)?;

        let out = TritonTensor::new(out_spec.clone(), driver.alloc_zeroed(byte_len(out_spec)?)?);
        let element_count = static_element_count(&out_spec.shape)?;
        if element_count == 0 {
            return Ok(out);
        }
        let loaded = self.load_kernel(driver, kernel)?;

        let mut in_ptr = input.buffer.device_ptr();
        let mut out_ptr = out.buffer.device_ptr();
        let mut n = u32::try_from(element_count)
            .map_err(|_| BackendError::execution("slice element count exceeds u32 range"))?;
        let mut od0 = out_dims[0];
        let mut od1 = out_dims[1];
        let mut od2 = out_dims[2];
        let mut od3 = out_dims[3];
        let mut is0 = in_strides[0];
        let mut is1 = in_strides[1];
        let mut is2 = in_strides[2];
        let mut is3 = in_strides[3];
        let mut st0 = starts4[0];
        let mut st1 = starts4[1];
        let mut st2 = starts4[2];
        let mut st3 = starts4[3];
        let mut opaque_ptr = 0u64;
        let mut params = [
            (&mut in_ptr as *mut u64).cast::<c_void>(),
            (&mut out_ptr as *mut u64).cast::<c_void>(),
            (&mut n as *mut u32).cast::<c_void>(),
            (&mut od0 as *mut i32).cast::<c_void>(),
            (&mut od1 as *mut i32).cast::<c_void>(),
            (&mut od2 as *mut i32).cast::<c_void>(),
            (&mut od3 as *mut i32).cast::<c_void>(),
            (&mut is0 as *mut i32).cast::<c_void>(),
            (&mut is1 as *mut i32).cast::<c_void>(),
            (&mut is2 as *mut i32).cast::<c_void>(),
            (&mut is3 as *mut i32).cast::<c_void>(),
            (&mut st0 as *mut i32).cast::<c_void>(),
            (&mut st1 as *mut i32).cast::<c_void>(),
            (&mut st2 as *mut i32).cast::<c_void>(),
            (&mut st3 as *mut i32).cast::<c_void>(),
            (&mut opaque_ptr as *mut u64).cast::<c_void>(),
        ];
        launch_1d(driver, &loaded, n, 256, &mut params)?;
        Ok(out)
    }

    fn execute_dynamic_slice(
        &self,
        driver: &Arc<CudaDriver>,
        kernels: &HashMap<&str, &KernelSpec>,
        input: &TritonTensor,
        starts: &TritonTensor,
        spec: &gpt_rs::backend::spec::DynamicSliceSpec,
        out_spec: &TensorSpec,
    ) -> BackendResult<TritonTensor> {
        if starts.spec.dtype != DType::Si32 {
            return Err(BackendError::execution("dynamic_slice starts must be Si32"));
        }
        let rank = input.spec.shape.rank();
        if spec.sizes.len() != rank {
            return Err(BackendError::execution(
                "dynamic_slice sizes length must match input rank",
            ));
        }
        let starts_values = read_i32_tensor(starts)?;
        if starts_values.len() != rank {
            return Err(BackendError::execution(
                "dynamic_slice starts length must match input rank",
            ));
        }
        let mut static_starts = Vec::with_capacity(rank);
        for value in starts_values {
            if value < 0 {
                return Err(BackendError::execution(
                    "dynamic_slice starts must be non-negative",
                ));
            }
            static_starts.push(value as usize);
        }

        match out_spec.dtype {
            DType::F32 => self.execute_slice_f32(driver, kernels, input, &static_starts, out_spec),
            DType::Si32 => {
                // Rank-1 Si32 dynamic slice is required by decode attention query position path.
                let in_dims = static_dims_or_error(&input.spec.shape, |_| {
                    BackendError::execution(
                        "dynamic dimensions are not supported by triton runtime",
                    )
                })?;
                if in_dims.len() != 1 || static_starts.len() != 1 || spec.sizes.len() != 1 {
                    return Err(BackendError::execution(
                        "dynamic_slice Si32 path currently supports rank-1 only",
                    ));
                }
                let start = static_starts[0];
                let len = spec.sizes[0];
                if match start.checked_add(len) {
                    Some(end) => end > in_dims[0],
                    None => true,
                } {
                    return Err(BackendError::execution(
                        "dynamic_slice Si32 bounds check failed",
                    ));
                }
                let out =
                    TritonTensor::new(out_spec.clone(), driver.alloc_zeroed(byte_len(out_spec)?)?);
                let offset_bytes = start
                    .checked_mul(4)
                    .ok_or_else(|| BackendError::execution("dynamic_slice Si32 offset overflow"))?;
                let src_ptr = input
                    .buffer
                    .device_ptr()
                    .checked_add(u64::try_from(offset_bytes).map_err(|_| {
                        BackendError::execution("dynamic_slice Si32 offset conversion overflow")
                    })?)
                    .ok_or_else(|| {
                        BackendError::execution("dynamic_slice Si32 pointer overflow")
                    })?;
                driver.copy_device_to_device(out.buffer.device_ptr(), src_ptr, len * 4)?;
                Ok(out)
            }
            _ => Err(BackendError::execution(format!(
                "dynamic_slice unsupported dtype {:?}",
                out_spec.dtype
            ))),
        }
    }

    fn execute_transpose(
        &self,
        driver: &Arc<CudaDriver>,
        kernels: &HashMap<&str, &KernelSpec>,
        input: &TritonTensor,
        spec: &gpt_rs::backend::spec::TransposeSpec,
        out_spec: &TensorSpec,
    ) -> BackendResult<TritonTensor> {
        if input.spec.dtype != DType::F32 || out_spec.dtype != DType::F32 {
            return Err(BackendError::execution(
                "transpose runtime currently supports F32 only",
            ));
        }
        let kernel = kernels.get(TRANSPOSE_KERNEL_ID).ok_or_else(|| {
            BackendError::execution("missing transpose kernel in triton artifact")
        })?;
        if !matches!(kernel.kind, KernelKind::TransposeF32Rank5) {
            return Err(BackendError::execution(format!(
                "unexpected transpose kernel kind: {:?}",
                kernel.kind
            )));
        }
        let (out_dims, in_strides) =
            transpose_rank5_layout(&input.spec.shape, &out_spec.shape, &spec.perm)?;
        let out = TritonTensor::new(out_spec.clone(), driver.alloc_zeroed(byte_len(out_spec)?)?);
        let element_count = static_element_count(&out_spec.shape)?;
        if element_count == 0 {
            return Ok(out);
        }
        let loaded = self.load_kernel(driver, kernel)?;

        let mut in_ptr = input.buffer.device_ptr();
        let mut out_ptr = out.buffer.device_ptr();
        let mut n = u32::try_from(element_count)
            .map_err(|_| BackendError::execution("transpose element count exceeds u32 range"))?;
        let mut od0 = out_dims[0];
        let mut od1 = out_dims[1];
        let mut od2 = out_dims[2];
        let mut od3 = out_dims[3];
        let mut od4 = out_dims[4];
        let mut is0 = in_strides[0];
        let mut is1 = in_strides[1];
        let mut is2 = in_strides[2];
        let mut is3 = in_strides[3];
        let mut is4 = in_strides[4];
        let mut opaque_ptr = 0u64;
        let mut params = [
            (&mut in_ptr as *mut u64).cast::<c_void>(),
            (&mut out_ptr as *mut u64).cast::<c_void>(),
            (&mut n as *mut u32).cast::<c_void>(),
            (&mut od0 as *mut i32).cast::<c_void>(),
            (&mut od1 as *mut i32).cast::<c_void>(),
            (&mut od2 as *mut i32).cast::<c_void>(),
            (&mut od3 as *mut i32).cast::<c_void>(),
            (&mut od4 as *mut i32).cast::<c_void>(),
            (&mut is0 as *mut i32).cast::<c_void>(),
            (&mut is1 as *mut i32).cast::<c_void>(),
            (&mut is2 as *mut i32).cast::<c_void>(),
            (&mut is3 as *mut i32).cast::<c_void>(),
            (&mut is4 as *mut i32).cast::<c_void>(),
            (&mut opaque_ptr as *mut u64).cast::<c_void>(),
        ];
        launch_1d(driver, &loaded, n, 256, &mut params)?;
        Ok(out)
    }

    fn execute_concat(
        &self,
        driver: &Arc<CudaDriver>,
        kernels: &HashMap<&str, &KernelSpec>,
        lhs: &TritonTensor,
        rhs: &TritonTensor,
        spec: &gpt_rs::backend::spec::ConcatSpec,
        out_spec: &TensorSpec,
    ) -> BackendResult<TritonTensor> {
        if lhs.spec.dtype != DType::F32
            || rhs.spec.dtype != DType::F32
            || out_spec.dtype != DType::F32
        {
            return Err(BackendError::execution(
                "concat runtime currently supports F32 only",
            ));
        }
        let kernel = kernels
            .get(CONCAT_KERNEL_ID)
            .ok_or_else(|| BackendError::execution("missing concat kernel in triton artifact"))?;
        if !matches!(kernel.kind, KernelKind::ConcatF32Rank4) {
            return Err(BackendError::execution(format!(
                "unexpected concat kernel kind: {:?}",
                kernel.kind
            )));
        }
        let (out_dims, lhs_strides, rhs_strides, axis, split) =
            concat_rank4_layout(&lhs.spec.shape, &rhs.spec.shape, &out_spec.shape, spec.axis)?;
        let out = TritonTensor::new(out_spec.clone(), driver.alloc_zeroed(byte_len(out_spec)?)?);
        let element_count = static_element_count(&out_spec.shape)?;
        if element_count == 0 {
            return Ok(out);
        }
        let loaded = self.load_kernel(driver, kernel)?;

        let mut lhs_ptr = lhs.buffer.device_ptr();
        let mut rhs_ptr = rhs.buffer.device_ptr();
        let mut out_ptr = out.buffer.device_ptr();
        let mut n = u32::try_from(element_count)
            .map_err(|_| BackendError::execution("concat element count exceeds u32 range"))?;
        let mut od0 = out_dims[0];
        let mut od1 = out_dims[1];
        let mut od2 = out_dims[2];
        let mut od3 = out_dims[3];
        let mut axis_i32 = axis;
        let mut split_i32 = split;
        let mut ls0 = lhs_strides[0];
        let mut ls1 = lhs_strides[1];
        let mut ls2 = lhs_strides[2];
        let mut ls3 = lhs_strides[3];
        let mut rs0 = rhs_strides[0];
        let mut rs1 = rhs_strides[1];
        let mut rs2 = rhs_strides[2];
        let mut rs3 = rhs_strides[3];
        let mut opaque_ptr = 0u64;
        let mut params = [
            (&mut lhs_ptr as *mut u64).cast::<c_void>(),
            (&mut rhs_ptr as *mut u64).cast::<c_void>(),
            (&mut out_ptr as *mut u64).cast::<c_void>(),
            (&mut n as *mut u32).cast::<c_void>(),
            (&mut od0 as *mut i32).cast::<c_void>(),
            (&mut od1 as *mut i32).cast::<c_void>(),
            (&mut od2 as *mut i32).cast::<c_void>(),
            (&mut od3 as *mut i32).cast::<c_void>(),
            (&mut axis_i32 as *mut i32).cast::<c_void>(),
            (&mut split_i32 as *mut i32).cast::<c_void>(),
            (&mut ls0 as *mut i32).cast::<c_void>(),
            (&mut ls1 as *mut i32).cast::<c_void>(),
            (&mut ls2 as *mut i32).cast::<c_void>(),
            (&mut ls3 as *mut i32).cast::<c_void>(),
            (&mut rs0 as *mut i32).cast::<c_void>(),
            (&mut rs1 as *mut i32).cast::<c_void>(),
            (&mut rs2 as *mut i32).cast::<c_void>(),
            (&mut rs3 as *mut i32).cast::<c_void>(),
            (&mut opaque_ptr as *mut u64).cast::<c_void>(),
        ];
        launch_1d(driver, &loaded, n, 256, &mut params)?;
        Ok(out)
    }

    fn execute_iota(
        &self,
        driver: &Arc<CudaDriver>,
        kernel: &KernelSpec,
        spec: &gpt_rs::backend::spec::IotaSpec,
        out_spec: &TensorSpec,
    ) -> BackendResult<TritonTensor> {
        if out_spec.dtype != DType::Si32 {
            return Err(BackendError::execution(
                "iota runtime currently supports Si32 output only",
            ));
        }
        if !matches!(kernel.kind, KernelKind::IotaSi32Rank4) {
            return Err(BackendError::execution(format!(
                "unexpected iota kernel kind: {:?}",
                kernel.kind
            )));
        }
        let out_dims = static_dims_or_error(&out_spec.shape, |_| {
            BackendError::execution("dynamic dimensions are not supported by triton runtime")
        })?;
        if spec.axis >= out_dims.len() {
            return Err(BackendError::execution("iota axis out of bounds"));
        }
        if out_dims.len() > 4 {
            return Err(BackendError::execution(
                "iota runtime supports rank <= 4 only",
            ));
        }
        let out = TritonTensor::new(out_spec.clone(), driver.alloc_zeroed(byte_len(out_spec)?)?);
        let element_count = static_element_count(&out_spec.shape)?;
        if element_count == 0 {
            return Ok(out);
        }
        let loaded = self.load_kernel(driver, kernel)?;
        let (od0, od1, od2, od3) = align_dims4(&out_dims)?;
        let aligned_axis = (4usize - out_dims.len())
            .checked_add(spec.axis)
            .ok_or_else(|| BackendError::execution("iota axis alignment overflow"))?;
        let axis = i32::try_from(aligned_axis)
            .map_err(|_| BackendError::execution("iota axis exceeds i32 range"))?;

        let mut out_ptr = out.buffer.device_ptr();
        let mut n = u32::try_from(element_count)
            .map_err(|_| BackendError::execution("iota element count exceeds u32 range"))?;
        let mut od0 = od0;
        let mut od1 = od1;
        let mut od2 = od2;
        let mut od3 = od3;
        let mut axis_i32 = axis;
        let mut opaque_ptr = 0u64;
        let mut params = [
            (&mut out_ptr as *mut u64).cast::<c_void>(),
            (&mut n as *mut u32).cast::<c_void>(),
            (&mut od0 as *mut i32).cast::<c_void>(),
            (&mut od1 as *mut i32).cast::<c_void>(),
            (&mut od2 as *mut i32).cast::<c_void>(),
            (&mut od3 as *mut i32).cast::<c_void>(),
            (&mut axis_i32 as *mut i32).cast::<c_void>(),
            (&mut opaque_ptr as *mut u64).cast::<c_void>(),
        ];
        launch_1d(driver, &loaded, n, 256, &mut params)?;
        Ok(out)
    }

    fn execute_compare(
        &self,
        driver: &Arc<CudaDriver>,
        kernel: &KernelSpec,
        op: ComparisonOp,
        lhs: &TritonTensor,
        rhs: &TritonTensor,
        out_spec: &TensorSpec,
    ) -> BackendResult<TritonTensor> {
        if lhs.spec != rhs.spec {
            return Err(BackendError::execution("compare operand spec mismatch"));
        }
        if lhs.spec.dtype != DType::Si32 || out_spec.dtype != DType::I1 {
            return Err(BackendError::execution(
                "compare runtime currently supports Si32 -> I1 only",
            ));
        }
        if !matches!(kernel.kind, KernelKind::CompareSi32I1) {
            return Err(BackendError::execution(format!(
                "unexpected compare kernel kind: {:?}",
                kernel.kind
            )));
        }
        let out = TritonTensor::new(out_spec.clone(), driver.alloc_zeroed(byte_len(out_spec)?)?);
        let element_count = static_element_count(&out_spec.shape)?;
        if element_count == 0 {
            return Ok(out);
        }
        let loaded = self.load_kernel(driver, kernel)?;
        let opcode = compare_opcode(op);
        let mut lhs_ptr = lhs.buffer.device_ptr();
        let mut rhs_ptr = rhs.buffer.device_ptr();
        let mut out_ptr = out.buffer.device_ptr();
        let mut n = u32::try_from(element_count)
            .map_err(|_| BackendError::execution("compare element count exceeds u32 range"))?;
        let mut op_u32 = opcode;
        let mut opaque_ptr = 0u64;
        let mut params = [
            (&mut lhs_ptr as *mut u64).cast::<c_void>(),
            (&mut rhs_ptr as *mut u64).cast::<c_void>(),
            (&mut out_ptr as *mut u64).cast::<c_void>(),
            (&mut n as *mut u32).cast::<c_void>(),
            (&mut op_u32 as *mut u32).cast::<c_void>(),
            (&mut opaque_ptr as *mut u64).cast::<c_void>(),
        ];
        launch_1d(driver, &loaded, n, 256, &mut params)?;
        Ok(out)
    }

    fn execute_select(
        &self,
        driver: &Arc<CudaDriver>,
        kernel: &KernelSpec,
        predicate: &TritonTensor,
        when_true: &TritonTensor,
        when_false: &TritonTensor,
        out_spec: &TensorSpec,
    ) -> BackendResult<TritonTensor> {
        if predicate.spec.dtype != DType::I1 {
            return Err(BackendError::execution("select predicate must be I1"));
        }
        if when_true.spec != when_false.spec || when_true.spec != *out_spec {
            return Err(BackendError::execution(
                "select branch/output spec mismatch",
            ));
        }
        if out_spec.dtype != DType::F32 {
            return Err(BackendError::execution(
                "select runtime currently supports F32 branches only",
            ));
        }
        if !matches!(kernel.kind, KernelKind::SelectI1F32) {
            return Err(BackendError::execution(format!(
                "unexpected select kernel kind: {:?}",
                kernel.kind
            )));
        }
        let out = TritonTensor::new(out_spec.clone(), driver.alloc_zeroed(byte_len(out_spec)?)?);
        let element_count = static_element_count(&out_spec.shape)?;
        if element_count == 0 {
            return Ok(out);
        }
        let loaded = self.load_kernel(driver, kernel)?;
        let mut pred_ptr = predicate.buffer.device_ptr();
        let mut true_ptr = when_true.buffer.device_ptr();
        let mut false_ptr = when_false.buffer.device_ptr();
        let mut out_ptr = out.buffer.device_ptr();
        let mut n = u32::try_from(element_count)
            .map_err(|_| BackendError::execution("select element count exceeds u32 range"))?;
        let mut opaque_ptr = 0u64;
        let mut params = [
            (&mut pred_ptr as *mut u64).cast::<c_void>(),
            (&mut true_ptr as *mut u64).cast::<c_void>(),
            (&mut false_ptr as *mut u64).cast::<c_void>(),
            (&mut out_ptr as *mut u64).cast::<c_void>(),
            (&mut n as *mut u32).cast::<c_void>(),
            (&mut opaque_ptr as *mut u64).cast::<c_void>(),
        ];
        launch_1d(driver, &loaded, n, 256, &mut params)?;
        Ok(out)
    }

    fn execute_take(
        &self,
        driver: &Arc<CudaDriver>,
        kernel: &KernelSpec,
        params: &TritonTensor,
        indices: &TritonTensor,
        out_spec: &TensorSpec,
    ) -> BackendResult<TritonTensor> {
        if params.spec.dtype != DType::F32
            || indices.spec.dtype != DType::Si32
            || out_spec.dtype != DType::F32
        {
            return Err(BackendError::execution(
                "take runtime expects F32 params, Si32 indices, F32 output",
            ));
        }
        if !matches!(kernel.kind, KernelKind::TakeF32I32) {
            return Err(BackendError::execution(format!(
                "unexpected take kernel kind: {:?}",
                kernel.kind
            )));
        }
        let params_dims = static_dims_or_error(&params.spec.shape, |_| {
            BackendError::execution("dynamic dimensions are not supported by triton runtime")
        })?;
        if params_dims.len() != 2 {
            return Err(BackendError::execution(
                "take runtime currently supports rank-2 params only",
            ));
        }
        let index_count = static_element_count(&indices.spec.shape)?;
        let embed_dim = params_dims[1];
        let vocab = params_dims[0];
        let expected_out = index_count
            .checked_mul(embed_dim)
            .ok_or_else(|| BackendError::execution("take output element count overflow"))?;
        let out_count = static_element_count(&out_spec.shape)?;
        if expected_out != out_count {
            return Err(BackendError::execution(format!(
                "take output shape mismatch: expected {} elements, got {}",
                expected_out, out_count
            )));
        }

        let out = TritonTensor::new(out_spec.clone(), driver.alloc_zeroed(byte_len(out_spec)?)?);
        if out_count == 0 {
            return Ok(out);
        }
        let loaded = self.load_kernel(driver, kernel)?;
        let mut weight_ptr = params.buffer.device_ptr();
        let mut indices_ptr = indices.buffer.device_ptr();
        let mut out_ptr = out.buffer.device_ptr();
        let mut n = u32::try_from(out_count)
            .map_err(|_| BackendError::execution("take element count exceeds u32 range"))?;
        let mut embed = i32::try_from(embed_dim)
            .map_err(|_| BackendError::execution("take embed_dim exceeds i32 range"))?;
        let mut vocab = i32::try_from(vocab)
            .map_err(|_| BackendError::execution("take vocab exceeds i32 range"))?;
        let mut opaque_ptr = 0u64;
        let mut params_arr = [
            (&mut weight_ptr as *mut u64).cast::<c_void>(),
            (&mut indices_ptr as *mut u64).cast::<c_void>(),
            (&mut out_ptr as *mut u64).cast::<c_void>(),
            (&mut n as *mut u32).cast::<c_void>(),
            (&mut embed as *mut i32).cast::<c_void>(),
            (&mut vocab as *mut i32).cast::<c_void>(),
            (&mut opaque_ptr as *mut u64).cast::<c_void>(),
        ];
        launch_1d(driver, &loaded, n, 256, &mut params_arr)?;
        Ok(out)
    }

    fn execute_dynamic_update_slice(
        &self,
        driver: &Arc<CudaDriver>,
        kernel: &KernelSpec,
        base: &TritonTensor,
        update: &TritonTensor,
        starts: &TritonTensor,
        out_spec: &TensorSpec,
    ) -> BackendResult<TritonTensor> {
        if base.spec.dtype != DType::F32
            || update.spec.dtype != DType::F32
            || starts.spec.dtype != DType::Si32
            || out_spec.dtype != DType::F32
        {
            return Err(BackendError::execution(
                "dynamic_update_slice runtime currently supports F32 base/update with Si32 starts",
            ));
        }
        if !matches!(kernel.kind, KernelKind::DynamicUpdateSliceF32Rank4) {
            return Err(BackendError::execution(format!(
                "unexpected dynamic_update_slice kernel kind: {:?}",
                kernel.kind
            )));
        }
        let base_dims = static_dims_or_error(&base.spec.shape, |_| {
            BackendError::execution("dynamic dimensions are not supported by triton runtime")
        })?;
        let update_dims = static_dims_or_error(&update.spec.shape, |_| {
            BackendError::execution("dynamic dimensions are not supported by triton runtime")
        })?;
        if base_dims.len() > 4 {
            return Err(BackendError::execution(
                "dynamic_update_slice runtime supports rank <= 4 only",
            ));
        }
        let starts_values = read_i32_tensor(starts)?;
        if starts_values.len() != base_dims.len() {
            return Err(BackendError::execution(
                "dynamic_update_slice starts length mismatch",
            ));
        }
        let mut starts_usize = Vec::with_capacity(starts_values.len());
        for value in starts_values {
            if value < 0 {
                return Err(BackendError::execution(
                    "dynamic_update_slice starts must be non-negative",
                ));
            }
            starts_usize.push(value as usize);
        }
        for axis in 0..base_dims.len() {
            let end = starts_usize[axis]
                .checked_add(update_dims[axis])
                .ok_or_else(|| BackendError::execution("dynamic_update_slice bounds overflow"))?;
            if end > base_dims[axis] {
                return Err(BackendError::execution(
                    "dynamic_update_slice update exceeds base bounds",
                ));
            }
        }

        let out = TritonTensor::new(out_spec.clone(), driver.alloc_zeroed(byte_len(out_spec)?)?);
        driver.copy_device_to_device(
            out.buffer.device_ptr(),
            base.buffer.device_ptr(),
            byte_len(out_spec)?,
        )?;
        let update_elems = static_element_count(&update.spec.shape)?;
        if update_elems == 0 {
            return Ok(out);
        }
        let loaded = self.load_kernel(driver, kernel)?;
        let (update_dims4, out_strides4, starts4) =
            dynamic_update_rank4_layout(&update_dims, &base_dims, &starts_usize)?;

        let mut update_ptr = update.buffer.device_ptr();
        let mut out_ptr = out.buffer.device_ptr();
        let mut n = u32::try_from(update_elems).map_err(|_| {
            BackendError::execution("dynamic_update_slice update elements exceeds u32 range")
        })?;
        let mut ud0 = update_dims4[0];
        let mut ud1 = update_dims4[1];
        let mut ud2 = update_dims4[2];
        let mut ud3 = update_dims4[3];
        let mut os0 = out_strides4[0];
        let mut os1 = out_strides4[1];
        let mut os2 = out_strides4[2];
        let mut os3 = out_strides4[3];
        let mut st0 = starts4[0];
        let mut st1 = starts4[1];
        let mut st2 = starts4[2];
        let mut st3 = starts4[3];
        let mut opaque_ptr = 0u64;
        let mut params = [
            (&mut update_ptr as *mut u64).cast::<c_void>(),
            (&mut out_ptr as *mut u64).cast::<c_void>(),
            (&mut n as *mut u32).cast::<c_void>(),
            (&mut ud0 as *mut i32).cast::<c_void>(),
            (&mut ud1 as *mut i32).cast::<c_void>(),
            (&mut ud2 as *mut i32).cast::<c_void>(),
            (&mut ud3 as *mut i32).cast::<c_void>(),
            (&mut os0 as *mut i32).cast::<c_void>(),
            (&mut os1 as *mut i32).cast::<c_void>(),
            (&mut os2 as *mut i32).cast::<c_void>(),
            (&mut os3 as *mut i32).cast::<c_void>(),
            (&mut st0 as *mut i32).cast::<c_void>(),
            (&mut st1 as *mut i32).cast::<c_void>(),
            (&mut st2 as *mut i32).cast::<c_void>(),
            (&mut st3 as *mut i32).cast::<c_void>(),
            (&mut opaque_ptr as *mut u64).cast::<c_void>(),
        ];
        launch_1d(driver, &loaded, n, 256, &mut params)?;
        Ok(out)
    }

    fn execute_reduce(
        &self,
        driver: &Arc<CudaDriver>,
        kernels: &HashMap<&str, &KernelSpec>,
        input_spec: &TensorSpec,
        out_spec: &TensorSpec,
        spec: &gpt_rs::backend::spec::ReduceSpec,
        input: &TritonTensor,
    ) -> BackendResult<TritonTensor> {
        match spec.kind {
            ReduceKind::Sum => {
                let kernel = kernels.get(REDUCE_SUM_LAST_AXIS_KERNEL_ID).ok_or_else(|| {
                    BackendError::execution(
                        "missing reduce_sum_last_axis kernel in triton artifact",
                    )
                })?;
                self.execute_reduce_sum_last_axis(driver, kernel, input_spec, out_spec, spec, input)
            }
            ReduceKind::Max => {
                let kernel = kernels.get(REDUCE_MAX_LAST_AXIS_KERNEL_ID).ok_or_else(|| {
                    BackendError::execution(
                        "missing reduce_max_last_axis kernel in triton artifact",
                    )
                })?;
                self.execute_reduce_max_last_axis(driver, kernel, input_spec, out_spec, spec, input)
            }
            other => Err(BackendError::execution(format!(
                "reduce runtime unsupported kind {:?}",
                other
            ))),
        }
    }

    fn execute_reduce_max_last_axis(
        &self,
        driver: &Arc<CudaDriver>,
        kernel: &KernelSpec,
        input_spec: &TensorSpec,
        out_spec: &TensorSpec,
        spec: &gpt_rs::backend::spec::ReduceSpec,
        input: &TritonTensor,
    ) -> BackendResult<TritonTensor> {
        if !matches!(kernel.kind, KernelKind::ReduceMaxLastAxisF32) {
            return Err(BackendError::execution(format!(
                "unexpected reduce_max kernel kind: {:?}",
                kernel.kind
            )));
        }
        if input.spec != *input_spec {
            return Err(BackendError::execution("reduce_max tensor/spec mismatch"));
        }
        if input_spec.dtype != DType::F32 || out_spec.dtype != DType::F32 {
            return Err(BackendError::execution(
                "reduce_max currently supports F32 only",
            ));
        }
        let input_dims = static_dims_or_error(&input_spec.shape, |_| {
            BackendError::execution("dynamic dimensions are not supported by triton runtime")
        })?;
        if input_dims.is_empty() {
            return Err(BackendError::execution(
                "reduce_max does not support scalar inputs",
            ));
        }
        let last_axis = input_dims.len() - 1;
        if spec.axes.as_slice() != [last_axis] {
            return Err(BackendError::execution(
                "reduce_max supports reducing the last axis only",
            ));
        }
        let rows = input_dims[..last_axis]
            .iter()
            .try_fold(1usize, |acc: usize, dim| acc.checked_mul(*dim))
            .ok_or_else(|| BackendError::execution("reduce_max row dimension overflow"))?;
        let cols = input_dims[last_axis];
        let out = TritonTensor::new(out_spec.clone(), driver.alloc_zeroed(byte_len(out_spec)?)?);
        if rows == 0 || cols == 0 {
            return Ok(out);
        }
        let loaded = self.load_kernel(driver, kernel)?;
        let mut in_ptr = input.buffer.device_ptr();
        let mut out_ptr = out.buffer.device_ptr();
        let mut rows_i32 = i32::try_from(rows)
            .map_err(|_| BackendError::execution("reduce_max rows exceeds i32 range"))?;
        let mut cols_i32 = i32::try_from(cols)
            .map_err(|_| BackendError::execution("reduce_max cols exceeds i32 range"))?;
        let mut opaque_ptr = 0u64;
        let mut params = [
            (&mut in_ptr as *mut u64).cast::<c_void>(),
            (&mut out_ptr as *mut u64).cast::<c_void>(),
            (&mut rows_i32 as *mut i32).cast::<c_void>(),
            (&mut cols_i32 as *mut i32).cast::<c_void>(),
            (&mut opaque_ptr as *mut u64).cast::<c_void>(),
        ];
        let rows_u32 = u32::try_from(rows)
            .map_err(|_| BackendError::execution("reduce_max rows exceeds u32 range"))?;
        launch_program_grid(driver, &loaded, rows_u32, 256, rows_u32, &mut params)?;
        Ok(out)
    }

    fn execute_extract_patches(
        &self,
        driver: &Arc<CudaDriver>,
        kernel: &KernelSpec,
        input: &TritonTensor,
        spec: &gpt_rs::backend::spec::ExtractPatchesSpec,
        out_spec: &TensorSpec,
    ) -> BackendResult<TritonTensor> {
        if !matches!(kernel.kind, KernelKind::ExtractPatchesNhwcF32) {
            return Err(BackendError::execution(format!(
                "unexpected extract_patches kernel kind: {:?}",
                kernel.kind
            )));
        }
        if input.spec.dtype != DType::F32 || out_spec.dtype != DType::F32 {
            return Err(BackendError::execution(
                "extract_patches runtime currently supports F32 only",
            ));
        }
        let in_dims = static_dims_or_error(&input.spec.shape, |_| {
            BackendError::execution("dynamic dimensions are not supported by triton runtime")
        })?;
        let out_dims = static_dims_or_error(&out_spec.shape, |_| {
            BackendError::execution("dynamic dimensions are not supported by triton runtime")
        })?;
        if in_dims.len() != 4 || out_dims.len() != 4 {
            return Err(BackendError::execution(
                "extract_patches runtime expects rank-4 NHWC input/output",
            ));
        }
        if spec.window.len() != 2
            || spec.strides.len() != 2
            || spec.dilation.len() != 2
            || spec.padding.len() != 2
        {
            return Err(BackendError::execution(
                "extract_patches runtime currently supports 2D window only",
            ));
        }
        let out = TritonTensor::new(out_spec.clone(), driver.alloc_zeroed(byte_len(out_spec)?)?);
        let elem_count = static_element_count(&out_spec.shape)?;
        if elem_count == 0 {
            return Ok(out);
        }
        let patch_dim = out_dims[3];
        let loaded = self.load_kernel(driver, kernel)?;

        let mut in_ptr = input.buffer.device_ptr();
        let mut out_ptr = out.buffer.device_ptr();
        let mut n = u32::try_from(elem_count).map_err(|_| {
            BackendError::execution("extract_patches element count exceeds u32 range")
        })?;
        let mut in_h = i32::try_from(in_dims[1])
            .map_err(|_| BackendError::execution("extract_patches in_h exceeds i32 range"))?;
        let mut in_w = i32::try_from(in_dims[2])
            .map_err(|_| BackendError::execution("extract_patches in_w exceeds i32 range"))?;
        let mut in_c = i32::try_from(in_dims[3])
            .map_err(|_| BackendError::execution("extract_patches in_c exceeds i32 range"))?;
        let mut out_h = i32::try_from(out_dims[1])
            .map_err(|_| BackendError::execution("extract_patches out_h exceeds i32 range"))?;
        let mut out_w = i32::try_from(out_dims[2])
            .map_err(|_| BackendError::execution("extract_patches out_w exceeds i32 range"))?;
        let mut k_h = i32::try_from(spec.window[0])
            .map_err(|_| BackendError::execution("extract_patches k_h exceeds i32 range"))?;
        let mut k_w = i32::try_from(spec.window[1])
            .map_err(|_| BackendError::execution("extract_patches k_w exceeds i32 range"))?;
        let mut s_h = i32::try_from(spec.strides[0])
            .map_err(|_| BackendError::execution("extract_patches s_h exceeds i32 range"))?;
        let mut s_w = i32::try_from(spec.strides[1])
            .map_err(|_| BackendError::execution("extract_patches s_w exceeds i32 range"))?;
        let mut d_h = i32::try_from(spec.dilation[0])
            .map_err(|_| BackendError::execution("extract_patches d_h exceeds i32 range"))?;
        let mut d_w = i32::try_from(spec.dilation[1])
            .map_err(|_| BackendError::execution("extract_patches d_w exceeds i32 range"))?;
        let mut pad_top = i32::try_from(spec.padding[0].0)
            .map_err(|_| BackendError::execution("extract_patches pad_top exceeds i32 range"))?;
        let mut pad_left = i32::try_from(spec.padding[1].0)
            .map_err(|_| BackendError::execution("extract_patches pad_left exceeds i32 range"))?;
        let mut patch_dim_i32 = i32::try_from(patch_dim)
            .map_err(|_| BackendError::execution("extract_patches patch_dim exceeds i32 range"))?;
        let mut opaque_ptr = 0u64;
        let mut params = [
            (&mut in_ptr as *mut u64).cast::<c_void>(),
            (&mut out_ptr as *mut u64).cast::<c_void>(),
            (&mut n as *mut u32).cast::<c_void>(),
            (&mut in_h as *mut i32).cast::<c_void>(),
            (&mut in_w as *mut i32).cast::<c_void>(),
            (&mut in_c as *mut i32).cast::<c_void>(),
            (&mut out_h as *mut i32).cast::<c_void>(),
            (&mut out_w as *mut i32).cast::<c_void>(),
            (&mut k_h as *mut i32).cast::<c_void>(),
            (&mut k_w as *mut i32).cast::<c_void>(),
            (&mut s_h as *mut i32).cast::<c_void>(),
            (&mut s_w as *mut i32).cast::<c_void>(),
            (&mut d_h as *mut i32).cast::<c_void>(),
            (&mut d_w as *mut i32).cast::<c_void>(),
            (&mut pad_top as *mut i32).cast::<c_void>(),
            (&mut pad_left as *mut i32).cast::<c_void>(),
            (&mut patch_dim_i32 as *mut i32).cast::<c_void>(),
            (&mut opaque_ptr as *mut u64).cast::<c_void>(),
        ];
        launch_1d(driver, &loaded, n, 256, &mut params)?;
        Ok(out)
    }

    fn execute_reduce_window(
        &self,
        driver: &Arc<CudaDriver>,
        kernel: &KernelSpec,
        input: &TritonTensor,
        spec: &gpt_rs::backend::spec::ReduceWindowSpec,
        out_spec: &TensorSpec,
    ) -> BackendResult<TritonTensor> {
        if !matches!(kernel.kind, KernelKind::ReduceWindowMaxNhwcF32) {
            return Err(BackendError::execution(format!(
                "unexpected reduce_window kernel kind: {:?}",
                kernel.kind
            )));
        }
        if spec.reduce != ReduceKind::Max {
            return Err(BackendError::execution(
                "reduce_window runtime currently supports max only",
            ));
        }
        if input.spec.dtype != DType::F32 || out_spec.dtype != DType::F32 {
            return Err(BackendError::execution(
                "reduce_window runtime currently supports F32 only",
            ));
        }
        let in_dims = static_dims_or_error(&input.spec.shape, |_| {
            BackendError::execution("dynamic dimensions are not supported by triton runtime")
        })?;
        let out_dims = static_dims_or_error(&out_spec.shape, |_| {
            BackendError::execution("dynamic dimensions are not supported by triton runtime")
        })?;
        if in_dims.len() != 4 || out_dims.len() != 4 {
            return Err(BackendError::execution(
                "reduce_window runtime expects rank-4 NHWC input/output",
            ));
        }
        if spec.window_dims.len() != 4
            || spec.strides.len() != 4
            || spec.padding.len() != 4
            || spec.base_dilation.len() != 4
            || spec.window_dilation.len() != 4
        {
            return Err(BackendError::execution(
                "reduce_window runtime currently supports rank-4 specs only",
            ));
        }
        if spec.window_dims[0] != 1 || spec.window_dims[3] != 1 {
            return Err(BackendError::execution(
                "reduce_window runtime expects NHWC window with N/C window = 1",
            ));
        }
        if spec.base_dilation != vec![1, 1, 1, 1] {
            return Err(BackendError::execution(
                "reduce_window runtime currently supports base_dilation=[1,1,1,1] only",
            ));
        }

        let out = TritonTensor::new(out_spec.clone(), driver.alloc_zeroed(byte_len(out_spec)?)?);
        let elem_count = static_element_count(&out_spec.shape)?;
        if elem_count == 0 {
            return Ok(out);
        }
        let loaded = self.load_kernel(driver, kernel)?;

        let mut in_ptr = input.buffer.device_ptr();
        let mut out_ptr = out.buffer.device_ptr();
        let mut n = u32::try_from(elem_count).map_err(|_| {
            BackendError::execution("reduce_window element count exceeds u32 range")
        })?;
        let mut in_h = i32::try_from(in_dims[1])
            .map_err(|_| BackendError::execution("reduce_window in_h exceeds i32 range"))?;
        let mut in_w = i32::try_from(in_dims[2])
            .map_err(|_| BackendError::execution("reduce_window in_w exceeds i32 range"))?;
        let mut in_c = i32::try_from(in_dims[3])
            .map_err(|_| BackendError::execution("reduce_window in_c exceeds i32 range"))?;
        let mut out_h = i32::try_from(out_dims[1])
            .map_err(|_| BackendError::execution("reduce_window out_h exceeds i32 range"))?;
        let mut out_w = i32::try_from(out_dims[2])
            .map_err(|_| BackendError::execution("reduce_window out_w exceeds i32 range"))?;
        let mut k_h = i32::try_from(spec.window_dims[1])
            .map_err(|_| BackendError::execution("reduce_window k_h exceeds i32 range"))?;
        let mut k_w = i32::try_from(spec.window_dims[2])
            .map_err(|_| BackendError::execution("reduce_window k_w exceeds i32 range"))?;
        let mut s_h = i32::try_from(spec.strides[1])
            .map_err(|_| BackendError::execution("reduce_window s_h exceeds i32 range"))?;
        let mut s_w = i32::try_from(spec.strides[2])
            .map_err(|_| BackendError::execution("reduce_window s_w exceeds i32 range"))?;
        let mut d_h = i32::try_from(spec.window_dilation[1])
            .map_err(|_| BackendError::execution("reduce_window d_h exceeds i32 range"))?;
        let mut d_w = i32::try_from(spec.window_dilation[2])
            .map_err(|_| BackendError::execution("reduce_window d_w exceeds i32 range"))?;
        let mut pad_top = i32::try_from(spec.padding[1].0)
            .map_err(|_| BackendError::execution("reduce_window pad_top exceeds i32 range"))?;
        let mut pad_left = i32::try_from(spec.padding[2].0)
            .map_err(|_| BackendError::execution("reduce_window pad_left exceeds i32 range"))?;
        let mut opaque_ptr = 0u64;
        let mut params = [
            (&mut in_ptr as *mut u64).cast::<c_void>(),
            (&mut out_ptr as *mut u64).cast::<c_void>(),
            (&mut n as *mut u32).cast::<c_void>(),
            (&mut in_h as *mut i32).cast::<c_void>(),
            (&mut in_w as *mut i32).cast::<c_void>(),
            (&mut in_c as *mut i32).cast::<c_void>(),
            (&mut out_h as *mut i32).cast::<c_void>(),
            (&mut out_w as *mut i32).cast::<c_void>(),
            (&mut k_h as *mut i32).cast::<c_void>(),
            (&mut k_w as *mut i32).cast::<c_void>(),
            (&mut s_h as *mut i32).cast::<c_void>(),
            (&mut s_w as *mut i32).cast::<c_void>(),
            (&mut d_h as *mut i32).cast::<c_void>(),
            (&mut d_w as *mut i32).cast::<c_void>(),
            (&mut pad_top as *mut i32).cast::<c_void>(),
            (&mut pad_left as *mut i32).cast::<c_void>(),
            (&mut opaque_ptr as *mut u64).cast::<c_void>(),
        ];
        launch_1d(driver, &loaded, n, 128, &mut params)?;
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

        let lhs_dims = static_dims_or_error(&lhs_spec.shape, |_| {
            BackendError::execution("dynamic dimensions are not supported by triton runtime")
        })?;
        let rhs_dims = static_dims_or_error(&rhs_spec.shape, |_| {
            BackendError::execution("dynamic dimensions are not supported by triton runtime")
        })?;
        let out_dims = static_dims_or_error(&out_spec.shape, |_| {
            BackendError::execution("dynamic dimensions are not supported by triton runtime")
        })?;
        let cublas = self.cublas(driver)?;

        // Rank-2 matrix multiplication: [M,K] · [K,N] => [M,N].
        if spec.batch_lhs.is_empty()
            && spec.batch_rhs.is_empty()
            && spec.contract_lhs.as_slice() == [1]
            && spec.contract_rhs.as_slice() == [0]
        {
            if lhs_dims.len() != 2 || rhs_dims.len() != 2 || out_dims.len() != 2 {
                return Err(BackendError::execution(
                    "dot_general rank-2 path expects rank-2 tensors",
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

            let out =
                TritonTensor::new(out_spec.clone(), driver.alloc_zeroed(byte_len(out_spec)?)?);
            cublas.sgemm_row_major(&lhs.buffer, &rhs.buffer, &out.buffer, m, n, k)?;
            return Ok(out);
        }

        // Batched rank-3 matrix multiplication: [B,M,K] · [B,K,N] => [B,M,N].
        if spec.batch_lhs.as_slice() == [0]
            && spec.batch_rhs.as_slice() == [0]
            && spec.contract_lhs.as_slice() == [2]
            && spec.contract_rhs.as_slice() == [1]
        {
            if lhs_dims.len() != 3 || rhs_dims.len() != 3 || out_dims.len() != 3 {
                return Err(BackendError::execution(
                    "dot_general batched path expects rank-3 tensors",
                ));
            }

            let batches = lhs_dims[0];
            let m = lhs_dims[1];
            let k = lhs_dims[2];
            let rhs_batches = rhs_dims[0];
            let k_rhs = rhs_dims[1];
            let n = rhs_dims[2];

            if batches != rhs_batches
                || out_dims[0] != batches
                || out_dims[1] != m
                || out_dims[2] != n
                || k != k_rhs
            {
                return Err(BackendError::execution(
                    "dot_general shape mismatch for batched matrix multiplication",
                ));
            }

            let out =
                TritonTensor::new(out_spec.clone(), driver.alloc_zeroed(byte_len(out_spec)?)?);
            let lhs_stride = m
                .checked_mul(k)
                .ok_or_else(|| BackendError::execution("batched lhs stride overflow"))?;
            let rhs_stride = k
                .checked_mul(n)
                .ok_or_else(|| BackendError::execution("batched rhs stride overflow"))?;
            let out_stride = m
                .checked_mul(n)
                .ok_or_else(|| BackendError::execution("batched out stride overflow"))?;
            let cfg = StridedBatchedGemmConfig {
                m,
                n,
                k,
                lhs_stride,
                rhs_stride,
                out_stride,
                batches,
            };
            cublas.sgemm_row_major_strided_batched(&lhs.buffer, &rhs.buffer, &out.buffer, cfg)?;
            return Ok(out);
        }

        // Batched rank-3 matrix multiplication: [B,M,K] · [B,N,K] => [B,M,N].
        if spec.batch_lhs.as_slice() == [0]
            && spec.batch_rhs.as_slice() == [0]
            && spec.contract_lhs.as_slice() == [2]
            && spec.contract_rhs.as_slice() == [2]
        {
            if lhs_dims.len() != 3 || rhs_dims.len() != 3 || out_dims.len() != 3 {
                return Err(BackendError::execution(
                    "dot_general batched path expects rank-3 tensors",
                ));
            }

            let batches = lhs_dims[0];
            let m = lhs_dims[1];
            let k = lhs_dims[2];
            let rhs_batches = rhs_dims[0];
            let n = rhs_dims[1];
            let k_rhs = rhs_dims[2];

            if batches != rhs_batches
                || out_dims[0] != batches
                || out_dims[1] != m
                || out_dims[2] != n
                || k != k_rhs
            {
                return Err(BackendError::execution(
                    "dot_general shape mismatch for batched matrix multiplication [B,M,K] · [B,N,K]",
                ));
            }

            let out =
                TritonTensor::new(out_spec.clone(), driver.alloc_zeroed(byte_len(out_spec)?)?);
            let lhs_stride = m
                .checked_mul(k)
                .ok_or_else(|| BackendError::execution("batched lhs stride overflow"))?;
            let rhs_stride = n
                .checked_mul(k)
                .ok_or_else(|| BackendError::execution("batched rhs stride overflow"))?;
            let out_stride = m
                .checked_mul(n)
                .ok_or_else(|| BackendError::execution("batched out stride overflow"))?;
            let cfg = StridedBatchedGemmConfig {
                m,
                n,
                k,
                lhs_stride,
                rhs_stride,
                out_stride,
                batches,
            };
            cublas.sgemm_row_major_strided_batched_rhs_transposed(
                &lhs.buffer,
                &rhs.buffer,
                &out.buffer,
                cfg,
            )?;
            return Ok(out);
        }

        Err(BackendError::execution(
            "dot_general runtime supports rank-2 MxK·KxN and selected rank-3 batched variants",
        ))
    }

    fn execute_reduce_sum_last_axis(
        &self,
        driver: &Arc<CudaDriver>,
        kernel: &KernelSpec,
        input_spec: &TensorSpec,
        out_spec: &TensorSpec,
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
        if !matches!(kernel.kind, KernelKind::ReduceSumLastAxisF32) {
            return Err(BackendError::execution(format!(
                "unexpected reduce_sum kernel kind: {:?}",
                kernel.kind
            )));
        }

        let input_dims = static_dims_or_error(&input_spec.shape, |_| {
            BackendError::execution("dynamic dimensions are not supported by triton runtime")
        })?;
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
            .try_fold(1usize, |acc: usize, dim| acc.checked_mul(*dim))
            .ok_or_else(|| BackendError::execution("reduce row dimension overflow"))?;
        let cols = input_dims[last_axis];

        let out = TritonTensor::new(out_spec.clone(), driver.alloc_zeroed(byte_len(out_spec)?)?);
        if rows == 0 || cols == 0 {
            return Ok(out);
        }
        let loaded = self.load_kernel(driver, kernel)?;
        let mut in_ptr = input.buffer.device_ptr();
        let mut out_ptr = out.buffer.device_ptr();
        let mut rows_i32 = i32::try_from(rows)
            .map_err(|_| BackendError::execution("reduce_sum rows exceeds i32 range"))?;
        let mut cols_i32 = i32::try_from(cols)
            .map_err(|_| BackendError::execution("reduce_sum cols exceeds i32 range"))?;
        let mut opaque_ptr = 0u64;
        let mut params = [
            (&mut in_ptr as *mut u64).cast::<c_void>(),
            (&mut out_ptr as *mut u64).cast::<c_void>(),
            (&mut rows_i32 as *mut i32).cast::<c_void>(),
            (&mut cols_i32 as *mut i32).cast::<c_void>(),
            (&mut opaque_ptr as *mut u64).cast::<c_void>(),
        ];
        let rows_u32 = u32::try_from(rows)
            .map_err(|_| BackendError::execution("reduce_sum rows exceeds u32 range"))?;
        launch_program_grid(driver, &loaded, rows_u32, 256, rows_u32, &mut params)?;
        Ok(out)
    }

    fn load_kernel(
        &self,
        driver: &Arc<CudaDriver>,
        spec: &KernelSpec,
    ) -> BackendResult<Arc<LoadedKernel>> {
        if let Some(found) = EXECUTION_KERNEL_STACK.with(|stack| {
            stack
                .borrow()
                .last()
                .and_then(|kernels| kernels.get(spec.id.as_str()).cloned())
        }) {
            return Ok(found);
        }

        let compiled = self.compiler.compile(spec)?;
        if let Some(found) = self
            .loaded_kernels
            .lock()
            .expect("triton loaded kernel cache poisoned")
            .get(&compiled.fingerprint)
            .cloned()
        {
            profiling::cache_event("triton_backend.module_hit_mem");
            return Ok(found);
        }
        profiling::cache_event("triton_backend.module_miss_mem");

        let function = {
            let _load_scope = profiling::compile_scope("triton_backend.dlopen");
            let module = driver.load_ptx_module(compiled.ptx.as_ref())?;
            driver.get_function(&module, compiled.symbol.as_ref())?
        };
        let loaded = Arc::new(LoadedKernel {
            fingerprint: compiled.fingerprint,
            function,
            profile_signature: profiling::signature_id(&spec.id),
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
}

fn first_missing_operand_value(
    instruction: &gpt_rs::backend::spec::Instruction,
    values: &HashMap<ValueId, TritonTensor>,
) -> Option<ValueId> {
    for operand in &instruction.operands {
        match operand {
            Operand::Value(id) if !values.contains_key(id) => return Some(*id),
            Operand::TupleElement { tuple, .. } if !values.contains_key(tuple) => {
                return Some(*tuple)
            }
            Operand::Literal(_) | Operand::Value(_) | Operand::TupleElement { .. } => {}
        }
    }
    None
}

struct LoadedKernel {
    #[allow(dead_code)]
    fingerprint: u64,
    function: CudaFunction,
    profile_signature: Option<u32>,
}

struct ExecutionKernelGuard;

impl ExecutionKernelGuard {
    fn push(kernels: HashMap<String, Arc<LoadedKernel>>) -> Self {
        EXECUTION_KERNEL_STACK.with(|stack| stack.borrow_mut().push(kernels));
        Self
    }
}

impl Drop for ExecutionKernelGuard {
    fn drop(&mut self) {
        EXECUTION_KERNEL_STACK.with(|stack| {
            let _ = stack.borrow_mut().pop();
        });
    }
}

fn literal_to_tensor(
    driver: &Arc<CudaDriver>,
    literal: &TensorLiteral,
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

    let buffer = driver.alloc_and_upload(literal.bytes.as_ref())?;
    Ok(TritonTensor::new(literal.spec.clone(), buffer))
}

fn output_tensor_spec(output: &ValueType) -> BackendResult<TensorSpec> {
    match output {
        ValueType::Tensor(spec) => Ok(spec.clone()),
        ValueType::Tuple(_) => Err(BackendError::execution(
            "tuple outputs are not supported by triton runtime",
        )),
    }
}

fn byte_len(spec: &TensorSpec) -> BackendResult<usize> {
    spec.byte_len().ok_or_else(|| {
        BackendError::execution(format!(
            "cannot compute byte length for dtype {:?} and shape {:?}",
            spec.dtype,
            spec.shape.dims()
        ))
    })
}

fn static_element_count(shape: &gpt_rs::backend::spec::Shape) -> BackendResult<usize> {
    let dims = static_dims_or_error(shape, |_| {
        BackendError::execution("dynamic dimensions are not supported by triton runtime")
    })?;
    checked_element_count_or_error(dims.as_slice(), || {
        BackendError::execution("element count overflow")
    })
}

fn ensure_static_broadcastable(
    out_shape: &gpt_rs::backend::spec::Shape,
    in_shape: &gpt_rs::backend::spec::Shape,
    context: &str,
) -> BackendResult<()> {
    let out_dims = static_dims_or_error(out_shape, |_| {
        BackendError::execution("dynamic dimensions are not supported by triton runtime")
    })?;
    let in_dims = static_dims_or_error(in_shape, |_| {
        BackendError::execution("dynamic dimensions are not supported by triton runtime")
    })?;
    if in_dims.len() > out_dims.len() {
        return Err(BackendError::execution(format!(
            "{context} rank mismatch: input rank {} exceeds output rank {}",
            in_dims.len(),
            out_dims.len()
        )));
    }
    let offset = out_dims.len() - in_dims.len();
    for (idx, in_dim) in in_dims.iter().enumerate() {
        let out_dim = out_dims[offset + idx];
        if *in_dim != 1 && *in_dim != out_dim {
            return Err(BackendError::execution(format!(
                "{context} shape mismatch: input dim {} (axis {}) is incompatible with output dim {}",
                in_dim,
                idx,
                out_dim
            )));
        }
    }
    Ok(())
}

fn align_dims4(dims: &[usize]) -> BackendResult<(i32, i32, i32, i32)> {
    if dims.len() > 4 {
        return Err(BackendError::execution(format!(
            "rank {} exceeds rank-4 support",
            dims.len()
        )));
    }
    let mut aligned = [1usize; 4];
    let offset = 4 - dims.len();
    for (idx, value) in dims.iter().enumerate() {
        aligned[offset + idx] = *value;
    }
    Ok((
        i32::try_from(aligned[0]).map_err(|_| BackendError::execution("dim0 exceeds i32 range"))?,
        i32::try_from(aligned[1]).map_err(|_| BackendError::execution("dim1 exceeds i32 range"))?,
        i32::try_from(aligned[2]).map_err(|_| BackendError::execution("dim2 exceeds i32 range"))?,
        i32::try_from(aligned[3]).map_err(|_| BackendError::execution("dim3 exceeds i32 range"))?,
    ))
}

fn broadcast_rank4_layout(
    in_shape: &gpt_rs::backend::spec::Shape,
    out_shape: &gpt_rs::backend::spec::Shape,
) -> BackendResult<([i32; 4], [i32; 4])> {
    let in_dims = static_dims_or_error(in_shape, |_| {
        BackendError::execution("dynamic dimensions are not supported by triton runtime")
    })?;
    let out_dims = static_dims_or_error(out_shape, |_| {
        BackendError::execution("dynamic dimensions are not supported by triton runtime")
    })?;
    if out_dims.len() > 4 || in_dims.len() > 4 {
        return Err(BackendError::execution(
            "broadcast runtime supports rank <= 4 only",
        ));
    }
    if out_dims.len() < in_dims.len() {
        return Err(BackendError::execution(
            "broadcast result rank must be >= input rank",
        ));
    }

    let mut out4 = [1usize; 4];
    let mut in4 = [1usize; 4];
    for (idx, dim) in out_dims.iter().enumerate() {
        out4[4 - out_dims.len() + idx] = *dim;
    }
    for (idx, dim) in in_dims.iter().enumerate() {
        in4[4 - in_dims.len() + idx] = *dim;
    }

    let base_strides =
        contiguous_strides_or_error(&in4, || BackendError::execution("stride overflow"))?;
    let mut in_strides = [0i32; 4];
    let mut out_i32 = [0i32; 4];
    for axis in 0..4 {
        let in_dim = in4[axis];
        let out_dim = out4[axis];
        if !(in_dim == out_dim || in_dim == 1) {
            return Err(BackendError::execution(format!(
                "broadcast dim mismatch at axis {axis}: input={in_dim}, output={out_dim}"
            )));
        }
        let stride = if in_dim == 1 && out_dim > 1 {
            0usize
        } else {
            base_strides[axis]
        };
        in_strides[axis] = i32::try_from(stride)
            .map_err(|_| BackendError::execution("broadcast input stride exceeds i32 range"))?;
        out_i32[axis] = i32::try_from(out_dim)
            .map_err(|_| BackendError::execution("broadcast output dim exceeds i32 range"))?;
    }

    Ok((out_i32, in_strides))
}

fn slice_rank4_layout(
    in_shape: &gpt_rs::backend::spec::Shape,
    out_shape: &gpt_rs::backend::spec::Shape,
    starts: &[usize],
) -> BackendResult<([i32; 4], [i32; 4], [i32; 4])> {
    let in_dims = static_dims_or_error(in_shape, |_| {
        BackendError::execution("dynamic dimensions are not supported by triton runtime")
    })?;
    let out_dims = static_dims_or_error(out_shape, |_| {
        BackendError::execution("dynamic dimensions are not supported by triton runtime")
    })?;
    if in_dims.len() > 4 || out_dims.len() > 4 {
        return Err(BackendError::execution(
            "slice runtime supports rank <= 4 only",
        ));
    }
    if in_dims.len() != out_dims.len() || starts.len() != in_dims.len() {
        return Err(BackendError::execution("slice starts/sizes rank mismatch"));
    }
    for axis in 0..in_dims.len() {
        if match starts[axis].checked_add(out_dims[axis]) {
            Some(end) => end > in_dims[axis],
            None => true,
        } {
            return Err(BackendError::execution(format!(
                "slice out of bounds at axis {axis}"
            )));
        }
    }

    let mut in4 = [1usize; 4];
    let mut out4 = [1usize; 4];
    let mut starts4 = [0usize; 4];
    let offset = 4 - in_dims.len();
    in4[offset..(offset + in_dims.len())].copy_from_slice(&in_dims);
    out4[offset..(offset + out_dims.len())].copy_from_slice(&out_dims);
    starts4[offset..(offset + starts.len())].copy_from_slice(starts);
    let strides = contiguous_strides_or_error(&in4, || BackendError::execution("stride overflow"))?;
    let mut out_i32 = [0i32; 4];
    let mut strides_i32 = [0i32; 4];
    let mut starts_i32 = [0i32; 4];
    for axis in 0..4 {
        out_i32[axis] = i32::try_from(out4[axis])
            .map_err(|_| BackendError::execution("slice output dim exceeds i32 range"))?;
        strides_i32[axis] = i32::try_from(strides[axis])
            .map_err(|_| BackendError::execution("slice input stride exceeds i32 range"))?;
        starts_i32[axis] = i32::try_from(starts4[axis])
            .map_err(|_| BackendError::execution("slice start exceeds i32 range"))?;
    }
    Ok((out_i32, strides_i32, starts_i32))
}

fn transpose_rank5_layout(
    in_shape: &gpt_rs::backend::spec::Shape,
    out_shape: &gpt_rs::backend::spec::Shape,
    perm: &[usize],
) -> BackendResult<([i32; 5], [i32; 5])> {
    let in_dims = static_dims_or_error(in_shape, |_| {
        BackendError::execution("dynamic dimensions are not supported by triton runtime")
    })?;
    let out_dims = static_dims_or_error(out_shape, |_| {
        BackendError::execution("dynamic dimensions are not supported by triton runtime")
    })?;
    if in_dims.len() > 5 || out_dims.len() > 5 {
        return Err(BackendError::execution(
            "transpose runtime supports rank <= 5 only",
        ));
    }
    if in_dims.len() != out_dims.len() || perm.len() != in_dims.len() {
        return Err(BackendError::execution(
            "transpose rank/permutation mismatch",
        ));
    }
    let input_strides =
        contiguous_strides_or_error(&in_dims, || BackendError::execution("stride overflow"))?;
    let mut mapped_strides = vec![0usize; perm.len()];
    let mut seen = vec![false; perm.len()];
    for (axis, src_axis) in perm.iter().enumerate() {
        if *src_axis >= perm.len() || seen[*src_axis] {
            return Err(BackendError::execution(
                "transpose permutation must be a valid unique permutation",
            ));
        }
        seen[*src_axis] = true;
        mapped_strides[axis] = input_strides[*src_axis];
    }

    let mut out5 = [1usize; 5];
    let mut strides5 = [0usize; 5];
    let offset = 5 - out_dims.len();
    out5[offset..(offset + out_dims.len())].copy_from_slice(&out_dims);
    strides5[offset..(offset + out_dims.len())].copy_from_slice(&mapped_strides);

    let mut out_i32 = [0i32; 5];
    let mut strides_i32 = [0i32; 5];
    for axis in 0..5 {
        out_i32[axis] = i32::try_from(out5[axis])
            .map_err(|_| BackendError::execution("transpose output dim exceeds i32 range"))?;
        strides_i32[axis] = i32::try_from(strides5[axis])
            .map_err(|_| BackendError::execution("transpose input stride exceeds i32 range"))?;
    }
    Ok((out_i32, strides_i32))
}

fn normalize_axis(axis: isize, rank: usize) -> BackendResult<usize> {
    let rank_isize =
        isize::try_from(rank).map_err(|_| BackendError::execution("rank exceeds isize"))?;
    let normalized = if axis < 0 { rank_isize + axis } else { axis };
    if normalized < 0 || normalized >= rank_isize {
        return Err(BackendError::execution(format!(
            "axis {axis} out of bounds for rank {rank}"
        )));
    }
    usize::try_from(normalized).map_err(|_| BackendError::execution("axis conversion overflow"))
}

type ConcatRank4Layout = ([i32; 4], [i32; 4], [i32; 4], i32, i32);

fn concat_rank4_layout(
    lhs_shape: &gpt_rs::backend::spec::Shape,
    rhs_shape: &gpt_rs::backend::spec::Shape,
    out_shape: &gpt_rs::backend::spec::Shape,
    axis: isize,
) -> BackendResult<ConcatRank4Layout> {
    let lhs_dims = static_dims_or_error(lhs_shape, |_| {
        BackendError::execution("dynamic dimensions are not supported by triton runtime")
    })?;
    let rhs_dims = static_dims_or_error(rhs_shape, |_| {
        BackendError::execution("dynamic dimensions are not supported by triton runtime")
    })?;
    let out_dims = static_dims_or_error(out_shape, |_| {
        BackendError::execution("dynamic dimensions are not supported by triton runtime")
    })?;
    if lhs_dims.len() > 4 || rhs_dims.len() > 4 || out_dims.len() > 4 {
        return Err(BackendError::execution(
            "concat runtime supports rank <= 4 only",
        ));
    }
    if lhs_dims.len() != rhs_dims.len() || lhs_dims.len() != out_dims.len() {
        return Err(BackendError::execution("concat rank mismatch"));
    }
    let axis = normalize_axis(axis, lhs_dims.len())?;
    for idx in 0..lhs_dims.len() {
        if idx == axis {
            if match lhs_dims[idx].checked_add(rhs_dims[idx]) {
                Some(sum) => sum != out_dims[idx],
                None => true,
            } {
                return Err(BackendError::execution(
                    "concat output axis dimension mismatch",
                ));
            }
        } else if lhs_dims[idx] != rhs_dims[idx] || lhs_dims[idx] != out_dims[idx] {
            return Err(BackendError::execution(format!(
                "concat non-axis dimension mismatch at axis {idx}"
            )));
        }
    }

    let lhs_strides =
        contiguous_strides_or_error(&lhs_dims, || BackendError::execution("stride overflow"))?;
    let rhs_strides =
        contiguous_strides_or_error(&rhs_dims, || BackendError::execution("stride overflow"))?;

    let mut out4 = [1usize; 4];
    let mut lhs4 = [0usize; 4];
    let mut rhs4 = [0usize; 4];
    let offset = 4 - out_dims.len();
    out4[offset..(offset + out_dims.len())].copy_from_slice(&out_dims);
    lhs4[offset..(offset + out_dims.len())].copy_from_slice(&lhs_strides);
    rhs4[offset..(offset + out_dims.len())].copy_from_slice(&rhs_strides);

    let mut out_i32 = [0i32; 4];
    let mut lhs_i32 = [0i32; 4];
    let mut rhs_i32 = [0i32; 4];
    for idx in 0..4 {
        out_i32[idx] = i32::try_from(out4[idx])
            .map_err(|_| BackendError::execution("concat output dim exceeds i32 range"))?;
        lhs_i32[idx] = i32::try_from(lhs4[idx])
            .map_err(|_| BackendError::execution("concat lhs stride exceeds i32 range"))?;
        rhs_i32[idx] = i32::try_from(rhs4[idx])
            .map_err(|_| BackendError::execution("concat rhs stride exceeds i32 range"))?;
    }

    let axis_i32 = i32::try_from(offset + axis)
        .map_err(|_| BackendError::execution("concat axis exceeds i32 range"))?;
    let split_i32 = i32::try_from(lhs_dims[axis])
        .map_err(|_| BackendError::execution("concat split exceeds i32 range"))?;
    Ok((out_i32, lhs_i32, rhs_i32, axis_i32, split_i32))
}

fn dynamic_update_rank4_layout(
    update_dims: &[usize],
    out_dims: &[usize],
    starts: &[usize],
) -> BackendResult<([i32; 4], [i32; 4], [i32; 4])> {
    if update_dims.len() > 4 || out_dims.len() > 4 {
        return Err(BackendError::execution(
            "dynamic_update_slice runtime supports rank <= 4 only",
        ));
    }
    if update_dims.len() != out_dims.len() || starts.len() != out_dims.len() {
        return Err(BackendError::execution(
            "dynamic_update_slice rank mismatch",
        ));
    }
    let out_strides =
        contiguous_strides_or_error(out_dims, || BackendError::execution("stride overflow"))?;

    let mut update4 = [1usize; 4];
    let mut out_strides4 = [0usize; 4];
    let mut starts4 = [0usize; 4];
    let offset = 4 - out_dims.len();
    update4[offset..(offset + out_dims.len())].copy_from_slice(update_dims);
    out_strides4[offset..(offset + out_dims.len())].copy_from_slice(&out_strides);
    starts4[offset..(offset + out_dims.len())].copy_from_slice(starts);

    let mut update_i32 = [0i32; 4];
    let mut out_strides_i32 = [0i32; 4];
    let mut starts_i32 = [0i32; 4];
    for idx in 0..4 {
        update_i32[idx] = i32::try_from(update4[idx])
            .map_err(|_| BackendError::execution("dynamic_update update dim exceeds i32 range"))?;
        out_strides_i32[idx] = i32::try_from(out_strides4[idx]).map_err(|_| {
            BackendError::execution("dynamic_update output stride exceeds i32 range")
        })?;
        starts_i32[idx] = i32::try_from(starts4[idx])
            .map_err(|_| BackendError::execution("dynamic_update start exceeds i32 range"))?;
    }
    Ok((update_i32, out_strides_i32, starts_i32))
}

fn launch_1d(
    driver: &Arc<CudaDriver>,
    kernel: &LoadedKernel,
    n: u32,
    block_x: u32,
    params: &mut [*mut c_void],
) -> BackendResult<()> {
    if n == 0 {
        return Ok(());
    }
    let grid_x = n.div_ceil(block_x);
    launch_program_grid(driver, kernel, grid_x, block_x, n, params)
}

fn launch_program_grid(
    driver: &Arc<CudaDriver>,
    kernel: &LoadedKernel,
    grid_x: u32,
    block_x: u32,
    work_elements: u32,
    params: &mut [*mut c_void],
) -> BackendResult<()> {
    if grid_x == 0 {
        return Ok(());
    }
    launch_program_grid_2d(driver, kernel, grid_x, 1, block_x, work_elements, params)
}

fn launch_program_grid_2d(
    driver: &Arc<CudaDriver>,
    kernel: &LoadedKernel,
    grid_x: u32,
    grid_y: u32,
    block_x: u32,
    work_elements: u32,
    params: &mut [*mut c_void],
) -> BackendResult<()> {
    if grid_x == 0 || grid_y == 0 {
        return Ok(());
    }
    KERNEL_LAUNCH_COUNT.fetch_add(1, Ordering::Relaxed);
    let _scope = profiling::backend_scope_with_meta("backend.triton.kernel", || {
        let meta = kernel
            .profile_signature
            .map(ScopeMeta::signature)
            .unwrap_or_default();
        meta.with_work(WorkStats {
            elements: u64::from(work_elements),
            ..WorkStats::default()
        })
    });
    if GPU_EVENT_TIMING_ENABLED.load(Ordering::Relaxed) {
        let host_start = std::time::Instant::now();
        let elapsed_ms = driver.time_with_events("backend.triton.kernel", || {
            driver.launch_kernel(
                &kernel.function,
                (grid_x, grid_y, 1),
                (block_x, 1, 1),
                0,
                params,
            )
        })?;
        let host_duration = host_start.elapsed();
        let gpu_duration = Duration::from_secs_f64(f64::from(elapsed_ms) * 1e-3);
        let work = WorkStats {
            elements: u64::from(work_elements),
            ..WorkStats::default()
        };
        profiling::record_backend_aggregate_with_signature(
            "backend.triton.kernel_gpu",
            kernel.profile_signature,
            1,
            gpu_duration,
            work,
        );
        profiling::record_backend_aggregate_with_signature(
            "backend.triton.kernel_host",
            kernel.profile_signature,
            1,
            host_duration,
            work,
        );
        Ok(())
    } else {
        driver.launch_kernel(
            &kernel.function,
            (grid_x, grid_y, 1),
            (block_x, 1, 1),
            0,
            params,
        )
    }
}

fn read_i32_tensor(tensor: &TritonTensor) -> BackendResult<Vec<i32>> {
    if tensor.spec.dtype != DType::Si32 {
        return Err(BackendError::execution(
            "read_i32_tensor requires Si32 tensor",
        ));
    }
    let bytes = tensor.buffer.read_to_vec()?;
    if bytes.len() % 4 != 0 {
        return Err(BackendError::execution(
            "Si32 tensor byte length is not divisible by 4",
        ));
    }
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        out.push(i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
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

fn unary_opcode(op: ElementwiseUnaryOp) -> BackendResult<u32> {
    match op {
        ElementwiseUnaryOp::Neg => Ok(0),
        ElementwiseUnaryOp::Abs => Ok(1),
        ElementwiseUnaryOp::Exp => Ok(2),
        ElementwiseUnaryOp::Log => Ok(3),
        ElementwiseUnaryOp::Tanh => Ok(4),
        ElementwiseUnaryOp::Erf => Ok(5),
        ElementwiseUnaryOp::Rsqrt => Ok(6),
        ElementwiseUnaryOp::Reciprocal => Ok(7),
    }
}

fn compare_opcode(op: ComparisonOp) -> u32 {
    match op {
        ComparisonOp::Less => 0,
        ComparisonOp::LessEqual => 1,
        ComparisonOp::Equal => 2,
        ComparisonOp::GreaterEqual => 3,
        ComparisonOp::Greater => 4,
        ComparisonOp::NotEqual => 5,
    }
}

fn cublas_profile_scope(
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

struct DotGeneralArgs<'a> {
    spec: &'a gpt_rs::backend::spec::DotGeneralSpec,
    lhs_spec: &'a TensorSpec,
    rhs_spec: &'a TensorSpec,
    out_spec: &'a TensorSpec,
}

struct DotBiasRank2Args<'a> {
    driver: &'a Arc<CudaDriver>,
    kernel: &'a KernelSpec,
    plan: &'a fused_dot_epilogue::FusedDotBiasPlan,
    lhs: &'a TritonTensor,
    rhs: &'a TritonTensor,
    bias: &'a TritonTensor,
    out_spec: &'a TensorSpec,
}

#[derive(Copy, Clone)]
struct StridedBatchedGemmConfig {
    m: usize,
    n: usize,
    k: usize,
    lhs_stride: usize,
    rhs_stride: usize,
    out_stride: usize,
    batches: usize,
}

type CublasStatus = i32;
type CublasHandle = *mut c_void;

const CUBLAS_STATUS_SUCCESS: CublasStatus = 0;
const CUBLAS_OP_N: i32 = 0;
const CUBLAS_OP_T: i32 = 1;

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
    sgemm: CublasSgemmFn,
    sgemm_strided_batched: CublasSgemmStridedBatchedFn,
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

    fn new(driver: Arc<CudaDriver>) -> BackendResult<Self> {
        let lib = load_cublas_library()?;
        let fns = CublasFns {
            create: load_cublas_symbol(&lib, b"cublasCreate_v2\0")?,
            destroy: load_cublas_symbol(&lib, b"cublasDestroy_v2\0")?,
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
        self.sgemm_row_major_raw(
            lhs.device_ptr(),
            rhs.device_ptr(),
            out.device_ptr(),
            m,
            n,
            k,
        )
    }

    fn sgemm_row_major_strided_batched(
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

    fn sgemm_row_major_strided_batched_rhs_transposed(
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

    fn sgemm_row_major_raw(
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
        let alpha = 1.0f32;
        let beta = 0.0f32;
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
                        &alpha as *const f32,
                        rhs_ptr as usize as *const f32,
                        n_i32,
                        lhs_ptr as usize as *const f32,
                        k_i32,
                        &beta as *const f32,
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
        let alpha = 1.0f32;
        let beta = 0.0f32;
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
                        &alpha as *const f32,
                        rhs_ptr as usize as *const f32,
                        n_i32,
                        rhs_stride_i64,
                        lhs_ptr as usize as *const f32,
                        k_i32,
                        lhs_stride_i64,
                        &beta as *const f32,
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
        let alpha = 1.0f32;
        let beta = 0.0f32;
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
                        &alpha as *const f32,
                        rhs_ptr as usize as *const f32,
                        k_i32,
                        rhs_stride_i64,
                        lhs_ptr as usize as *const f32,
                        k_i32,
                        lhs_stride_i64,
                        &beta as *const f32,
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

impl Default for TritonExecutor {
    fn default() -> Self {
        Self::new()
    }
}
