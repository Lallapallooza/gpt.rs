use std::collections::HashMap;
use std::sync::Arc;

use gpt_rs::backend::param_resolver::{InMemoryParamResolver, ParamResolver};
use gpt_rs::backend::spec::{
    BackendError, BackendResult, BroadcastToSpec, CastSpec, ComparisonOp, ConcatSpec, DType,
    Dimension, DotGeneralSpec, ElementwiseBinaryOp, ElementwiseUnaryOp, ExtractPatchesSpec,
    Instruction, IotaSpec, Literal, Operand, Operation, PortableBackend, Program, ReduceKind,
    ReduceSpec, ReduceWindowSpec, ReshapeSpec, Shape, TensorInit, TensorLiteral, TensorSpec,
    ValueId, ValueType,
};

#[derive(Clone)]
pub struct CpuTensor {
    pub spec: TensorSpec,
    pub data: TensorData,
}

#[derive(Clone)]
pub enum TensorData {
    F32(Arc<[f32]>),
    Si32(Arc<[i32]>),
    Bool(Arc<[u8]>),
}

pub trait CpuKernelInterceptor: Send + Sync {
    fn try_execute(
        &self,
        op: &Operation,
        inputs: &[CpuTensor],
        outputs: &[TensorSpec],
    ) -> Option<BackendResult<Vec<CpuTensor>>>;
}

#[derive(Default)]
pub struct NoopInterceptor;

impl CpuKernelInterceptor for NoopInterceptor {
    fn try_execute(
        &self,
        _op: &Operation,
        _inputs: &[CpuTensor],
        _outputs: &[TensorSpec],
    ) -> Option<BackendResult<Vec<CpuTensor>>> {
        None
    }
}

#[derive(Clone)]
pub struct GenericCpuBackend<I: CpuKernelInterceptor> {
    interceptor: Arc<I>,
    params: Arc<InMemoryParamResolver<CpuTensor>>,
}

impl<I: CpuKernelInterceptor> GenericCpuBackend<I> {
    pub fn with_interceptor(interceptor: I) -> Self {
        Self {
            interceptor: Arc::new(interceptor),
            params: Arc::new(InMemoryParamResolver::new()),
        }
    }

    pub fn with_arc(interceptor: Arc<I>) -> Self {
        Self {
            interceptor,
            params: Arc::new(InMemoryParamResolver::new()),
        }
    }

    pub fn interceptor(&self) -> &I {
        self.interceptor.as_ref()
    }
}

impl GenericCpuBackend<NoopInterceptor> {
    pub fn new() -> Self {
        Self::with_interceptor(NoopInterceptor)
    }
}

impl Default for GenericCpuBackend<NoopInterceptor> {
    fn default() -> Self {
        Self::new()
    }
}

pub type CpuPortableBackend = GenericCpuBackend<NoopInterceptor>;

impl<I: CpuKernelInterceptor> PortableBackend for GenericCpuBackend<I> {
    type TensorHandle = CpuTensor;

    fn backend_name(&self) -> &str {
        "cpu-portable"
    }

    fn param_resolver(&self) -> Option<Arc<dyn ParamResolver<Handle = Self::TensorHandle>>> {
        Some(self.params.clone())
    }

    fn materialize(&self, init: TensorInit) -> BackendResult<Self::TensorHandle> {
        match init {
            TensorInit::Literal(lit) => literal_to_tensor(&lit),
            TensorInit::Zeroed(spec) => zeroed_tensor(&spec),
        }
    }

    fn to_literal(&self, tensor: &Self::TensorHandle) -> BackendResult<TensorLiteral> {
        tensor_to_literal(tensor)
    }

    fn execute_instruction(
        &self,
        instruction: &Instruction,
        inputs: &[Self::TensorHandle],
    ) -> BackendResult<Vec<Self::TensorHandle>> {
        execute_operation(self.interceptor.as_ref(), instruction, inputs)
    }

    fn run_program(
        &self,
        program: &Program,
        entry_inputs: &[Self::TensorHandle],
    ) -> BackendResult<Vec<Self::TensorHandle>> {
        let function = program
            .functions
            .iter()
            .find(|f| f.name == program.entry)
            .ok_or_else(|| BackendError::execution("entry function not found"))?;

        if function.parameter_ids.len() != entry_inputs.len() {
            return Err(BackendError::execution("entry input arity mismatch"));
        }

        let mut values: HashMap<ValueId, CpuTensor> = HashMap::new();
        for (param_id, handle) in function.parameter_ids.iter().zip(entry_inputs.iter()) {
            values.insert(*param_id, handle.clone());
        }

        for (instr_index, instruction) in function.body.iter().enumerate() {
            let mut inputs = Vec::with_capacity(instruction.operands.len());
            for operand in &instruction.operands {
                let tensor = match operand {
                    Operand::Value(id) => values
                        .get(id)
                        .cloned()
                        .ok_or_else(|| BackendError::execution("operand value missing"))?,
                    Operand::TupleElement { .. } => {
                        return Err(BackendError::execution("tuple operands not supported"))
                    }
                    Operand::Literal(lit) => literal_to_tensor(lit)?,
                };
                inputs.push(tensor);
            }
            let mut outputs = execute_operation(self.interceptor.as_ref(), instruction, &inputs)
                .map_err(|err| {
                    augment_backend_error(err, &function.name, instr_index, instruction, &inputs)
                })?;
            if outputs.len() != 1 {
                return Err(BackendError::execution(
                    "instructions must produce exactly one result",
                ));
            }
            values.insert(
                instruction.id,
                outputs
                    .pop()
                    .expect("single output guaranteed by length check"),
            );
        }

        let mut results = Vec::with_capacity(function.result_ids.len());
        for id in &function.result_ids {
            let value = values
                .get(id)
                .cloned()
                .ok_or_else(|| BackendError::execution("missing function result value"))?;
            results.push(value);
        }
        Ok(results)
    }
}

fn literal_to_tensor(literal: &TensorLiteral) -> BackendResult<CpuTensor> {
    match literal.spec.dtype {
        DType::F32 => {
            let data = bytes_to_f32(&literal.bytes)?;
            Ok(CpuTensor {
                spec: literal.spec.clone(),
                data: TensorData::F32(Arc::from(data)),
            })
        }
        DType::Si32 => {
            let data = bytes_to_i32(&literal.bytes)?;
            Ok(CpuTensor {
                spec: literal.spec.clone(),
                data: TensorData::Si32(Arc::from(data)),
            })
        }
        DType::I1 => {
            let data = literal.bytes.iter().cloned().collect::<Vec<u8>>();
            Ok(CpuTensor {
                spec: literal.spec.clone(),
                data: TensorData::Bool(Arc::from(data)),
            })
        }
        _ => Err(BackendError::spec(
            gpt_rs::backend::spec::SpecErrorCode::DTypeNotSupported,
            format!("literal dtype {:?} unsupported", literal.spec.dtype),
        )),
    }
}

fn zeroed_tensor(spec: &TensorSpec) -> BackendResult<CpuTensor> {
    let elem_count = element_count(&spec.shape)?;
    match spec.dtype {
        DType::F32 => Ok(CpuTensor {
            spec: spec.clone(),
            data: TensorData::F32(Arc::from(vec![0.0; elem_count])),
        }),
        DType::Si32 => Ok(CpuTensor {
            spec: spec.clone(),
            data: TensorData::Si32(Arc::from(vec![0; elem_count])),
        }),
        DType::I1 => Ok(CpuTensor {
            spec: spec.clone(),
            data: TensorData::Bool(Arc::from(vec![0; elem_count])),
        }),
        _ => Err(BackendError::spec(
            gpt_rs::backend::spec::SpecErrorCode::DTypeNotSupported,
            format!("zero init dtype {:?} unsupported", spec.dtype),
        )),
    }
}

fn tensor_to_literal(tensor: &CpuTensor) -> BackendResult<TensorLiteral> {
    match &tensor.data {
        TensorData::F32(values) => Ok(TensorLiteral::new(
            tensor.spec.clone(),
            f32_to_bytes(values.as_ref()),
        )),
        TensorData::Si32(values) => Ok(TensorLiteral::new(
            tensor.spec.clone(),
            i32_to_bytes(values.as_ref()),
        )),
        TensorData::Bool(values) => Ok(TensorLiteral::new(tensor.spec.clone(), Arc::clone(values))),
    }
}

fn execute_operation(
    interceptor: &dyn CpuKernelInterceptor,
    instruction: &Instruction,
    inputs: &[CpuTensor],
) -> BackendResult<Vec<CpuTensor>> {
    let output_spec = match &instruction.output {
        ValueType::Tensor(spec) => spec.clone(),
        ValueType::Tuple(_) => {
            return Err(BackendError::execution(
                "tuple outputs are not supported in cpu portable backend",
            ))
        }
    };
    let output_specs = [output_spec];

    // `CustomCall` is already profiled by the interceptor implementations (they know the target and
    // can emit sub-scopes). Avoid double-counting / duplicated rows from the generic
    // `backend.custom_call` wrapper scope.
    if matches!(instruction.op, Operation::CustomCall(_)) {
        if let Some(result) = interceptor.try_execute(&instruction.op, inputs, &output_specs) {
            return result;
        }
    }

    let label = backend_operation_label(&instruction.op);
    let _prof_guard = gpt_rs::profiling::backend_scope_with_meta(label, || {
        let signature = gpt_rs::profiling::signature_id(&gpt_rs::profiling::tensor_spec_signature(
            &output_specs[0],
        ));
        let work = estimate_work_stats(&instruction.op, inputs, &output_specs[0]);
        let meta = signature
            .map(gpt_rs::profiling::ScopeMeta::signature)
            .unwrap_or_default();
        meta.with_work(work)
    });

    if let Some(result) = interceptor.try_execute(&instruction.op, inputs, &output_specs) {
        return result;
    }

    let result = match &instruction.op {
        Operation::Constant(literal) => vec![literal_to_tensor(literal)?],
        Operation::Reshape(spec) => vec![op_reshape(inputs, &output_specs[0], spec)?],
        Operation::Slice(spec) => vec![op_slice(inputs, &output_specs[0], spec)?],
        Operation::Transpose(spec) => vec![op_transpose(inputs, &output_specs[0], spec)?],
        Operation::BroadcastTo(spec) => vec![op_broadcast_to(inputs, &output_specs[0], spec)?],
        Operation::DotGeneral(spec) => vec![op_dot_general(inputs, &output_specs[0], spec)?],
        Operation::ElementwiseBinary(op) => {
            vec![op_elementwise_binary(inputs, &output_specs[0], *op)?]
        }
        Operation::ElementwiseUnary(op) => {
            vec![op_elementwise_unary(inputs, &output_specs[0], *op)?]
        }
        Operation::Reduce(spec) => vec![op_reduce(inputs, &output_specs[0], spec)?],
        Operation::ExtractPatches(spec) => {
            vec![op_extract_patches(inputs, &output_specs[0], spec)?]
        }
        Operation::ReduceWindow(spec) => vec![op_reduce_window(inputs, &output_specs[0], spec)?],
        Operation::Compare(spec) => vec![op_compare(inputs, &output_specs[0], spec)?],
        Operation::Select => vec![op_select(inputs, &output_specs[0])?],
        Operation::Cast(spec) => vec![op_cast(inputs, &output_specs[0], spec)?],
        Operation::Concat(spec) => vec![op_concat(inputs, &output_specs[0], spec)?],
        Operation::Take => vec![op_take(inputs, &output_specs[0])?],
        Operation::Gather(spec) => vec![op_gather(inputs, &output_specs[0], spec)?],
        Operation::Iota(spec) => vec![op_iota(&output_specs[0], spec)?],
        Operation::DynamicSlice(spec) => vec![op_dynamic_slice(inputs, &output_specs[0], spec)?],
        Operation::DynamicUpdateSlice(spec) => {
            vec![op_dynamic_update_slice(inputs, &output_specs[0], spec)?]
        }
        other => {
            return Err(BackendError::unimplemented(
                other_name(other),
                "operation not supported in cpu portable backend",
            ))
        }
    };
    Ok(result)
}

fn dtype_size_in_bytes(dtype: DType) -> u64 {
    dtype
        .bitwidth()
        .map(|bits| ((bits as u64).saturating_add(7)) / 8)
        .unwrap_or(0)
}

fn spec_elements(spec: &TensorSpec) -> u64 {
    spec.shape
        .dims()
        .iter()
        .map(|dim| match dim {
            Dimension::Static(v) => *v as u64,
            Dimension::Dynamic(_) => 0,
        })
        .product()
}

fn spec_bytes(spec: &TensorSpec) -> u64 {
    spec_elements(spec).saturating_mul(dtype_size_in_bytes(spec.dtype))
}

fn dot_general_flops(lhs: &TensorSpec, rhs: &TensorSpec, spec: &DotGeneralSpec) -> u64 {
    let lhs_dims = match static_dims(&lhs.shape) {
        Ok(dims) => dims,
        Err(_) => return 0,
    };
    let rhs_dims = match static_dims(&rhs.shape) {
        Ok(dims) => dims,
        Err(_) => return 0,
    };

    let mut batch = 1u128;
    for &axis in &spec.batch_lhs {
        if let Some(&dim) = lhs_dims.get(axis) {
            batch = batch.saturating_mul(dim as u128);
        } else {
            return 0;
        }
    }

    let mut contract = 1u128;
    for &axis in &spec.contract_lhs {
        if let Some(&dim) = lhs_dims.get(axis) {
            contract = contract.saturating_mul(dim as u128);
        } else {
            return 0;
        }
    }

    let mut lhs_free = 1u128;
    for (axis, &dim) in lhs_dims.iter().enumerate() {
        if spec.batch_lhs.contains(&axis) || spec.contract_lhs.contains(&axis) {
            continue;
        }
        lhs_free = lhs_free.saturating_mul(dim as u128);
    }

    let mut rhs_free = 1u128;
    for (axis, &dim) in rhs_dims.iter().enumerate() {
        if spec.batch_rhs.contains(&axis) || spec.contract_rhs.contains(&axis) {
            continue;
        }
        rhs_free = rhs_free.saturating_mul(dim as u128);
    }

    let output_elements = batch.saturating_mul(lhs_free).saturating_mul(rhs_free);
    let flops = output_elements.saturating_mul(contract).saturating_mul(2);
    flops.min(u64::MAX as u128) as u64
}

fn estimate_work_stats(
    op: &Operation,
    inputs: &[CpuTensor],
    output_spec: &TensorSpec,
) -> gpt_rs::profiling::WorkStats {
    let out_elements = spec_elements(output_spec);
    let out_bytes = spec_bytes(output_spec);
    let bytes_read = inputs.iter().map(|t| spec_bytes(&t.spec)).sum();

    let flops = match op {
        Operation::ElementwiseBinary(_)
        | Operation::ElementwiseUnary(_)
        | Operation::Compare(_)
        | Operation::Select
        | Operation::Cast(_) => out_elements,
        Operation::Reduce(_)
        | Operation::ReduceWindow(_)
        | Operation::BroadcastTo(_)
        | Operation::Concat(_)
        | Operation::DynamicUpdateSlice(_)
        | Operation::DynamicSlice(_)
        | Operation::Slice(_)
        | Operation::Transpose(_)
        | Operation::ExtractPatches(_)
        | Operation::Take
        | Operation::Gather(_)
        | Operation::Iota(_)
        | Operation::Pad(_)
        | Operation::Tile(_) => 0,
        Operation::DotGeneral(spec) => inputs
            .first()
            .and_then(|lhs| inputs.get(1).map(|rhs| (lhs, rhs)))
            .map(|(lhs, rhs)| dot_general_flops(&lhs.spec, &rhs.spec, spec))
            .unwrap_or(0),
        _ => 0,
    };

    let (alloc_bytes, alloc_count) = match op {
        Operation::Reshape(_) => (0, 0),
        _ => (out_bytes, 1),
    };

    gpt_rs::profiling::WorkStats {
        elements: out_elements,
        bytes_read,
        bytes_written: out_bytes,
        flops,
        alloc_bytes,
        alloc_count,
    }
}

fn augment_backend_error(
    error: BackendError,
    function_name: &str,
    instruction_index: usize,
    instruction: &Instruction,
    inputs: &[CpuTensor],
) -> BackendError {
    match error {
        BackendError::Execution { message } => BackendError::Execution {
            message: format!(
                "{message} (at function `{}` instruction #{}, {} id {:?} operands [{}])",
                function_name,
                instruction_index,
                backend_operation_label(&instruction.op),
                instruction.id,
                format_operands(&instruction.operands, inputs)
            ),
        },
        BackendError::Unimplemented { op, reason } => BackendError::Unimplemented {
            op,
            reason: format!(
                "{} (while executing function `{}` instruction #{}, {} id {:?} operands [{}])",
                reason,
                function_name,
                instruction_index,
                backend_operation_label(&instruction.op),
                instruction.id,
                format_operands(&instruction.operands, inputs)
            ),
        },
        other => other,
    }
}

fn format_operands(operands: &[Operand], inputs: &[CpuTensor]) -> String {
    if operands.is_empty() {
        return String::from("<none>");
    }

    operands
        .iter()
        .zip(inputs.iter())
        .map(|(operand, tensor)| format_operand(operand, tensor))
        .collect::<Vec<_>>()
        .join(", ")
}

fn format_operand(operand: &Operand, tensor: &CpuTensor) -> String {
    let shape_desc = match static_dims(&tensor.spec.shape) {
        Ok(dims) => {
            if dims.is_empty() {
                String::from("[]")
            } else {
                format!(
                    "[{}]",
                    dims.iter()
                        .map(|dim| dim.to_string())
                        .collect::<Vec<_>>()
                        .join("x")
                )
            }
        }
        Err(_) => String::from("dynamic"),
    };

    let dtype_desc = format!("{:?}", tensor.spec.dtype);

    match operand {
        Operand::Value(id) => format!("value {:?} {} dtype={}", id, shape_desc, dtype_desc),
        Operand::Literal(_) => format!("literal {} dtype={}", shape_desc, dtype_desc),
        Operand::TupleElement { tuple, index } => format!(
            "tuple {:?}[{}] {} dtype={}",
            tuple, index, shape_desc, dtype_desc
        ),
    }
}

#[allow(unreachable_patterns)]
fn backend_operation_label(op: &Operation) -> &'static str {
    match op {
        Operation::Reshape(_) => "backend.reshape",
        Operation::Slice(_) => "backend.slice",
        Operation::Transpose(_) => "backend.transpose",
        Operation::BroadcastTo(_) => "backend.broadcast_to",
        Operation::DotGeneral(_) => "backend.dot_general",
        Operation::ElementwiseBinary(kind) => match kind {
            ElementwiseBinaryOp::Add => "backend.elementwise_binary.add",
            ElementwiseBinaryOp::Sub => "backend.elementwise_binary.sub",
            ElementwiseBinaryOp::Mul => "backend.elementwise_binary.mul",
            ElementwiseBinaryOp::Div => "backend.elementwise_binary.div",
            ElementwiseBinaryOp::Maximum => "backend.elementwise_binary.maximum",
            ElementwiseBinaryOp::Minimum => "backend.elementwise_binary.minimum",
        },
        Operation::ElementwiseUnary(kind) => match kind {
            ElementwiseUnaryOp::Neg => "backend.elementwise_unary.neg",
            ElementwiseUnaryOp::Abs => "backend.elementwise_unary.abs",
            ElementwiseUnaryOp::Exp => "backend.elementwise_unary.exp",
            ElementwiseUnaryOp::Log => "backend.elementwise_unary.log",
            ElementwiseUnaryOp::Tanh => "backend.elementwise_unary.tanh",
            ElementwiseUnaryOp::Erf => "backend.elementwise_unary.erf",
            ElementwiseUnaryOp::Rsqrt => "backend.elementwise_unary.rsqrt",
            ElementwiseUnaryOp::Reciprocal => "backend.elementwise_unary.reciprocal",
        },
        Operation::Reduce(spec) => match spec.kind {
            ReduceKind::Sum => "backend.reduce.sum",
            ReduceKind::Max => "backend.reduce.max",
            ReduceKind::Min => "backend.reduce.min",
        },
        Operation::Compare(spec) => match spec.op {
            ComparisonOp::Equal => "backend.compare.equal",
            ComparisonOp::NotEqual => "backend.compare.not_equal",
            ComparisonOp::Greater => "backend.compare.greater",
            ComparisonOp::GreaterEqual => "backend.compare.greater_equal",
            ComparisonOp::Less => "backend.compare.less",
            ComparisonOp::LessEqual => "backend.compare.less_equal",
        },
        Operation::Select => "backend.select",
        Operation::Take => "backend.take",
        Operation::Gather(_) => "backend.gather",
        Operation::Iota(_) => "backend.iota",
        Operation::ScatterAdd(_) => "backend.scatter_add",
        Operation::RngUniform(_) => "backend.rng_uniform",
        Operation::RngNormal(_) => "backend.rng_normal",
        Operation::TopK(_) => "backend.top_k",
        Operation::Pad(_) => "backend.pad",
        Operation::Concat(_) => "backend.concat",
        Operation::Tile(_) => "backend.tile",
        Operation::DynamicSlice(_) => "backend.dynamic_slice",
        Operation::DynamicUpdateSlice(_) => "backend.dynamic_update_slice",
        Operation::Cast(_) => "backend.cast",
        Operation::StopGradient => "backend.stop_gradient",
        Operation::Constant(_) => "backend.constant",
        Operation::While(_) => "backend.while",
        Operation::Scan(_) => "backend.scan",
        Operation::Cond(_) => "backend.cond",
        Operation::ExtractPatches(_) => "backend.extract_patches",
        Operation::ReduceWindow(_) => "backend.reduce_window",
        Operation::SegmentReduce(_) => "backend.segment_reduce",
        Operation::ArgMax(_) => "backend.argmax",
        Operation::Quantize(_) => "backend.quantize",
        Operation::Dequantize(_) => "backend.dequantize",
        Operation::Requantize(_) => "backend.requantize",
        Operation::CustomCall(_) => "backend.custom_call",
        _ => "backend.other",
    }
}

fn other_name(op: &Operation) -> &'static str {
    match op {
        Operation::Constant(_) => "constant",
        Operation::Reshape(_) => "reshape",
        Operation::Slice(_) => "slice",
        Operation::Transpose(_) => "transpose",
        Operation::BroadcastTo(_) => "broadcast_to",
        Operation::DotGeneral(_) => "dot_general",
        Operation::ElementwiseBinary(_) => "elementwise_binary",
        Operation::ElementwiseUnary(_) => "elementwise_unary",
        Operation::Reduce(_) => "reduce",
        Operation::Compare(_) => "compare",
        Operation::Select => "select",
        Operation::Concat(_) => "concat",
        Operation::Take => "take",
        Operation::Gather(_) => "gather",
        Operation::Iota(_) => "iota",
        Operation::ExtractPatches(_) => "extract_patches",
        Operation::CustomCall(_) => "custom_call",
        _ => "op",
    }
}

fn op_reshape(
    inputs: &[CpuTensor],
    output: &TensorSpec,
    _spec: &ReshapeSpec,
) -> BackendResult<CpuTensor> {
    let input = expect_single(inputs)?;
    if element_count(&input.spec.shape)? != element_count(&output.shape)? {
        return Err(BackendError::execution("reshape element count mismatch"));
    }
    let data = match &input.data {
        TensorData::F32(values) => TensorData::F32(values.clone()),
        TensorData::Si32(values) => TensorData::Si32(values.clone()),
        TensorData::Bool(values) => TensorData::Bool(values.clone()),
    };
    Ok(CpuTensor {
        spec: output.clone(),
        data,
    })
}

fn op_slice(
    inputs: &[CpuTensor],
    output: &TensorSpec,
    spec: &gpt_rs::backend::spec::SliceSpec,
) -> BackendResult<CpuTensor> {
    let input = expect_single(inputs)?;
    let input_dims = static_dims(&input.spec.shape)?;
    if spec.starts.len() != input_dims.len() || spec.sizes.len() != input_dims.len() {
        return Err(BackendError::execution("slice rank mismatch"));
    }
    match &input.data {
        TensorData::F32(values) => {
            let values = values.as_ref();
            let out_dims = static_dims(&output.shape)?;
            if out_dims.len() != input_dims.len() {
                return Err(BackendError::execution("slice output rank mismatch"));
            }
            let out_len: usize = out_dims.iter().product();
            let mut result = vec![0.0f32; out_len];

            if input_dims.len() == 2 {
                let (m_in, n_in) = (input_dims[0], input_dims[1]);
                let (m_out, n_out) = (out_dims[0], out_dims[1]);
                let (m_start, n_start) = (spec.starts[0], spec.starts[1]);

                if m_start + m_out > m_in || n_start + n_out > n_in {
                    return Err(BackendError::execution("slice out of bounds"));
                }

                for row in 0..m_out {
                    let src_offset = (m_start + row) * n_in + n_start;
                    let dst_offset = row * n_out;
                    result[dst_offset..dst_offset + n_out]
                        .copy_from_slice(&values[src_offset..src_offset + n_out]);
                }
            } else if input_dims.len() == 3 {
                let (a_in, b_in, c_in) = (input_dims[0], input_dims[1], input_dims[2]);
                let (a_out, b_out, c_out) = (out_dims[0], out_dims[1], out_dims[2]);
                let (a_start, b_start, c_start) = (spec.starts[0], spec.starts[1], spec.starts[2]);

                if a_start + a_out > a_in || b_start + b_out > b_in || c_start + c_out > c_in {
                    return Err(BackendError::execution("slice out of bounds"));
                }

                for a in 0..a_out {
                    for b in 0..b_out {
                        let src_offset = ((a_start + a) * b_in + (b_start + b)) * c_in + c_start;
                        let dst_offset = (a * b_out + b) * c_out;
                        result[dst_offset..dst_offset + c_out]
                            .copy_from_slice(&values[src_offset..src_offset + c_out]);
                    }
                }
            } else {
                let strides = compute_strides(&input_dims);
                for (idx, slot) in result.iter_mut().enumerate() {
                    let coord = unravel_index(idx, &out_dims);
                    let mut in_index = 0usize;
                    for (dim, &c) in coord.iter().enumerate() {
                        let source = spec.starts[dim] + c;
                        in_index += source * strides[dim];
                    }
                    *slot = values[in_index];
                }
            }
            Ok(CpuTensor {
                spec: output.clone(),
                data: TensorData::F32(Arc::from(result)),
            })
        }
        _ => Err(BackendError::execution("slice only supports f32 tensors")),
    }
}

fn op_dynamic_slice(
    inputs: &[CpuTensor],
    output: &TensorSpec,
    spec: &gpt_rs::backend::spec::DynamicSliceSpec,
) -> BackendResult<CpuTensor> {
    if inputs.len() != 2 {
        return Err(BackendError::execution(
            "dynamic_slice expects (input, starts)",
        ));
    }
    let input = &inputs[0];
    let starts_tensor = &inputs[1];

    let input_dims = static_dims(&input.spec.shape)?;
    if input_dims.len() != spec.sizes.len() {
        return Err(BackendError::execution(
            "dynamic_slice sizes length must match input rank",
        ));
    }

    let starts_vals = match &starts_tensor.data {
        TensorData::Si32(values) => values.as_ref(),
        _ => return Err(BackendError::execution("dynamic_slice starts must be si32")),
    };
    let starts_dims = static_dims(&starts_tensor.spec.shape)?;
    if starts_dims.len() != 1 || starts_dims[0] != input_dims.len() {
        return Err(BackendError::execution(
            "dynamic_slice starts must be 1-D of length equal to rank",
        ));
    }

    let mut clamped_starts = Vec::with_capacity(input_dims.len());
    for axis in 0..input_dims.len() {
        let dim = input_dims[axis];
        let size = spec.sizes[axis];
        if size > dim {
            return Err(BackendError::execution(
                "dynamic_slice size exceeds dimension",
            ));
        }
        let max_start = dim - size;
        let mut start = starts_vals[axis] as isize;
        if start < 0 {
            start = 0;
        }
        let start = start as usize;
        clamped_starts.push(if start > max_start { max_start } else { start });
    }

    let out_dims = static_dims(&output.shape)?;
    if out_dims != spec.sizes {
        return Err(BackendError::execution(
            "dynamic_slice output shape mismatch",
        ));
    }

    match &input.data {
        TensorData::F32(values) => {
            let strides = compute_strides(&input_dims);
            let out_len: usize = out_dims.iter().product();
            let mut result = vec![0.0f32; out_len];
            for (idx, slot) in result.iter_mut().enumerate() {
                let coord = unravel_index(idx, &out_dims);
                let mut in_index = 0usize;
                for (axis, &c) in coord.iter().enumerate() {
                    in_index += (clamped_starts[axis] + c) * strides[axis];
                }
                *slot = values[in_index];
            }
            Ok(CpuTensor {
                spec: output.clone(),
                data: TensorData::F32(Arc::from(result)),
            })
        }
        TensorData::Si32(values) => {
            let strides = compute_strides(&input_dims);
            let out_len: usize = out_dims.iter().product();
            let mut result = vec![0i32; out_len];
            for (idx, slot) in result.iter_mut().enumerate() {
                let coord = unravel_index(idx, &out_dims);
                let mut in_index = 0usize;
                for (axis, &c) in coord.iter().enumerate() {
                    in_index += (clamped_starts[axis] + c) * strides[axis];
                }
                *slot = values[in_index];
            }
            Ok(CpuTensor {
                spec: output.clone(),
                data: TensorData::Si32(Arc::from(result)),
            })
        }
        _ => Err(BackendError::execution(
            "dynamic_slice only supports f32 and si32 tensors",
        )),
    }
}

fn op_dynamic_update_slice(
    inputs: &[CpuTensor],
    output: &TensorSpec,
    spec: &gpt_rs::backend::spec::DynamicUpdateSliceSpec,
) -> BackendResult<CpuTensor> {
    if inputs.len() != 3 {
        return Err(BackendError::execution(
            "dynamic_update_slice expects (base, update, starts)",
        ));
    }

    let base = &inputs[0];
    let update = &inputs[1];
    let starts_tensor = &inputs[2];

    let base_dims = static_dims(&base.spec.shape)?;
    let update_dims = static_dims(&update.spec.shape)?;

    if base_dims.len() != spec.sizes.len() {
        return Err(BackendError::execution(
            "dynamic_update_slice sizes length must match base rank",
        ));
    }
    if update_dims != spec.sizes {
        return Err(BackendError::execution(
            "dynamic_update_slice update shape mismatch",
        ));
    }
    if base.spec.dtype != update.spec.dtype {
        return Err(BackendError::execution(
            "dynamic_update_slice dtype mismatch between base and update",
        ));
    }

    let starts_vals = match &starts_tensor.data {
        TensorData::Si32(values) => values.as_ref(),
        _ => {
            return Err(BackendError::execution(
                "dynamic_update_slice starts must be si32",
            ))
        }
    };
    let starts_dims = static_dims(&starts_tensor.spec.shape)?;
    if starts_dims.len() != 1 || starts_dims[0] != base_dims.len() {
        return Err(BackendError::execution(
            "dynamic_update_slice starts must be 1-D of length equal to rank",
        ));
    }
    let mut clamped_starts = Vec::with_capacity(base_dims.len());
    for axis in 0..base_dims.len() {
        let dim = base_dims[axis];
        let size = spec.sizes[axis];
        if size > dim {
            return Err(BackendError::execution(
                "dynamic_update_slice size exceeds dimension",
            ));
        }
        let max_start = dim - size;
        let mut start = starts_vals[axis] as isize;
        if start < 0 {
            start = 0;
        }
        let start = start as usize;
        clamped_starts.push(if start > max_start { max_start } else { start });
    }

    if &base.spec != output {
        return Err(BackendError::execution(
            "dynamic_update_slice output spec must match base",
        ));
    }

    match (&base.data, &update.data) {
        (TensorData::F32(base_vals), TensorData::F32(update_vals)) => {
            let mut result = base_vals.as_ref().to_vec();

            if base_dims.len() == 3
                && update_dims.len() == 3
                && update_dims[0] == base_dims[0]
                && update_dims[2] == base_dims[2]
                && clamped_starts[0] == 0
                && clamped_starts[2] == 0
            {
                let heads = base_dims[0];
                let base_seq = base_dims[1];
                let head_dim = base_dims[2];
                let update_seq = update_dims[1];
                let start_seq = clamped_starts[1];

                for head in 0..heads {
                    let dst_head_base = head * base_seq * head_dim;
                    let src_head_base = head * update_seq * head_dim;
                    for t in 0..update_seq {
                        let dst_offset = dst_head_base + (start_seq + t) * head_dim;
                        let src_offset = src_head_base + t * head_dim;
                        result[dst_offset..dst_offset + head_dim]
                            .copy_from_slice(&update_vals[src_offset..src_offset + head_dim]);
                    }
                }
            } else {
                let base_strides = compute_strides(&base_dims);
                let update_strides = compute_strides(&update_dims);
                for idx in 0..update_vals.len() {
                    let coord = unravel_index(idx, &update_dims);
                    let mut dest_index = 0usize;
                    let mut src_index = 0usize;
                    for (axis, &c) in coord.iter().enumerate() {
                        dest_index += (clamped_starts[axis] + c) * base_strides[axis];
                        src_index += c * update_strides[axis];
                    }
                    result[dest_index] = update_vals[src_index];
                }
            }
            Ok(CpuTensor {
                spec: output.clone(),
                data: TensorData::F32(Arc::from(result)),
            })
        }
        (TensorData::Si32(base_vals), TensorData::Si32(update_vals)) => {
            let mut result = base_vals.as_ref().to_vec();

            if base_dims.len() == 3
                && update_dims.len() == 3
                && update_dims[0] == base_dims[0]
                && update_dims[2] == base_dims[2]
                && clamped_starts[0] == 0
                && clamped_starts[2] == 0
            {
                let heads = base_dims[0];
                let base_seq = base_dims[1];
                let head_dim = base_dims[2];
                let update_seq = update_dims[1];
                let start_seq = clamped_starts[1];

                for head in 0..heads {
                    let dst_head_base = head * base_seq * head_dim;
                    let src_head_base = head * update_seq * head_dim;
                    for t in 0..update_seq {
                        let dst_offset = dst_head_base + (start_seq + t) * head_dim;
                        let src_offset = src_head_base + t * head_dim;
                        result[dst_offset..dst_offset + head_dim]
                            .copy_from_slice(&update_vals[src_offset..src_offset + head_dim]);
                    }
                }
            } else {
                let base_strides = compute_strides(&base_dims);
                let update_strides = compute_strides(&update_dims);
                for idx in 0..update_vals.len() {
                    let coord = unravel_index(idx, &update_dims);
                    let mut dest_index = 0usize;
                    let mut src_index = 0usize;
                    for (axis, &c) in coord.iter().enumerate() {
                        dest_index += (clamped_starts[axis] + c) * base_strides[axis];
                        src_index += c * update_strides[axis];
                    }
                    result[dest_index] = update_vals[src_index];
                }
            }
            Ok(CpuTensor {
                spec: output.clone(),
                data: TensorData::Si32(Arc::from(result)),
            })
        }
        _ => Err(BackendError::execution(
            "dynamic_update_slice only supports f32 and si32 tensors",
        )),
    }
}

fn op_transpose(
    inputs: &[CpuTensor],
    output: &TensorSpec,
    spec: &gpt_rs::backend::spec::TransposeSpec,
) -> BackendResult<CpuTensor> {
    let input = expect_single(inputs)?;
    let input_dims = static_dims(&input.spec.shape)?;
    if spec.perm.len() != input_dims.len() {
        return Err(BackendError::execution("transpose rank mismatch"));
    }
    match &input.data {
        TensorData::F32(values) => {
            let values = values.as_ref();
            let out_dims = static_dims(&output.shape)?;
            let mut result = vec![0.0f32; out_dims.iter().product()];
            let input_strides = compute_strides(&input_dims);
            for (idx, slot) in result.iter_mut().enumerate() {
                let out_coord = unravel_index(idx, &out_dims);
                let mut in_index = 0usize;
                for (out_axis, &out_c) in out_coord.iter().enumerate() {
                    let in_axis = spec.perm[out_axis];
                    in_index += out_c * input_strides[in_axis];
                }
                *slot = values[in_index];
            }
            Ok(CpuTensor {
                spec: output.clone(),
                data: TensorData::F32(Arc::from(result)),
            })
        }
        _ => Err(BackendError::execution(
            "transpose only supports f32 tensors",
        )),
    }
}

fn op_broadcast_to(
    inputs: &[CpuTensor],
    output: &TensorSpec,
    spec: &BroadcastToSpec,
) -> BackendResult<CpuTensor> {
    let input = expect_single(inputs)?;
    let input_dims = static_dims(&input.spec.shape)?;
    let out_dims = static_dims(&output.shape)?;
    let spec_dims = static_dims(&spec.result_shape)?;
    if out_dims != spec_dims {
        return Err(BackendError::execution(
            "broadcast_to result shape mismatch",
        ));
    }
    if out_dims.len() < input_dims.len() {
        return Err(BackendError::execution(
            "broadcast_to result rank must be >= operand rank",
        ));
    }
    let rank_diff = out_dims.len() - input_dims.len();
    for (axis, &dim) in input_dims.iter().enumerate() {
        let out_dim = out_dims[rank_diff + axis];
        if dim != 1 && dim != out_dim {
            return Err(BackendError::execution("broadcast_to dim mismatch"));
        }
    }
    let out_len: usize = out_dims.iter().product();
    match &input.data {
        TensorData::F32(values) => Ok(CpuTensor {
            spec: output.clone(),
            data: TensorData::F32(Arc::from(broadcast_to_f32(
                values.as_ref(),
                &input_dims,
                &out_dims,
                out_len,
            ))),
        }),
        TensorData::Si32(values) => Ok(CpuTensor {
            spec: output.clone(),
            data: TensorData::Si32(Arc::from(broadcast_to_i32(
                values.as_ref(),
                &input_dims,
                &out_dims,
                out_len,
            ))),
        }),
        TensorData::Bool(values) => Ok(CpuTensor {
            spec: output.clone(),
            data: TensorData::Bool(Arc::from(broadcast_to_u8(
                values.as_ref(),
                &input_dims,
                &out_dims,
                out_len,
            ))),
        }),
    }
}

fn broadcast_to_f32(
    input: &[f32],
    input_dims: &[usize],
    out_dims: &[usize],
    out_len: usize,
) -> Vec<f32> {
    if out_len == 0 {
        return Vec::new();
    }
    if input_dims == out_dims {
        return input.to_vec();
    }
    let rank_diff = out_dims.len().saturating_sub(input_dims.len());
    let mut aligned_in_dims = vec![1usize; out_dims.len()];
    aligned_in_dims[rank_diff..].copy_from_slice(input_dims);
    let in_strides = compute_strides(&aligned_in_dims);
    let out_strides = compute_strides(out_dims);
    let mut out = vec![0.0f32; out_len];
    broadcast_rec_f32(
        &mut out,
        input,
        0,
        0,
        0,
        out_dims,
        &aligned_in_dims,
        &out_strides,
        &in_strides,
    );
    out
}

#[allow(clippy::too_many_arguments)]
fn broadcast_rec_f32(
    out: &mut [f32],
    input: &[f32],
    axis: usize,
    out_offset: usize,
    in_offset: usize,
    out_dims: &[usize],
    in_dims: &[usize],
    out_strides: &[usize],
    in_strides: &[usize],
) {
    let rank = out_dims.len();
    if axis >= rank {
        out[out_offset] = input[in_offset];
        return;
    }

    if in_dims[axis..] == out_dims[axis..] {
        let block_len: usize = out_dims[axis..].iter().product();
        out[out_offset..out_offset + block_len]
            .copy_from_slice(&input[in_offset..in_offset + block_len]);
        return;
    }

    if in_dims[axis..].iter().all(|&dim| dim == 1) {
        let block_len: usize = out_dims[axis..].iter().product();
        out[out_offset..out_offset + block_len].fill(input[in_offset]);
        return;
    }

    let out_dim = out_dims[axis];
    let in_dim = in_dims[axis];
    let out_step = out_strides[axis];
    if in_dim == 1 {
        for i in 0..out_dim {
            broadcast_rec_f32(
                out,
                input,
                axis + 1,
                out_offset + i * out_step,
                in_offset,
                out_dims,
                in_dims,
                out_strides,
                in_strides,
            );
        }
    } else {
        let in_step = in_strides[axis];
        for i in 0..out_dim {
            broadcast_rec_f32(
                out,
                input,
                axis + 1,
                out_offset + i * out_step,
                in_offset + i * in_step,
                out_dims,
                in_dims,
                out_strides,
                in_strides,
            );
        }
    }
}

fn broadcast_to_i32(
    input: &[i32],
    input_dims: &[usize],
    out_dims: &[usize],
    out_len: usize,
) -> Vec<i32> {
    if out_len == 0 {
        return Vec::new();
    }
    if input_dims == out_dims {
        return input.to_vec();
    }
    let rank_diff = out_dims.len().saturating_sub(input_dims.len());
    let mut aligned_in_dims = vec![1usize; out_dims.len()];
    aligned_in_dims[rank_diff..].copy_from_slice(input_dims);
    let in_strides = compute_strides(&aligned_in_dims);
    let out_strides = compute_strides(out_dims);
    let mut out = vec![0i32; out_len];
    broadcast_rec_i32(
        &mut out,
        input,
        0,
        0,
        0,
        out_dims,
        &aligned_in_dims,
        &out_strides,
        &in_strides,
    );
    out
}

#[allow(clippy::too_many_arguments)]
fn broadcast_rec_i32(
    out: &mut [i32],
    input: &[i32],
    axis: usize,
    out_offset: usize,
    in_offset: usize,
    out_dims: &[usize],
    in_dims: &[usize],
    out_strides: &[usize],
    in_strides: &[usize],
) {
    let rank = out_dims.len();
    if axis >= rank {
        out[out_offset] = input[in_offset];
        return;
    }

    if in_dims[axis..] == out_dims[axis..] {
        let block_len: usize = out_dims[axis..].iter().product();
        out[out_offset..out_offset + block_len]
            .copy_from_slice(&input[in_offset..in_offset + block_len]);
        return;
    }

    if in_dims[axis..].iter().all(|&dim| dim == 1) {
        let block_len: usize = out_dims[axis..].iter().product();
        out[out_offset..out_offset + block_len].fill(input[in_offset]);
        return;
    }

    let out_dim = out_dims[axis];
    let in_dim = in_dims[axis];
    let out_step = out_strides[axis];
    if in_dim == 1 {
        for i in 0..out_dim {
            broadcast_rec_i32(
                out,
                input,
                axis + 1,
                out_offset + i * out_step,
                in_offset,
                out_dims,
                in_dims,
                out_strides,
                in_strides,
            );
        }
    } else {
        let in_step = in_strides[axis];
        for i in 0..out_dim {
            broadcast_rec_i32(
                out,
                input,
                axis + 1,
                out_offset + i * out_step,
                in_offset + i * in_step,
                out_dims,
                in_dims,
                out_strides,
                in_strides,
            );
        }
    }
}

fn broadcast_to_u8(
    input: &[u8],
    input_dims: &[usize],
    out_dims: &[usize],
    out_len: usize,
) -> Vec<u8> {
    if out_len == 0 {
        return Vec::new();
    }
    if input_dims == out_dims {
        return input.to_vec();
    }
    let rank_diff = out_dims.len().saturating_sub(input_dims.len());
    let mut aligned_in_dims = vec![1usize; out_dims.len()];
    aligned_in_dims[rank_diff..].copy_from_slice(input_dims);
    let in_strides = compute_strides(&aligned_in_dims);
    let out_strides = compute_strides(out_dims);
    let mut out = vec![0u8; out_len];
    broadcast_rec_u8(
        &mut out,
        input,
        0,
        0,
        0,
        out_dims,
        &aligned_in_dims,
        &out_strides,
        &in_strides,
    );
    out
}

#[allow(clippy::too_many_arguments)]
fn broadcast_rec_u8(
    out: &mut [u8],
    input: &[u8],
    axis: usize,
    out_offset: usize,
    in_offset: usize,
    out_dims: &[usize],
    in_dims: &[usize],
    out_strides: &[usize],
    in_strides: &[usize],
) {
    let rank = out_dims.len();
    if axis >= rank {
        out[out_offset] = input[in_offset];
        return;
    }

    if in_dims[axis..] == out_dims[axis..] {
        let block_len: usize = out_dims[axis..].iter().product();
        out[out_offset..out_offset + block_len]
            .copy_from_slice(&input[in_offset..in_offset + block_len]);
        return;
    }

    if in_dims[axis..].iter().all(|&dim| dim == 1) {
        let block_len: usize = out_dims[axis..].iter().product();
        out[out_offset..out_offset + block_len].fill(input[in_offset]);
        return;
    }

    let out_dim = out_dims[axis];
    let in_dim = in_dims[axis];
    let out_step = out_strides[axis];
    if in_dim == 1 {
        for i in 0..out_dim {
            broadcast_rec_u8(
                out,
                input,
                axis + 1,
                out_offset + i * out_step,
                in_offset,
                out_dims,
                in_dims,
                out_strides,
                in_strides,
            );
        }
    } else {
        let in_step = in_strides[axis];
        for i in 0..out_dim {
            broadcast_rec_u8(
                out,
                input,
                axis + 1,
                out_offset + i * out_step,
                in_offset + i * in_step,
                out_dims,
                in_dims,
                out_strides,
                in_strides,
            );
        }
    }
}

fn f32_to_i32_trunc_saturating(value: f32) -> i32 {
    if value.is_nan() {
        return 0;
    }
    let truncated = value.trunc();
    if truncated > i32::MAX as f32 {
        i32::MAX
    } else if truncated < i32::MIN as f32 {
        i32::MIN
    } else {
        truncated as i32
    }
}

fn op_cast(inputs: &[CpuTensor], output: &TensorSpec, spec: &CastSpec) -> BackendResult<CpuTensor> {
    let input = expect_single(inputs)?;
    if output.dtype != spec.dtype {
        return Err(BackendError::execution("cast output dtype mismatch"));
    }
    if static_dims(&input.spec.shape)? != static_dims(&output.shape)? {
        return Err(BackendError::execution("cast shape mismatch"));
    }

    match (&input.data, output.dtype) {
        (TensorData::F32(values), DType::F32) => Ok(CpuTensor {
            spec: output.clone(),
            data: TensorData::F32(values.clone()),
        }),
        (TensorData::Si32(values), DType::Si32) => Ok(CpuTensor {
            spec: output.clone(),
            data: TensorData::Si32(values.clone()),
        }),
        (TensorData::Bool(values), DType::I1) => Ok(CpuTensor {
            spec: output.clone(),
            data: TensorData::Bool(values.clone()),
        }),
        (TensorData::Si32(values), DType::F32) => {
            let mut result = Vec::with_capacity(values.len());
            for &value in values.iter() {
                result.push(value as f32);
            }
            Ok(CpuTensor {
                spec: output.clone(),
                data: TensorData::F32(Arc::from(result)),
            })
        }
        (TensorData::F32(values), DType::Si32) => {
            let mut result = Vec::with_capacity(values.len());
            for &value in values.iter() {
                result.push(f32_to_i32_trunc_saturating(value));
            }
            Ok(CpuTensor {
                spec: output.clone(),
                data: TensorData::Si32(Arc::from(result)),
            })
        }
        (TensorData::Bool(values), DType::Si32) => {
            let mut result = Vec::with_capacity(values.len());
            for &value in values.iter() {
                result.push(if value == 0 { 0 } else { 1 });
            }
            Ok(CpuTensor {
                spec: output.clone(),
                data: TensorData::Si32(Arc::from(result)),
            })
        }
        (TensorData::Si32(values), DType::I1) => {
            let mut result = Vec::with_capacity(values.len());
            for &value in values.iter() {
                result.push((value != 0) as u8);
            }
            Ok(CpuTensor {
                spec: output.clone(),
                data: TensorData::Bool(Arc::from(result)),
            })
        }
        (TensorData::Bool(values), DType::F32) => {
            let mut result = Vec::with_capacity(values.len());
            for &value in values.iter() {
                result.push(if value == 0 { 0.0 } else { 1.0 });
            }
            Ok(CpuTensor {
                spec: output.clone(),
                data: TensorData::F32(Arc::from(result)),
            })
        }
        (TensorData::F32(values), DType::I1) => {
            let mut result = Vec::with_capacity(values.len());
            for &value in values.iter() {
                result.push((value != 0.0) as u8);
            }
            Ok(CpuTensor {
                spec: output.clone(),
                data: TensorData::Bool(Arc::from(result)),
            })
        }
        _ => Err(BackendError::unimplemented(
            "cast",
            format!("cast not supported for dtype {:?}", output.dtype),
        )),
    }
}

fn op_take(inputs: &[CpuTensor], output: &TensorSpec) -> BackendResult<CpuTensor> {
    if inputs.len() != 2 {
        return Err(BackendError::execution("take expects two operands"));
    }
    let data = &inputs[0];
    let indices = &inputs[1];

    let data_dims = static_dims(&data.spec.shape)?;
    if data_dims.is_empty() {
        return Err(BackendError::execution("take requires rank >= 1"));
    }
    let indices_dims = static_dims(&indices.spec.shape)?;
    let out_dims = static_dims(&output.shape)?;

    let vocab = data_dims[0];
    let inner = data_dims.iter().skip(1).product::<usize>();
    let indices_len = indices_dims.iter().product::<usize>();
    let expected_out: usize = indices_len * inner;
    if out_dims.iter().product::<usize>() != expected_out {
        return Err(BackendError::execution("take output shape mismatch"));
    }

    let indices_values = match &indices.data {
        TensorData::Si32(values) => values.as_ref(),
        _ => return Err(BackendError::execution("take indices must be si32")),
    };

    match &data.data {
        TensorData::F32(values) => {
            let values = values.as_ref();
            let mut result = vec![0.0f32; expected_out];
            for (i, &index) in indices_values.iter().enumerate() {
                if index < 0 || index as usize >= vocab {
                    return Err(BackendError::execution("take index out of bounds"));
                }
                let idx = index as usize;
                let src_offset = idx * inner;
                let dst_offset = i * inner;
                result[dst_offset..dst_offset + inner]
                    .copy_from_slice(&values[src_offset..src_offset + inner]);
            }
            Ok(CpuTensor {
                spec: output.clone(),
                data: TensorData::F32(Arc::from(result)),
            })
        }
        _ => Err(BackendError::execution("take only supports f32 tensors")),
    }
}

fn op_elementwise_binary(
    inputs: &[CpuTensor],
    output: &TensorSpec,
    op: ElementwiseBinaryOp,
) -> BackendResult<CpuTensor> {
    if inputs.len() != 2 {
        return Err(BackendError::execution(
            "elementwise binary expects 2 inputs",
        ));
    }
    let lhs = &inputs[0];
    let rhs = &inputs[1];
    match (&lhs.data, &rhs.data) {
        (TensorData::F32(a), TensorData::F32(b)) => {
            if a.len() != b.len() {
                return Err(BackendError::execution("elementwise size mismatch"));
            }
            let mut result = Vec::with_capacity(a.len());
            for (x, y) in a.iter().zip(b.iter()) {
                let value = match op {
                    ElementwiseBinaryOp::Add => x + y,
                    ElementwiseBinaryOp::Sub => x - y,
                    ElementwiseBinaryOp::Mul => x * y,
                    ElementwiseBinaryOp::Div => x / y,
                    ElementwiseBinaryOp::Maximum => x.max(*y),
                    ElementwiseBinaryOp::Minimum => x.min(*y),
                };
                result.push(value);
            }
            Ok(CpuTensor {
                spec: output.clone(),
                data: TensorData::F32(Arc::from(result)),
            })
        }
        _ => Err(BackendError::execution(
            "elementwise binary only supports f32 tensors",
        )),
    }
}

fn op_elementwise_unary(
    inputs: &[CpuTensor],
    output: &TensorSpec,
    op: ElementwiseUnaryOp,
) -> BackendResult<CpuTensor> {
    let input = expect_single(inputs)?;
    let values = match &input.data {
        TensorData::F32(values) => values.as_ref(),
        _ => {
            return Err(BackendError::execution(
                "elementwise unary only supports f32 tensors",
            ));
        }
    };

    let result: Vec<f32> = match op {
        ElementwiseUnaryOp::Neg => values.iter().map(|&x| -x).collect(),
        ElementwiseUnaryOp::Abs => values.iter().map(|&x| x.abs()).collect(),
        ElementwiseUnaryOp::Exp => values.iter().map(|&x| x.exp()).collect(),
        ElementwiseUnaryOp::Log => values.iter().map(|&x| x.ln()).collect(),
        ElementwiseUnaryOp::Tanh => values.iter().map(|&x| x.tanh()).collect(),
        ElementwiseUnaryOp::Erf => values.iter().map(|&x| libm::erff(x)).collect(),
        ElementwiseUnaryOp::Rsqrt => values.iter().map(|&x| 1.0 / x.sqrt()).collect(),
        ElementwiseUnaryOp::Reciprocal => values.iter().map(|&x| 1.0 / x).collect(),
    };

    Ok(CpuTensor {
        spec: output.clone(),
        data: TensorData::F32(Arc::from(result)),
    })
}

fn op_reduce(
    inputs: &[CpuTensor],
    output: &TensorSpec,
    spec: &ReduceSpec,
) -> BackendResult<CpuTensor> {
    let input = expect_single(inputs)?;
    if let Some(accum_dtype) = spec.accum_dtype {
        if accum_dtype != DType::F32 {
            return Err(BackendError::unimplemented(
                "reduce",
                "accum_dtype override not supported",
            ));
        }
    }
    if let Some(out_dtype) = spec.out_dtype {
        if out_dtype != output.dtype {
            return Err(BackendError::execution("reduce out_dtype mismatch"));
        }
        if out_dtype != DType::F32 {
            return Err(BackendError::unimplemented(
                "reduce",
                "out_dtype not supported",
            ));
        }
    }
    if spec.axes.len() != 1 {
        return Err(BackendError::execution(
            "reduce currently supports single axis",
        ));
    }
    if !spec.keepdims {
        return Err(BackendError::execution("reduce requires keepdims=true"));
    }
    let axis = spec.axes[0];
    match &input.data {
        TensorData::F32(values) => {
            let values = values.as_ref();
            let dims = static_dims(&input.spec.shape)?;
            if axis >= dims.len() {
                return Err(BackendError::execution("reduce axis out of range"));
            }
            let axis_len = dims[axis];
            let strides = compute_strides(&dims);
            let inner = strides[axis];
            let outer = values.len() / (axis_len * inner);
            let mut result = vec![0.0f32; outer * inner];
            for outer_idx in 0..outer {
                for inner_idx in 0..inner {
                    let base = outer_idx * axis_len * inner + inner_idx;
                    let mut acc = match spec.kind {
                        ReduceKind::Sum => 0.0f32,
                        ReduceKind::Max => f32::NEG_INFINITY,
                        ReduceKind::Min => f32::INFINITY,
                    };
                    for a in 0..axis_len {
                        let idx = base + a * inner;
                        let value = values[idx];
                        acc = match spec.kind {
                            ReduceKind::Sum => acc + value,
                            ReduceKind::Max => acc.max(value),
                            ReduceKind::Min => acc.min(value),
                        };
                    }
                    result[outer_idx * inner + inner_idx] = acc;
                }
            }
            Ok(CpuTensor {
                spec: output.clone(),
                data: TensorData::F32(Arc::from(result)),
            })
        }
        _ => Err(BackendError::execution("reduce only supports f32 tensors")),
    }
}

fn op_extract_patches(
    inputs: &[CpuTensor],
    output: &TensorSpec,
    spec: &ExtractPatchesSpec,
) -> BackendResult<CpuTensor> {
    let input = expect_single(inputs)?;
    if input.spec.dtype != DType::F32 || output.dtype != DType::F32 {
        return Err(BackendError::execution(
            "extract_patches only supports f32 tensors",
        ));
    }

    let pad_value = match spec.pad_value {
        Literal::Float(value) => value as f32,
        _ => {
            return Err(BackendError::unimplemented(
                "extract_patches",
                "pad_value only supports float literals",
            ));
        }
    };

    let dims = static_dims(&input.spec.shape)?;
    if dims.len() != 4 {
        return Err(BackendError::unimplemented(
            "extract_patches",
            "only rank-4 NHWC inputs are supported",
        ));
    }
    let (n, h, w, c) = (dims[0], dims[1], dims[2], dims[3]);
    if spec.window.len() != 2
        || spec.strides.len() != 2
        || spec.dilation.len() != 2
        || spec.padding.len() != 2
    {
        return Err(BackendError::execution(
            "extract_patches expects 2D window/stride/dilation/padding",
        ));
    }

    let (k_h, k_w) = (spec.window[0], spec.window[1]);
    let (s_h, s_w) = (spec.strides[0], spec.strides[1]);
    let (d_h, d_w) = (spec.dilation[0], spec.dilation[1]);
    let (pad_top, pad_bottom) = spec.padding[0];
    let (pad_left, pad_right) = spec.padding[1];

    if k_h == 0 || k_w == 0 || s_h == 0 || s_w == 0 || d_h == 0 || d_w == 0 {
        return Err(BackendError::execution(
            "extract_patches window/stride/dilation must be > 0",
        ));
    }

    let eff_kh = (k_h - 1)
        .checked_mul(d_h)
        .and_then(|v| v.checked_add(1))
        .ok_or_else(|| BackendError::execution("extract_patches kernel overflow"))?;
    let eff_kw = (k_w - 1)
        .checked_mul(d_w)
        .and_then(|v| v.checked_add(1))
        .ok_or_else(|| BackendError::execution("extract_patches kernel overflow"))?;

    let padded_h = h
        .checked_add(pad_top)
        .and_then(|v| v.checked_add(pad_bottom))
        .ok_or_else(|| BackendError::execution("extract_patches padded height overflow"))?;
    let padded_w = w
        .checked_add(pad_left)
        .and_then(|v| v.checked_add(pad_right))
        .ok_or_else(|| BackendError::execution("extract_patches padded width overflow"))?;

    if padded_h < eff_kh || padded_w < eff_kw {
        return Err(BackendError::execution(
            "extract_patches window exceeds padded input",
        ));
    }

    let out_h = (padded_h - eff_kh) / s_h + 1;
    let out_w = (padded_w - eff_kw) / s_w + 1;
    let patch_dim = k_h
        .checked_mul(k_w)
        .and_then(|v| v.checked_mul(c))
        .ok_or_else(|| BackendError::execution("extract_patches patch size overflow"))?;

    let out_dims = static_dims(&output.shape)?;
    if out_dims.as_slice() != [n, out_h, out_w, patch_dim] {
        return Err(BackendError::execution(format!(
            "extract_patches output shape mismatch: expected [{n}x{out_h}x{out_w}x{patch_dim}], got {:?}",
            out_dims
        )));
    }

    let values = match &input.data {
        TensorData::F32(values) => values.as_ref(),
        _ => unreachable!("dtype guard above"),
    };

    let out_len = out_dims.iter().product::<usize>();
    let mut result = vec![pad_value; out_len];

    for n_idx in 0..n {
        for oh in 0..out_h {
            let base_h = oh as isize * s_h as isize - pad_top as isize;
            for ow in 0..out_w {
                let base_w = ow as isize * s_w as isize - pad_left as isize;
                let out_base = ((n_idx * out_h + oh) * out_w + ow) * patch_dim;

                for kh in 0..k_h {
                    let in_h = base_h + kh as isize * d_h as isize;
                    if in_h < 0 || in_h >= h as isize {
                        continue;
                    }
                    for kw in 0..k_w {
                        let in_w = base_w + kw as isize * d_w as isize;
                        if in_w < 0 || in_w >= w as isize {
                            continue;
                        }
                        let in_offset = ((n_idx * h + in_h as usize) * w + in_w as usize) * c;
                        let patch_offset = (kh * k_w + kw) * c;
                        let dst = out_base + patch_offset;
                        result[dst..dst + c].copy_from_slice(&values[in_offset..in_offset + c]);
                    }
                }
            }
        }
    }

    Ok(CpuTensor {
        spec: output.clone(),
        data: TensorData::F32(Arc::from(result)),
    })
}

fn op_reduce_window(
    inputs: &[CpuTensor],
    output: &TensorSpec,
    spec: &ReduceWindowSpec,
) -> BackendResult<CpuTensor> {
    let input = expect_single(inputs)?;
    if input.spec.dtype != DType::F32 || output.dtype != DType::F32 {
        return Err(BackendError::execution(
            "reduce_window only supports f32 tensors",
        ));
    }
    if let Some(accum) = spec.accum_dtype {
        if accum != DType::F32 {
            return Err(BackendError::unimplemented(
                "reduce_window",
                "accum_dtype override not supported",
            ));
        }
    }

    let in_dims = static_dims(&input.spec.shape)?;
    if in_dims.len() != 4 {
        return Err(BackendError::unimplemented(
            "reduce_window",
            "only rank-4 tensors are supported",
        ));
    }

    if spec.window_dims.len() != 4
        || spec.strides.len() != 4
        || spec.padding.len() != 4
        || spec.base_dilation.len() != 4
        || spec.window_dilation.len() != 4
    {
        return Err(BackendError::execution(
            "reduce_window expects window/stride/padding/dilation rank 4",
        ));
    }

    let out_dims = static_dims(&output.shape)?;
    if out_dims.len() != 4 {
        return Err(BackendError::execution(
            "reduce_window output must have rank 4",
        ));
    }

    let identity = match spec.reduce {
        ReduceKind::Sum => 0.0f32,
        ReduceKind::Max => f32::NEG_INFINITY,
        ReduceKind::Min => f32::INFINITY,
    };

    let values = match &input.data {
        TensorData::F32(values) => values.as_ref(),
        _ => unreachable!("dtype guard above"),
    };

    let out_len = out_dims.iter().product::<usize>();
    let mut result = vec![identity; out_len];

    let (n_in, h_in, w_in, c_in) = (in_dims[0], in_dims[1], in_dims[2], in_dims[3]);
    let (n_out, h_out, w_out, c_out) = (out_dims[0], out_dims[1], out_dims[2], out_dims[3]);

    if c_in != c_out {
        return Err(BackendError::execution(
            "reduce_window expects channels dimension preserved",
        ));
    }

    let strides = compute_strides(&in_dims);

    for n_idx in 0..n_out {
        let start_n = n_idx as isize * spec.strides[0] as isize - spec.padding[0].0 as isize;
        for oh in 0..h_out {
            let start_h = oh as isize * spec.strides[1] as isize - spec.padding[1].0 as isize;
            for ow in 0..w_out {
                let start_w = ow as isize * spec.strides[2] as isize - spec.padding[2].0 as isize;
                for c_idx in 0..c_out {
                    let start_c =
                        c_idx as isize * spec.strides[3] as isize - spec.padding[3].0 as isize;

                    let mut acc = identity;
                    for wn in 0..spec.window_dims[0] {
                        let pos_n = start_n + wn as isize * spec.window_dilation[0] as isize;
                        let n_d = spec.base_dilation[0] as isize;
                        if pos_n < 0 || pos_n % n_d != 0 {
                            continue;
                        }
                        let in_n = pos_n / n_d;
                        if in_n < 0 || in_n >= n_in as isize {
                            continue;
                        }

                        for wh in 0..spec.window_dims[1] {
                            let pos_h = start_h + wh as isize * spec.window_dilation[1] as isize;
                            let h_d = spec.base_dilation[1] as isize;
                            if pos_h < 0 || pos_h % h_d != 0 {
                                continue;
                            }
                            let in_h = pos_h / h_d;
                            if in_h < 0 || in_h >= h_in as isize {
                                continue;
                            }

                            for ww in 0..spec.window_dims[2] {
                                let pos_w =
                                    start_w + ww as isize * spec.window_dilation[2] as isize;
                                let w_d = spec.base_dilation[2] as isize;
                                if pos_w < 0 || pos_w % w_d != 0 {
                                    continue;
                                }
                                let in_w = pos_w / w_d;
                                if in_w < 0 || in_w >= w_in as isize {
                                    continue;
                                }

                                for wc in 0..spec.window_dims[3] {
                                    let pos_c =
                                        start_c + wc as isize * spec.window_dilation[3] as isize;
                                    let c_d = spec.base_dilation[3] as isize;
                                    if pos_c < 0 || pos_c % c_d != 0 {
                                        continue;
                                    }
                                    let in_c = pos_c / c_d;
                                    if in_c < 0 || in_c >= c_in as isize {
                                        continue;
                                    }

                                    let linear = (in_n as usize) * strides[0]
                                        + (in_h as usize) * strides[1]
                                        + (in_w as usize) * strides[2]
                                        + (in_c as usize) * strides[3];
                                    let v = values[linear];
                                    acc = match spec.reduce {
                                        ReduceKind::Sum => acc + v,
                                        ReduceKind::Max => acc.max(v),
                                        ReduceKind::Min => acc.min(v),
                                    };
                                }
                            }
                        }
                    }

                    let out_index = ((n_idx * h_out + oh) * w_out + ow) * c_out + c_idx;
                    result[out_index] = acc;
                }
            }
        }
    }

    Ok(CpuTensor {
        spec: output.clone(),
        data: TensorData::F32(Arc::from(result)),
    })
}

fn op_dot_general(
    inputs: &[CpuTensor],
    output: &TensorSpec,
    spec: &DotGeneralSpec,
) -> BackendResult<CpuTensor> {
    if inputs.len() != 2 {
        return Err(BackendError::execution("dot_general expects two inputs"));
    }
    if let Some(accum_dtype) = spec.accum_dtype {
        if accum_dtype != DType::F32 {
            return Err(BackendError::unimplemented(
                "dot_general",
                "accum_dtype override not supported",
            ));
        }
    }
    if let Some(out_dtype) = spec.out_dtype {
        if out_dtype != output.dtype {
            return Err(BackendError::execution("dot_general out_dtype mismatch"));
        }
        if out_dtype != DType::F32 {
            return Err(BackendError::unimplemented(
                "dot_general",
                "out_dtype not supported",
            ));
        }
    }
    let lhs = &inputs[0];
    let rhs = &inputs[1];
    match (&lhs.data, &rhs.data) {
        (TensorData::F32(lhs_values), TensorData::F32(rhs_values)) => {
            if output.dtype != DType::F32 {
                return Err(BackendError::execution(
                    "dot_general only supports f32 output",
                ));
            }
            let lhs_values = lhs_values.as_ref();
            let rhs_values = rhs_values.as_ref();
            let lhs_dims = static_dims(&lhs.spec.shape)?;
            let rhs_dims = static_dims(&rhs.spec.shape)?;
            let lhs_strides = compute_strides(&lhs_dims);
            let rhs_strides = compute_strides(&rhs_dims);

            let mut lhs_batch_axes = spec.batch_lhs.clone();
            let mut rhs_batch_axes = spec.batch_rhs.clone();
            lhs_batch_axes.sort_unstable();
            rhs_batch_axes.sort_unstable();

            let lhs_contract_axes = spec.contract_lhs.clone();
            let rhs_contract_axes = spec.contract_rhs.clone();

            let lhs_free_axes: Vec<usize> = (0..lhs_dims.len())
                .filter(|ax| !lhs_batch_axes.contains(ax) && !lhs_contract_axes.contains(ax))
                .collect();
            let rhs_free_axes: Vec<usize> = (0..rhs_dims.len())
                .filter(|ax| !rhs_batch_axes.contains(ax) && !rhs_contract_axes.contains(ax))
                .collect();

            let batch_shape: Vec<usize> = lhs_batch_axes.iter().map(|&ax| lhs_dims[ax]).collect();
            let lhs_free_shape: Vec<usize> = lhs_free_axes.iter().map(|&ax| lhs_dims[ax]).collect();
            let rhs_free_shape: Vec<usize> = rhs_free_axes.iter().map(|&ax| rhs_dims[ax]).collect();
            let contract_shape: Vec<usize> =
                lhs_contract_axes.iter().map(|&ax| lhs_dims[ax]).collect();

            let mut output_data = vec![0.0f32; element_count(&output.shape)?];
            let mut out_index = 0usize;

            for batch_index in MultiIndex::new(&batch_shape) {
                for lhs_free_index in MultiIndex::new(&lhs_free_shape) {
                    for rhs_free_index in MultiIndex::new(&rhs_free_shape) {
                        let mut sum = 0.0f32;
                        for contract_index in MultiIndex::new(&contract_shape) {
                            let lhs_idx = build_index(
                                &lhs_strides,
                                &lhs_batch_axes,
                                &lhs_free_axes,
                                &lhs_contract_axes,
                                &batch_index,
                                &lhs_free_index,
                                &contract_index,
                            );
                            let rhs_idx = build_index(
                                &rhs_strides,
                                &rhs_batch_axes,
                                &rhs_free_axes,
                                &rhs_contract_axes,
                                &batch_index,
                                &rhs_free_index,
                                &contract_index,
                            );
                            sum += lhs_values[lhs_idx] * rhs_values[rhs_idx];
                        }
                        output_data[out_index] = sum;
                        out_index += 1;
                    }
                }
            }

            Ok(CpuTensor {
                spec: output.clone(),
                data: TensorData::F32(Arc::from(output_data)),
            })
        }
        _ => Err(BackendError::execution(
            "dot_general only supports f32 tensors",
        )),
    }
}

fn op_compare(
    inputs: &[CpuTensor],
    output: &TensorSpec,
    spec: &gpt_rs::backend::spec::CompareSpec,
) -> BackendResult<CpuTensor> {
    if inputs.len() != 2 {
        return Err(BackendError::execution("compare expects two inputs"));
    }
    let lhs = &inputs[0];
    let rhs = &inputs[1];
    match (&lhs.data, &rhs.data) {
        (TensorData::Si32(a), TensorData::Si32(b)) => {
            let mut result = Vec::with_capacity(a.len());
            for (&x, &y) in a.iter().zip(b.iter()) {
                let flag = match spec.op {
                    ComparisonOp::Less => x < y,
                    ComparisonOp::LessEqual => x <= y,
                    ComparisonOp::Equal => x == y,
                    ComparisonOp::GreaterEqual => x >= y,
                    ComparisonOp::Greater => x > y,
                    ComparisonOp::NotEqual => x != y,
                };
                result.push(flag as u8);
            }
            Ok(CpuTensor {
                spec: output.clone(),
                data: TensorData::Bool(Arc::from(result)),
            })
        }
        _ => Err(BackendError::execution(
            "compare only supports si32 tensors",
        )),
    }
}

fn op_select(inputs: &[CpuTensor], output: &TensorSpec) -> BackendResult<CpuTensor> {
    if inputs.len() != 3 {
        return Err(BackendError::execution("select expects three operands"));
    }
    match (&inputs[0].data, &inputs[1].data, &inputs[2].data) {
        (TensorData::Bool(pred), TensorData::F32(on_true), TensorData::F32(on_false)) => {
            let mut result = Vec::with_capacity(on_true.len());
            for ((&pred_flag, &true_val), &false_val) in
                pred.iter().zip(on_true.iter()).zip(on_false.iter())
            {
                result.push(if pred_flag != 0 { true_val } else { false_val });
            }
            Ok(CpuTensor {
                spec: output.clone(),
                data: TensorData::F32(Arc::from(result)),
            })
        }
        _ => Err(BackendError::execution(
            "select dtype combination unsupported",
        )),
    }
}

fn op_concat(
    inputs: &[CpuTensor],
    output: &TensorSpec,
    spec: &ConcatSpec,
) -> BackendResult<CpuTensor> {
    if inputs.is_empty() {
        return Err(BackendError::execution("concat expects at least one input"));
    }

    let out_dims = static_dims(&output.shape)?;
    let rank = out_dims.len();
    let axis = {
        let raw = spec.axis;
        let axis = if raw < 0 {
            let adj = rank as isize + raw;
            if adj < 0 {
                return Err(BackendError::execution("concat axis out of range"));
            }
            adj as usize
        } else {
            raw as usize
        };
        if axis >= rank {
            return Err(BackendError::execution("concat axis out of range"));
        }
        axis
    };

    let axis_inner = out_dims.iter().skip(axis + 1).product::<usize>();
    let outer = out_dims.iter().take(axis).product::<usize>();

    match output.dtype {
        DType::F32 => {
            let mut inputs_info = Vec::with_capacity(inputs.len());
            let mut axis_total = 0usize;
            for tensor in inputs {
                if tensor.spec.dtype != output.dtype {
                    return Err(BackendError::execution("concat requires matching dtypes"));
                }
                let dims = static_dims(&tensor.spec.shape)?;
                if dims.len() != rank {
                    return Err(BackendError::execution("concat rank mismatch"));
                }
                for (idx, (&dim, &out_dim)) in dims.iter().zip(out_dims.iter()).enumerate() {
                    if idx != axis && dim != out_dim {
                        return Err(BackendError::execution("concat dimension mismatch"));
                    }
                }
                axis_total += dims[axis];
                let data = match &tensor.data {
                    TensorData::F32(values) => values.clone(),
                    _ => return Err(BackendError::execution("concat only supports f32 tensors")),
                };
                inputs_info.push((dims[axis], data));
            }

            if axis_total != out_dims[axis] {
                return Err(BackendError::execution(
                    "concat inputs do not match output axis length",
                ));
            }

            let result_len: usize = out_dims.iter().product();
            let mut result = vec![0.0f32; result_len];
            let stride_outer = out_dims[axis] * axis_inner;

            for outer_idx in 0..outer {
                let mut dst_offset = outer_idx * stride_outer;
                for (axis_dim, data_arc) in &inputs_info {
                    let slice = data_arc.as_ref();
                    let chunk = (*axis_dim) * axis_inner;
                    let src_start = outer_idx * chunk;
                    let src_end = src_start + chunk;
                    result[dst_offset..dst_offset + chunk]
                        .copy_from_slice(&slice[src_start..src_end]);
                    dst_offset += chunk;
                }
            }

            Ok(CpuTensor {
                spec: output.clone(),
                data: TensorData::F32(Arc::from(result.into_boxed_slice())),
            })
        }
        _ => Err(BackendError::execution("concat only supports f32 tensors")),
    }
}

fn op_gather(
    inputs: &[CpuTensor],
    output: &TensorSpec,
    spec: &gpt_rs::backend::spec::GatherSpec,
) -> BackendResult<CpuTensor> {
    if inputs.len() != 2 {
        return Err(BackendError::execution("gather expects two operands"));
    }
    let data = &inputs[0];
    let indices = &inputs[1];
    if data.spec.dtype != output.dtype {
        return Err(BackendError::execution("gather output dtype mismatch"));
    }

    let data_dims = static_dims(&data.spec.shape)?;
    let indices_dims = static_dims(&indices.spec.shape)?;
    let out_dims = static_dims(&output.shape)?;
    if out_dims != indices_dims {
        return Err(BackendError::execution("gather output shape mismatch"));
    }
    if data_dims.len() != indices_dims.len() {
        return Err(BackendError::execution(
            "gather requires indices to have the same rank as the operand",
        ));
    }

    let axis = {
        let rank = data_dims.len();
        let mut axis = spec.axis;
        if axis < 0 {
            axis += rank as isize;
        }
        if axis < 0 || axis as usize >= rank {
            return Err(BackendError::execution("gather axis out of range"));
        }
        axis as usize
    };

    for dim in 0..data_dims.len() {
        if dim == axis {
            continue;
        }
        if indices_dims[dim] != data_dims[dim] {
            return Err(BackendError::execution("gather shape mismatch"));
        }
    }

    let index_values = match &indices.data {
        TensorData::Si32(values) => values.as_ref(),
        _ => return Err(BackendError::execution("gather indices must be si32")),
    };

    let data_strides = compute_strides(&data_dims);
    let out_len: usize = out_dims.iter().product();

    match &data.data {
        TensorData::F32(values) => {
            let values = values.as_ref();
            let mut result = vec![0.0f32; out_len];
            for (out_index, slot) in result.iter_mut().enumerate() {
                let mut coord = unravel_index(out_index, &out_dims);
                let index = index_values[out_index];
                if index < 0 || index as usize >= data_dims[axis] {
                    return Err(BackendError::execution("gather index out of bounds"));
                }
                coord[axis] = index as usize;
                let mut src_index = 0usize;
                for (dim, &c) in coord.iter().enumerate() {
                    src_index += c * data_strides[dim];
                }
                *slot = values[src_index];
            }
            Ok(CpuTensor {
                spec: output.clone(),
                data: TensorData::F32(Arc::from(result)),
            })
        }
        TensorData::Si32(values) => {
            let values = values.as_ref();
            let mut result = vec![0i32; out_len];
            for (out_index, slot) in result.iter_mut().enumerate() {
                let mut coord = unravel_index(out_index, &out_dims);
                let index = index_values[out_index];
                if index < 0 || index as usize >= data_dims[axis] {
                    return Err(BackendError::execution("gather index out of bounds"));
                }
                coord[axis] = index as usize;
                let mut src_index = 0usize;
                for (dim, &c) in coord.iter().enumerate() {
                    src_index += c * data_strides[dim];
                }
                *slot = values[src_index];
            }
            Ok(CpuTensor {
                spec: output.clone(),
                data: TensorData::Si32(Arc::from(result)),
            })
        }
        _ => Err(BackendError::execution(
            "gather only supports f32 and si32 tensors",
        )),
    }
}

fn op_iota(output: &TensorSpec, spec: &IotaSpec) -> BackendResult<CpuTensor> {
    let dims = static_dims(&spec.shape)?;
    let axis = spec.axis;
    if axis >= dims.len() {
        return Err(BackendError::execution("iota axis out of range"));
    }
    let len = element_count(&spec.shape)?;
    match spec.dtype {
        DType::Si32 => {
            let mut result = vec![0i32; len];
            for (idx, slot) in result.iter_mut().enumerate() {
                let coord = unravel_index(idx, &dims);
                *slot = coord[axis] as i32;
            }
            Ok(CpuTensor {
                spec: output.clone(),
                data: TensorData::Si32(Arc::from(result)),
            })
        }
        _ => Err(BackendError::execution("iota only supports si32 dtype")),
    }
}

fn expect_single(inputs: &[CpuTensor]) -> BackendResult<&CpuTensor> {
    if inputs.len() != 1 {
        Err(BackendError::execution("operation expects single input"))
    } else {
        Ok(&inputs[0])
    }
}

fn static_dims(shape: &Shape) -> BackendResult<Vec<usize>> {
    shape
        .dims()
        .iter()
        .map(|dim| match dim {
            Dimension::Static(v) => Ok(*v),
            Dimension::Dynamic(sym) => Err(BackendError::execution(format!(
                "dynamic dimension {} not supported at runtime",
                sym.as_str()
            ))),
        })
        .collect()
}

fn element_count(shape: &Shape) -> BackendResult<usize> {
    Ok(static_dims(shape)?.into_iter().product())
}

fn compute_strides(dims: &[usize]) -> Vec<usize> {
    let mut strides = vec![0; dims.len()];
    let mut acc = 1usize;
    for (i, dim) in dims.iter().enumerate().rev() {
        strides[i] = acc;
        acc *= *dim;
    }
    strides
}

fn unravel_index(mut index: usize, dims: &[usize]) -> Vec<usize> {
    let mut coords = vec![0; dims.len()];
    for (i, dim) in dims.iter().enumerate().rev() {
        coords[i] = index % *dim;
        index /= *dim;
    }
    coords
}

fn bytes_to_f32(bytes: &[u8]) -> BackendResult<Vec<f32>> {
    if !bytes.len().is_multiple_of(4) {
        return Err(BackendError::execution(
            "literal byte length mismatches f32",
        ));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn bytes_to_i32(bytes: &[u8]) -> BackendResult<Vec<i32>> {
    if !bytes.len().is_multiple_of(4) {
        return Err(BackendError::execution(
            "literal byte length mismatches i32",
        ));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn f32_to_bytes(values: &[f32]) -> std::sync::Arc<[u8]> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for &value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    std::sync::Arc::from(bytes.into_boxed_slice())
}

fn i32_to_bytes(values: &[i32]) -> std::sync::Arc<[u8]> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for &value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    std::sync::Arc::from(bytes.into_boxed_slice())
}

struct MultiIndex {
    shape: Vec<usize>,
    current: Vec<usize>,
    first: bool,
}

impl MultiIndex {
    fn new(shape: &[usize]) -> Self {
        Self {
            shape: shape.to_vec(),
            current: vec![0; shape.len()],
            first: true,
        }
    }
}

impl Iterator for MultiIndex {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.shape.is_empty() {
            if self.first {
                self.first = false;
                return Some(Vec::new());
            } else {
                return None;
            }
        }
        if self.first {
            self.first = false;
            return Some(self.current.clone());
        }
        for i in (0..self.current.len()).rev() {
            self.current[i] += 1;
            if self.current[i] < self.shape[i] {
                return Some(self.current.clone());
            }
            self.current[i] = 0;
        }
        None
    }
}

fn build_index(
    strides: &[usize],
    batch_axes: &[usize],
    free_axes: &[usize],
    contract_axes: &[usize],
    batch_index: &[usize],
    free_index: &[usize],
    contract_index: &[usize],
) -> usize {
    let mut index = 0usize;
    for (axis, &coord) in batch_axes.iter().zip(batch_index.iter()) {
        index += coord * strides[*axis];
    }
    for (axis, &coord) in free_axes.iter().zip(free_index.iter()) {
        index += coord * strides[*axis];
    }
    for (axis, &coord) in contract_axes.iter().zip(contract_index.iter()) {
        index += coord * strides[*axis];
    }
    index
}
