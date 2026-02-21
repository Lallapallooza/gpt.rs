mod device;
mod tensor;

use std::sync::Arc;

use gpt_rs::backend::conversion::{
    check_program_legality, default_entrypoint_name, get_conversion_target, plan_buffers_with,
    register_conversion_target, BufferizeOptions, ConversionCache, ConversionCacheKey,
    ConversionError, ConversionOptions, ConversionResult, ConversionTarget, ConvertedEntrypoint,
    ConvertedIr, LegalitySpec, OperationKind,
};
use gpt_rs::backend::param_resolver::{InMemoryParamResolver, ParamResolver};
use gpt_rs::backend::spec::{
    BackendError, BackendResult, DType, Instruction, PortableBackend, Program, TensorInit,
    TensorLiteral, TensorSpec,
};
use serde::Serialize;

pub use tensor::TritonTensor;

/// Triton backend (GPU-only contract).
///
/// This backend does not allow CPU fallback execution.
pub struct TritonBackend {
    params: Arc<InMemoryParamResolver<TritonTensor>>,
    conversion_cache: ConversionCache,
}

impl TritonBackend {
    pub fn new() -> Self {
        register_conversion_targets();
        Self {
            params: Arc::new(InMemoryParamResolver::new()),
            conversion_cache: ConversionCache::new(),
        }
    }

    pub fn is_available() -> bool {
        device::is_available()
    }
}

impl Default for TritonBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl PortableBackend for TritonBackend {
    type TensorHandle = TritonTensor;

    fn backend_name(&self) -> &str {
        "triton"
    }

    fn param_resolver(&self) -> Option<Arc<dyn ParamResolver<Handle = Self::TensorHandle>>> {
        Some(self.params.clone())
    }

    fn materialize(&self, init: TensorInit) -> BackendResult<Self::TensorHandle> {
        let driver = device::driver()?;
        match init {
            TensorInit::Literal(literal) => {
                let buffer = driver.alloc_and_upload(literal.bytes.as_ref())?;
                Ok(TritonTensor::new(literal.spec, buffer))
            }
            TensorInit::Zeroed(spec) => {
                let bytes = byte_len_for_spec(&spec)?;
                let buffer = driver.alloc_zeroed(bytes)?;
                Ok(TritonTensor::new(spec, buffer))
            }
        }
    }

    fn to_literal(&self, tensor: &Self::TensorHandle) -> BackendResult<TensorLiteral> {
        let bytes = tensor.buffer.read_to_vec()?;
        Ok(TensorLiteral::new(tensor.spec.clone(), Arc::from(bytes)))
    }

    fn execute_instruction(
        &self,
        instruction: &Instruction,
        inputs: &[Self::TensorHandle],
    ) -> BackendResult<Vec<Self::TensorHandle>> {
        let _ = (instruction, inputs);
        Err(BackendError::execution(
            "triton backend does not implement instruction execution yet; CPU fallback is forbidden",
        ))
    }

    fn run_program(
        &self,
        program: &Program,
        entry_inputs: &[Self::TensorHandle],
    ) -> BackendResult<Vec<Self::TensorHandle>> {
        let target = get_conversion_target("triton").ok_or_else(|| {
            BackendError::execution("conversion target 'triton' is not registered")
        })?;
        let options = ConversionOptions::default();
        let key = ConversionCacheKey::new(program, target.as_ref(), &options, None)
            .map_err(|err| BackendError::execution(err.to_string()))?;
        let _converted = self
            .conversion_cache
            .get_or_convert(key, || {
                target.check(program, &options)?;
                target.convert(program, &options)
            })
            .map_err(|err| BackendError::execution(err.to_string()))?;

        let _ = entry_inputs;
        Err(BackendError::execution(
            "triton backend native runtime execution is not implemented yet; CPU fallback is forbidden",
        ))
    }
}

#[derive(Debug, Clone, Default)]
pub struct TritonConversionTarget;

impl TritonConversionTarget {
    pub fn new() -> Self {
        Self
    }
}

impl ConversionTarget for TritonConversionTarget {
    fn name(&self) -> &str {
        "triton"
    }

    fn version(&self) -> u64 {
        1
    }

    fn file_extension(&self) -> &str {
        "triton"
    }

    fn check(&self, program: &Program, _options: &ConversionOptions) -> ConversionResult<()> {
        let legality = triton_legality_spec();
        if let Err(report) = check_program_legality(program, &legality) {
            let summary = report
                .diagnostics
                .first()
                .map(|d| d.message.clone())
                .unwrap_or_else(|| "unknown legality failure".to_string());
            return Err(ConversionError::new(format!(
                "triton legality check failed ({} diagnostics): {}",
                report.diagnostics.len(),
                summary
            )));
        }

        let buffer_opts = BufferizeOptions {
            require_static_shapes: true,
            require_known_dtypes: true,
        };
        plan_buffers_with(program, &buffer_opts)
            .map_err(|err| ConversionError::new(err.to_string()))?;
        Ok(())
    }

    fn convert(
        &self,
        program: &Program,
        _options: &ConversionOptions,
    ) -> ConversionResult<ConvertedIr> {
        self.check(program, &ConversionOptions::default())?;

        let entrypoint = default_entrypoint_name(program)?;
        let buffer_plan = plan_buffers_with(
            program,
            &BufferizeOptions {
                require_static_shapes: true,
                require_known_dtypes: true,
            },
        )
        .map_err(|err| ConversionError::new(err.to_string()))?;

        let module = TritonConvertedModule {
            bundle_version: 1,
            entrypoint: entrypoint.clone(),
            note: "bootstrap conversion output; kernel lowering pending in later milestones"
                .to_string(),
            function_count: program.functions.len(),
            region_count: program.regions.len(),
            bufferized_function_count: buffer_plan.functions.len(),
            bufferized_region_count: buffer_plan.regions.len(),
        };

        let encoded = serde_json::to_string_pretty(&module)
            .map_err(|err| ConversionError::new(err.to_string()))?;

        Ok(ConvertedIr {
            module: encoded,
            entrypoints: vec![ConvertedEntrypoint {
                ptir: program.entry.clone(),
                symbol: entrypoint,
            }],
        })
    }
}

#[derive(Debug, Clone, Serialize)]
struct TritonConvertedModule {
    bundle_version: u32,
    entrypoint: String,
    note: String,
    function_count: usize,
    region_count: usize,
    bufferized_function_count: usize,
    bufferized_region_count: usize,
}

fn triton_legality_spec() -> LegalitySpec {
    LegalitySpec::default()
        .allow_ops([
            OperationKind::Constant,
            OperationKind::ElementwiseUnary,
            OperationKind::ElementwiseBinary,
            OperationKind::StopGradient,
            OperationKind::Cast,
            OperationKind::Compare,
            OperationKind::Select,
            OperationKind::ArgMax,
            OperationKind::Reshape,
            OperationKind::Transpose,
            OperationKind::BroadcastTo,
            OperationKind::Slice,
            OperationKind::Concat,
            OperationKind::Pad,
            OperationKind::Tile,
            OperationKind::Iota,
            OperationKind::Reduce,
            OperationKind::Take,
            OperationKind::Gather,
            OperationKind::ScatterAdd,
            OperationKind::DynamicSlice,
            OperationKind::DynamicUpdateSlice,
            OperationKind::ScatterReduce,
            OperationKind::Cond,
            OperationKind::While,
            OperationKind::Scan,
            OperationKind::ExtractPatches,
            OperationKind::DotGeneral,
            OperationKind::ReduceWindow,
            OperationKind::RngUniform,
            OperationKind::RngNormal,
            OperationKind::TopK,
            OperationKind::SegmentReduce,
            OperationKind::Quantize,
            OperationKind::Dequantize,
            OperationKind::Requantize,
            OperationKind::CustomCall,
        ])
        .allow_dtypes([DType::F32, DType::Si32, DType::I1])
        .with_dynamic_dims(false)
}

pub fn register_conversion_targets() {
    register_conversion_target(Arc::new(TritonConversionTarget::new()));
}

pub fn register_triton_backend() {
    register_conversion_targets();
    gpt_rs::backend::registry::register_portable_backend("triton", TritonBackend::new);
}

#[gpt_rs::linkme::distributed_slice(gpt_rs::backend::registry::BACKEND_REGISTRARS)]
static REGISTER_TRITON_BACKEND: fn() = register_triton_backend;

fn byte_len_for_spec(spec: &TensorSpec) -> BackendResult<usize> {
    spec.byte_len().ok_or_else(|| {
        BackendError::execution(format!(
            "cannot compute byte length for Triton materialization: dtype={:?}, shape={:?}",
            spec.dtype,
            spec.shape.dims()
        ))
    })
}
