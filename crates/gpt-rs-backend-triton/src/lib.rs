mod artifact;
mod artifact_cache;
mod codegen;
mod compiler;
mod device;
pub mod kernels;
mod optimizer;
mod runtime;
mod targets;
mod tensor;

use std::sync::Arc;

use gpt_rs::backend::conversion::{
    check_program_legality, default_entrypoint_name, get_conversion_target, plan_buffers_with,
    register_conversion_target, BufferizeOptions, ConversionCache, ConversionCacheKey,
    ConversionError, ConversionOptions, ConversionResult, ConversionTarget, ConvertedEntrypoint,
    ConvertedIr, LegalitySpec, OperationKind,
};
use gpt_rs::backend::param_resolver::{InMemoryParamResolver, ParamResolver};
use gpt_rs::backend::pipeline::BackendPipeline;
use gpt_rs::backend::spec::{
    BackendError, BackendResult, DType, DecodeSampleRequest, Instruction, PortableBackend, Program,
    TensorInit, TensorLiteral, TensorSpec,
};

pub use tensor::TritonTensor;

/// Triton backend (GPU-only contract).
///
/// This backend does not allow CPU fallback execution.
pub struct TritonBackend {
    params: Arc<InMemoryParamResolver<TritonTensor>>,
    conversion_cache: ConversionCache,
    decoded_artifact_cache:
        artifact_cache::DecodedArtifactCache<ConversionCacheKey, artifact::TritonArtifact>,
    executor: runtime::TritonExecutor,
}

impl TritonBackend {
    pub fn new() -> Self {
        register_conversion_targets();
        Self {
            params: Arc::new(InMemoryParamResolver::new()),
            conversion_cache: ConversionCache::new(),
            decoded_artifact_cache: artifact_cache::DecodedArtifactCache::new(),
            executor: runtime::TritonExecutor::new(),
        }
    }

    pub fn is_available() -> bool {
        device::is_available()
    }

    pub fn convert_for_execution(
        &self,
        program: &Program,
        options: &ConversionOptions,
    ) -> ConversionResult<ConvertedIr> {
        convert_program_for_triton(program, options, Some(self))
    }

    fn decode_artifact_cached(
        &self,
        key: ConversionCacheKey,
        module: &str,
    ) -> BackendResult<Arc<artifact::TritonArtifact>> {
        let (artifact, hit) = self
            .decoded_artifact_cache
            .get_or_try_insert_with(key, || {
                let _scope = gpt_rs::profiling::compile_scope("triton_backend.decode_artifact");
                serde_json::from_str(module).map_err(|err| BackendError::execution(err.to_string()))
            })?;
        if hit {
            gpt_rs::profiling::cache_event("triton_backend.artifact_hit_mem");
        } else {
            gpt_rs::profiling::cache_event("triton_backend.artifact_miss_mem");
        }
        Ok(artifact)
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

    fn pipeline(&self) -> Option<Arc<dyn BackendPipeline<Self>>> {
        Some(Arc::new(optimizer::TritonPipeline))
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
            "triton backend instruction path is not implemented; use run_program",
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
        let converted = {
            let _convert_scope = gpt_rs::profiling::compile_scope("triton_backend.convert");
            self.conversion_cache
                .get_or_convert(key.clone(), || {
                    convert_program_for_triton(program, &options, Some(self))
                })
                .map_err(|err| BackendError::execution(err.to_string()))?
        };
        let artifact = self.decode_artifact_cached(key, &converted.module)?;
        self.executor
            .execute_artifact(artifact.as_ref(), entry_inputs)
    }

    fn supports_decode_sampling(&self, request: DecodeSampleRequest) -> bool {
        request.top_k.is_none()
    }

    fn sample_decode_token(
        &self,
        logits: &Self::TensorHandle,
        logits_spec: &TensorSpec,
        request: DecodeSampleRequest,
    ) -> BackendResult<Option<usize>> {
        self.executor
            .sample_decode_token(logits, logits_spec, request)
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
        6
    }

    fn file_extension(&self) -> &str {
        "triton"
    }

    fn check(&self, program: &Program, _options: &ConversionOptions) -> ConversionResult<()> {
        check_program_for_triton(program)
    }

    fn convert(
        &self,
        program: &Program,
        options: &ConversionOptions,
    ) -> ConversionResult<ConvertedIr> {
        convert_program_for_triton(program, options, None)
    }
}

fn check_program_for_triton(program: &Program) -> ConversionResult<()> {
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

fn convert_program_for_triton(
    program: &Program,
    options: &ConversionOptions,
    backend: Option<&TritonBackend>,
) -> ConversionResult<ConvertedIr> {
    check_program_for_triton(program)?;
    let optimized = optimizer::optimize_program_for_triton(program)?;

    let buffer_opts = BufferizeOptions {
        require_static_shapes: true,
        require_known_dtypes: true,
    };
    let buffer_plan = plan_buffers_with(&optimized, &buffer_opts)
        .map_err(|err| ConversionError::new(err.to_string()))?;

    let _ = (options, backend);
    let entrypoint = default_entrypoint_name(&optimized)?;
    let artifact = codegen::lower_program_to_artifact(&optimized, &entrypoint, buffer_plan)?;

    let encoded = serde_json::to_string_pretty(&artifact)
        .map_err(|err| ConversionError::new(err.to_string()))?;

    Ok(ConvertedIr {
        module: encoded,
        entrypoints: vec![ConvertedEntrypoint {
            ptir: program.entry.clone(),
            symbol: entrypoint,
        }],
    })
}

fn triton_legality_spec() -> LegalitySpec {
    LegalitySpec::default()
        .allow_ops([
            OperationKind::Constant,
            OperationKind::StopGradient,
            OperationKind::Reshape,
            OperationKind::Transpose,
            OperationKind::ElementwiseUnary,
            OperationKind::ElementwiseBinary,
            OperationKind::DotGeneral,
            OperationKind::Reduce,
            OperationKind::BroadcastTo,
            OperationKind::Slice,
            OperationKind::Concat,
            OperationKind::Iota,
            OperationKind::Compare,
            OperationKind::Select,
            OperationKind::Take,
            OperationKind::DynamicSlice,
            OperationKind::DynamicUpdateSlice,
            OperationKind::ExtractPatches,
            OperationKind::ReduceWindow,
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

pub fn set_gpu_event_timing(enabled: bool) {
    runtime::set_gpu_event_timing_enabled(enabled);
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
