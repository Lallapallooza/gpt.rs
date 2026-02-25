use std::collections::HashMap;
use std::ffi::CStr;
use std::os::raw::{c_char, c_void};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc, Mutex,
};
use std::time::Duration;

use gpt_rs::backend::conversion::{
    check_program_legality, default_entrypoint_name, get_conversion_target, plan_buffers_with,
    register_conversion_target, sanitize_symbol, BufferizeOptions, ConversionCache,
    ConversionCacheKey, ConversionError, ConversionOptions, ConversionResult, ConversionTarget,
    ConvertedEntrypoint, ConvertedIr, LegalityReport, LegalitySpec, OperationKind,
};
use gpt_rs::backend::hashing::{fnv1a_hash, FingerprintHasher};
use gpt_rs::backend::optimizer::{
    default_optimizer, EntryParam, EntrySignature, OptimizeConfig, OptimizeContext,
    OptimizeServices,
};
use gpt_rs::backend::param_resolver::{InMemoryParamResolver, ParamResolver};
use gpt_rs::backend::registry::register_portable_backend;
use gpt_rs::backend::spec::{
    BackendError, BackendResult, DType, Dimension, PortableBackend, Program, Shape, TensorInit,
    TensorLiteral, TensorSpec, ValueType,
};
use gpt_rs::tensor::InputRole;
use libloading::Library;

use crate::dtype::dtype_tag;
use crate::labels::backend_label_from_str;
mod codegen;
mod dtype;
mod kernels;
mod labels;
mod optimizer;
mod targets;

pub struct CConversionTarget;

impl CConversionTarget {
    pub fn new() -> Self {
        Self
    }
}

impl Default for CConversionTarget {
    fn default() -> Self {
        Self
    }
}

impl ConversionTarget for CConversionTarget {
    fn name(&self) -> &str {
        "c"
    }

    fn version(&self) -> u64 {
        34
    }

    fn file_extension(&self) -> &str {
        "c"
    }

    fn check(&self, program: &Program, _options: &ConversionOptions) -> ConversionResult<()> {
        let legality = c_legality_spec();
        if let Err(report) = check_program_legality(program, &legality) {
            return Err(legality_report_to_error(report));
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
        options: &ConversionOptions,
    ) -> ConversionResult<ConvertedIr> {
        let legality = c_legality_spec();
        if let Err(report) = check_program_legality(program, &legality) {
            return Err(legality_report_to_error(report));
        }
        let optimized = optimize_program_for_c(program)?;
        let buffer_opts = BufferizeOptions {
            require_static_shapes: true,
            require_known_dtypes: true,
        };
        let buffer_plan = plan_buffers_with(&optimized, &buffer_opts)
            .map_err(|err| ConversionError::new(err.to_string()))?;

        let entrypoint = match options.entrypoint_override.as_ref() {
            Some(name) => {
                let base = sanitize_symbol(name);
                let hash = gpt_rs::backend::conversion::hash_program(&optimized)?;
                format!("{base}__{hash:016x}")
            }
            None => default_entrypoint_name(&optimized)?,
        };

        let module = codegen::generate_c_module(&optimized, &entrypoint, &buffer_plan)?;

        Ok(ConvertedIr {
            module,
            entrypoints: vec![ConvertedEntrypoint {
                ptir: program.entry.clone(),
                symbol: entrypoint,
            }],
        })
    }
}

pub fn register_conversion_targets() {
    register_conversion_target(Arc::new(CConversionTarget::new()));
}

pub struct CBackend {
    cache: ConversionCache,
    compiled: Mutex<HashMap<u64, Arc<CompiledModule>>>,
    compile_gates: Mutex<HashMap<u64, Arc<Mutex<()>>>>,
    params: Arc<InMemoryParamResolver<CTensor>>,
}

impl CBackend {
    pub fn new() -> Self {
        register_conversion_targets();
        Self {
            cache: ConversionCache::new(),
            compiled: Mutex::new(HashMap::new()),
            compile_gates: Mutex::new(HashMap::new()),
            params: Arc::new(InMemoryParamResolver::new()),
        }
    }
}

impl Default for CBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl PortableBackend for CBackend {
    type TensorHandle = CTensor;

    fn backend_name(&self) -> &str {
        "c"
    }

    fn pipeline(&self) -> Option<Arc<dyn gpt_rs::backend::pipeline::BackendPipeline<Self>>> {
        Some(Arc::new(optimizer::CPipeline))
    }

    fn param_resolver(&self) -> Option<Arc<dyn ParamResolver<Handle = Self::TensorHandle>>> {
        Some(self.params.clone())
    }

    fn materialize(&self, init: TensorInit) -> BackendResult<Self::TensorHandle> {
        match init {
            TensorInit::Literal(literal) => CTensor::from_literal(literal),
            TensorInit::Zeroed(spec) => CTensor::zeroed(spec),
        }
    }

    fn to_literal(&self, tensor: &Self::TensorHandle) -> BackendResult<TensorLiteral> {
        tensor.to_literal()
    }

    fn execute_instruction(
        &self,
        instruction: &gpt_rs::backend::spec::Instruction,
        inputs: &[Self::TensorHandle],
    ) -> BackendResult<Vec<Self::TensorHandle>> {
        let _ = (instruction, inputs);
        Err(BackendError::execution(
            "C backend does not execute PTIR instructions yet",
        ))
    }

    fn run_program(
        &self,
        program: &Program,
        entry_inputs: &[Self::TensorHandle],
    ) -> BackendResult<Vec<Self::TensorHandle>> {
        let target = get_conversion_target("c")
            .ok_or_else(|| BackendError::execution("conversion target 'c' is not registered"))?;
        let options = ConversionOptions::default();
        let key = ConversionCacheKey::new(program, target.as_ref(), &options, None)
            .map_err(|err| BackendError::execution(err.to_string()))?;
        let converted = {
            let _convert_scope = gpt_rs::profiling::compile_scope("c_backend.convert");
            self.cache
                .get_or_convert(key.clone(), || {
                    target.check(program, &options)?;
                    target.convert(program, &options)
                })
                .map_err(|err| BackendError::execution(err.to_string()))?
        };

        let entrypoint = converted
            .entrypoints
            .first()
            .ok_or_else(|| BackendError::execution("converted module missing entrypoints"))?;

        let module = self
            .get_or_compile(&key, converted.as_ref(), &entrypoint.symbol)
            .map_err(|err| BackendError::execution(err.to_string()))?;

        let entry_fn = module.entry;
        let output_specs = entry_output_specs(program)?;
        let mut outputs: Vec<CTensor> = output_specs
            .into_iter()
            .map(CTensor::zeroed)
            .collect::<Result<_, _>>()?;

        let input_views: Vec<PtirTensor> =
            entry_inputs.iter().map(|t| t.as_ptir_tensor()).collect();
        let mut output_views: Vec<PtirTensor> =
            outputs.iter_mut().map(|t| t.as_ptir_tensor_mut()).collect();

        if let Some(prepare) = module.prepare {
            if !module.prepared.swap(true, Ordering::AcqRel) {
                unsafe { prepare(input_views.as_ptr(), input_views.len()) };
            }
        }

        let profile_enabled = c_profile_enabled();
        if profile_enabled {
            if let Some(profile_ops) = module.profile_ops.as_ref() {
                if let Some(reset) = profile_ops.reset {
                    unsafe { reset() };
                }
            }
        }

        let status = unsafe {
            (entry_fn)(
                input_views.as_ptr(),
                input_views.len(),
                output_views.as_mut_ptr(),
                output_views.len(),
            )
        };
        if status != 0 {
            return Err(BackendError::execution(
                "C backend execution failed while calling compiled entrypoint",
            ));
        }

        if profile_enabled {
            if let Some(profile_ops) = module.profile_ops.as_ref() {
                let count = profile_ops.signatures.len();
                if count > 0 {
                    let mut ns = vec![0u64; count];
                    let mut calls = vec![0u64; count];
                    let available = unsafe {
                        (profile_ops.snapshot)(ns.as_mut_ptr(), calls.as_mut_ptr(), ns.len())
                    };
                    let used = std::cmp::min(available, ns.len());
                    for idx in 0..used {
                        let call_count = calls[idx];
                        if call_count == 0 {
                            continue;
                        }
                        if let Some(label) = profile_ops.labels[idx] {
                            let duration = Duration::from_nanos(ns[idx]);
                            gpt_rs::profiling::record_backend_aggregate_with_signature(
                                label,
                                profile_ops.signatures[idx],
                                call_count,
                                duration,
                                profile_ops.work[idx],
                            );
                        }
                    }
                }
            }
        }

        Ok(outputs)
    }
}

/// Register the C backend with the global backend registry.
pub fn register_c_backend() {
    register_portable_backend("c", CBackend::new);
}

#[gpt_rs::linkme::distributed_slice(gpt_rs::backend::registry::BACKEND_REGISTRARS)]
static REGISTER_C_BACKEND: fn() = register_c_backend;

fn c_legality_spec() -> LegalitySpec {
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

fn optimize_program_for_c(program: &Program) -> ConversionResult<Program> {
    let mut optimized = program.clone();
    let backend = CBackend::new();
    let optimizer = default_optimizer(backend.pipeline());
    let params = backend.param_resolver();
    for function in optimized.functions.iter_mut() {
        let entry_params = function
            .parameter_ids
            .iter()
            .copied()
            .zip(function.parameters.iter().cloned())
            .map(|(id, ty)| EntryParam {
                id,
                ty,
                role: InputRole::Arg,
                stable_id: None,
            })
            .collect::<Vec<_>>();
        let entry = EntrySignature::new(entry_params);
        let services = OptimizeServices {
            params: params.as_deref(),
        };
        let cfg = OptimizeConfig::default();
        let mut cx = OptimizeContext::new(&backend, services, entry, cfg);
        let _ = optimizer.optimize(function, &mut cx);
    }
    Ok(optimized)
}

fn legality_report_to_error(report: LegalityReport) -> ConversionError {
    if report.diagnostics.is_empty() {
        return ConversionError::new("unknown legality failure");
    }
    let message = report
        .diagnostics
        .iter()
        .take(3)
        .map(|diag| diag.message.clone())
        .collect::<Vec<_>>()
        .join("; ");
    ConversionError::new(message)
}

#[repr(C)]
struct PtirTensor {
    dtype: u32,
    rank: u32,
    dims: *const i64,
    data: *mut c_void,
}

type CEntrypoint = unsafe extern "C" fn(*const PtirTensor, usize, *mut PtirTensor, usize) -> i32;
type CPrepare = unsafe extern "C" fn(*const PtirTensor, usize);
type CProfileOpReset = unsafe extern "C" fn();
type CProfileOpSnapshot = unsafe extern "C" fn(*mut u64, *mut u64, usize) -> usize;
type CProfileOpCount = unsafe extern "C" fn() -> usize;
type CProfileOpSignature = unsafe extern "C" fn(usize) -> *const c_char;
type CProfileOpLabel = unsafe extern "C" fn(usize) -> *const c_char;
type CProfileOpU64 = unsafe extern "C" fn(usize) -> u64;

struct CProfileOps {
    snapshot: CProfileOpSnapshot,
    reset: Option<CProfileOpReset>,
    labels: Vec<Option<&'static str>>,
    signatures: Vec<Option<u32>>,
    work: Vec<gpt_rs::profiling::WorkStats>,
}

struct CompiledModule {
    _lib: Library,
    entry: CEntrypoint,
    prepare: Option<CPrepare>,
    prepared: AtomicBool,
    profile_ops: Option<CProfileOps>,
}

impl CompiledModule {
    fn new(
        lib: Library,
        entry: CEntrypoint,
        prepare: Option<CPrepare>,
        profile_ops: Option<CProfileOps>,
    ) -> Self {
        Self {
            _lib: lib,
            entry,
            prepare,
            prepared: AtomicBool::new(false),
            profile_ops,
        }
    }
}

#[derive(Clone)]
pub struct CTensor {
    spec: TensorSpec,
    dims: Vec<i64>,
    data: Arc<Vec<u8>>,
}

impl CTensor {
    fn from_literal(literal: TensorLiteral) -> BackendResult<Self> {
        let dims = static_dims(&literal.spec.shape)?;
        Ok(Self {
            spec: literal.spec.clone(),
            dims,
            data: Arc::new(literal.bytes.as_ref().to_vec()),
        })
    }

    fn zeroed(spec: TensorSpec) -> BackendResult<Self> {
        let dims = static_dims(&spec.shape)?;
        let byte_len = element_count(&dims)?
            .checked_mul(dtype_size(spec.dtype)?)
            .ok_or_else(|| BackendError::execution("tensor byte size overflow"))?;
        Ok(Self {
            spec,
            dims,
            data: Arc::new(vec![0u8; byte_len]),
        })
    }

    fn to_literal(&self) -> BackendResult<TensorLiteral> {
        Ok(TensorLiteral::new(
            self.spec.clone(),
            Arc::<[u8]>::from(self.data.as_ref().clone()),
        ))
    }

    fn as_ptir_tensor(&self) -> PtirTensor {
        PtirTensor {
            dtype: dtype_tag(self.spec.dtype).unwrap_or(0),
            rank: self.dims.len() as u32,
            dims: self.dims.as_ptr(),
            data: self.data.as_ptr() as *mut c_void,
        }
    }

    fn as_ptir_tensor_mut(&mut self) -> PtirTensor {
        let data = Arc::make_mut(&mut self.data);
        PtirTensor {
            dtype: dtype_tag(self.spec.dtype).unwrap_or(0),
            rank: self.dims.len() as u32,
            dims: self.dims.as_ptr(),
            data: data.as_mut_ptr() as *mut c_void,
        }
    }
}

impl CBackend {
    fn get_or_compile(
        &self,
        key: &ConversionCacheKey,
        converted: &ConvertedIr,
        entrypoint: &str,
    ) -> BackendResult<Arc<CompiledModule>> {
        let fingerprint = cache_fingerprint(key, converted);
        if c_cache_debug_enabled() {
            let key_hash = cache_key_hash(key);
            let module_hash = fnv1a_hash(converted.module.as_bytes());
            eprintln!(
                "gpt-rs-backend-c: cache fingerprint={fingerprint:016x} key_hash={key_hash:016x} program_hash={program_hash:016x} module_hash={module_hash:016x} entrypoint={entrypoint}",
                program_hash = key.program_hash,
            );
        }
        if let Some(found) = self
            .compiled
            .lock()
            .expect("compiled cache poisoned")
            .get(&fingerprint)
            .cloned()
        {
            gpt_rs::profiling::cache_event("c_backend.module_hit_mem");
            return Ok(found);
        }

        gpt_rs::profiling::cache_event("c_backend.module_miss_mem");

        let gate = {
            let mut guard = self
                .compile_gates
                .lock()
                .expect("compiled gate cache poisoned");
            guard
                .entry(fingerprint)
                .or_insert_with(|| Arc::new(Mutex::new(())))
                .clone()
        };
        let _gate_lock = gate.lock().expect("compiled gate poisoned");
        if let Some(found) = self
            .compiled
            .lock()
            .expect("compiled cache poisoned")
            .get(&fingerprint)
            .cloned()
        {
            gpt_rs::profiling::cache_event("c_backend.module_hit_mem");
            return Ok(found);
        }

        let cache_dir = c_cache_dir();
        std::fs::create_dir_all(&cache_dir)
            .map_err(|err| BackendError::execution(err.to_string()))?;

        let src_path = cache_dir.join(format!("program_{fingerprint:016x}.c"));
        let ext = lib_ext();
        let lib_path = cache_dir.join(format!("libgpt_rs_c_{fingerprint:016x}{ext}"));

        if !lib_path.exists() {
            gpt_rs::profiling::cache_event("c_backend.module_miss_disk");
            std::fs::write(&src_path, &converted.module)
                .map_err(|err| BackendError::execution(err.to_string()))?;
            let _compile_scope = gpt_rs::profiling::compile_scope("c_backend.compile");
            compile_c(&src_path, &lib_path)?;
        } else {
            gpt_rs::profiling::cache_event("c_backend.module_hit_disk");
        }

        let _load_scope = gpt_rs::profiling::compile_scope("c_backend.dlopen");
        let lib = match unsafe { Library::new(&lib_path) } {
            Ok(lib) => lib,
            Err(_err) => {
                // A partially written shared library can happen if multiple threads compile the same
                // fingerprint concurrently. Retry a single time after forcing a recompile.
                let _ = std::fs::remove_file(&lib_path);
                std::fs::write(&src_path, &converted.module)
                    .map_err(|err| BackendError::execution(err.to_string()))?;
                let _compile_scope = gpt_rs::profiling::compile_scope("c_backend.compile");
                compile_c(&src_path, &lib_path)?;
                unsafe { Library::new(&lib_path) }
                    .map_err(|err| BackendError::execution(err.to_string()))?
            }
        };
        let entry = unsafe {
            lib.get::<CEntrypoint>(entrypoint.as_bytes())
                .map(|symbol| *symbol)
        }
        .map_err(|err| BackendError::execution(err.to_string()))?;

        let prepare = unsafe {
            lib.get::<CPrepare>(b"gpt_rs_c_prepare")
                .map(|symbol| *symbol)
                .ok()
        };
        let profile_ops = {
            let count_fn = unsafe { lib.get::<CProfileOpCount>(b"gpt_rs_c_prof_op_count") }.ok();
            let snapshot_fn =
                unsafe { lib.get::<CProfileOpSnapshot>(b"gpt_rs_c_prof_op_snapshot") }.ok();
            let label_fn = unsafe { lib.get::<CProfileOpLabel>(b"gpt_rs_c_prof_op_label") }.ok();
            let sig_fn =
                unsafe { lib.get::<CProfileOpSignature>(b"gpt_rs_c_prof_op_signature") }.ok();
            let elements_fn =
                unsafe { lib.get::<CProfileOpU64>(b"gpt_rs_c_prof_op_elements") }.ok();
            let bytes_read_fn =
                unsafe { lib.get::<CProfileOpU64>(b"gpt_rs_c_prof_op_bytes_read") }.ok();
            let bytes_written_fn =
                unsafe { lib.get::<CProfileOpU64>(b"gpt_rs_c_prof_op_bytes_written") }.ok();
            let flops_fn = unsafe { lib.get::<CProfileOpU64>(b"gpt_rs_c_prof_op_flops") }.ok();
            let reset_fn = unsafe { lib.get::<CProfileOpReset>(b"gpt_rs_c_prof_op_reset") }.ok();

            if let (
                Some(count_fn),
                Some(snapshot_fn),
                Some(label_fn),
                Some(sig_fn),
                Some(elements_fn),
                Some(bytes_read_fn),
                Some(bytes_written_fn),
                Some(flops_fn),
            ) = (
                count_fn,
                snapshot_fn,
                label_fn,
                sig_fn,
                elements_fn,
                bytes_read_fn,
                bytes_written_fn,
                flops_fn,
            ) {
                let count = unsafe { count_fn() };
                if count == 0 {
                    None
                } else {
                    let mut labels = Vec::with_capacity(count);
                    let mut signatures = Vec::with_capacity(count);
                    let mut work = Vec::with_capacity(count);
                    for idx in 0..count {
                        let label_ptr = unsafe { label_fn(idx) };
                        let label = if label_ptr.is_null() {
                            None
                        } else {
                            let raw = unsafe { CStr::from_ptr(label_ptr) }
                                .to_string_lossy()
                                .into_owned();
                            backend_label_from_str(&raw)
                        };
                        labels.push(label);
                        let sig_ptr = unsafe { sig_fn(idx) };
                        let sig = if sig_ptr.is_null() {
                            String::new()
                        } else {
                            unsafe { CStr::from_ptr(sig_ptr) }
                                .to_string_lossy()
                                .into_owned()
                        };
                        let signature_id = gpt_rs::profiling::signature_id(&sig);
                        let stats = gpt_rs::profiling::WorkStats {
                            elements: unsafe { elements_fn(idx) },
                            bytes_read: unsafe { bytes_read_fn(idx) },
                            bytes_written: unsafe { bytes_written_fn(idx) },
                            flops: unsafe { flops_fn(idx) },
                            alloc_bytes: 0,
                            alloc_count: 0,
                        };
                        signatures.push(signature_id);
                        work.push(stats);
                    }
                    Some(CProfileOps {
                        snapshot: *snapshot_fn,
                        reset: reset_fn.map(|sym| *sym),
                        labels,
                        signatures,
                        work,
                    })
                }
            } else {
                None
            }
        };

        let module = Arc::new(CompiledModule::new(lib, entry, prepare, profile_ops));
        self.compiled
            .lock()
            .expect("compiled cache poisoned")
            .insert(fingerprint, Arc::clone(&module));
        Ok(module)
    }
}

fn entry_output_specs(program: &Program) -> BackendResult<Vec<TensorSpec>> {
    let func = program
        .functions
        .iter()
        .find(|f| f.name == program.entry)
        .ok_or_else(|| BackendError::execution("entry function not found"))?;
    flatten_value_types(&func.results).map_err(|err| BackendError::execution(err.to_string()))
}

fn flatten_value_types(types: &[ValueType]) -> ConversionResult<Vec<TensorSpec>> {
    let mut out = Vec::new();
    for ty in types {
        flatten_value_type(ty, &mut out)?;
    }
    Ok(out)
}

fn flatten_value_type(ty: &ValueType, out: &mut Vec<TensorSpec>) -> ConversionResult<()> {
    match ty {
        ValueType::Tensor(spec) => {
            out.push(spec.clone());
            Ok(())
        }
        ValueType::Tuple(elements) => {
            for element in elements {
                flatten_value_type(element, out)?;
            }
            Ok(())
        }
    }
}

fn static_dims(shape: &Shape) -> BackendResult<Vec<i64>> {
    let mut dims = Vec::with_capacity(shape.rank());
    for dim in shape.dims() {
        match dim {
            Dimension::Static(value) => dims.push(*value as i64),
            Dimension::Dynamic(_) => {
                return Err(BackendError::execution(
                    "dynamic dimensions are not supported by C backend",
                ))
            }
        }
    }
    Ok(dims)
}

fn element_count(dims: &[i64]) -> BackendResult<usize> {
    let mut count = 1usize;
    for dim in dims {
        if *dim <= 0 {
            return Err(BackendError::execution("invalid dimension"));
        }
        count = count
            .checked_mul(*dim as usize)
            .ok_or_else(|| BackendError::execution("dimension overflow"))?;
    }
    Ok(count)
}

fn dtype_size(dtype: DType) -> BackendResult<usize> {
    match dtype {
        DType::F32 => Ok(4),
        DType::F16 | DType::Bf16 => Ok(2),
        DType::Si32 => Ok(4),
        DType::I1 => Ok(1),
        _ => Err(BackendError::execution(
            "dtype not supported by C backend runtime",
        )),
    }
}

fn parse_bool(value: &str) -> bool {
    let normalized = value.trim().to_ascii_lowercase();
    matches!(normalized.as_str(), "1" | "true" | "yes" | "on")
}

fn c_profile_enabled() -> bool {
    match std::env::var("GPTRS_PROFILE_BACKEND") {
        Ok(value) if !value.trim().is_empty() => parse_bool(&value),
        _ => false,
    }
}

fn c_accelerated_kernels_supported() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        std::arch::is_x86_feature_detected!("avx512f") && std::arch::is_x86_feature_detected!("fma")
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        false
    }
}

fn c_cache_debug_enabled() -> bool {
    match std::env::var("GPTRS_C_CACHE_DEBUG") {
        Ok(value) if !value.trim().is_empty() => parse_bool(&value),
        _ => false,
    }
}

fn c_cache_dir() -> PathBuf {
    match std::env::var("GPTRS_C_CACHE_DIR") {
        Ok(value) if !value.trim().is_empty() => PathBuf::from(value.trim()),
        _ => std::env::temp_dir().join("gpt_rs_c_backend"),
    }
}

fn cache_fingerprint(key: &ConversionCacheKey, converted: &ConvertedIr) -> u64 {
    let mut hasher = FingerprintHasher::new();
    hasher.write(key);
    hasher.write(&converted.module);
    hasher.write_u8(if c_accelerated_kernels_supported() {
        1
    } else {
        0
    });
    hasher.finish()
}

fn cache_key_hash(key: &ConversionCacheKey) -> u64 {
    let mut hasher = FingerprintHasher::new();
    hasher.write(key);
    hasher.finish()
}

fn lib_ext() -> &'static str {
    if cfg!(target_os = "macos") {
        ".dylib"
    } else if cfg!(target_os = "windows") {
        ".dll"
    } else {
        ".so"
    }
}

fn compile_c(src: &Path, out: &Path) -> BackendResult<()> {
    static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

    let compiler = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let mut cmd = Command::new(&compiler);
    if cfg!(target_os = "macos") {
        cmd.arg("-dynamiclib");
    } else {
        cmd.arg("-shared").arg("-fPIC");
    }
    cmd.arg("-O3");
    if !cfg!(target_os = "windows") {
        cmd.arg("-march=native");
    }
    if !cfg!(target_os = "windows") && c_accelerated_kernels_supported() {
        cmd.arg("-mavx512f");
        cmd.arg("-mfma");
    } else if !cfg!(target_os = "windows") {
        // Build baseline C kernels when AVX512/FMA are unavailable.
        cmd.arg("-mno-avx512f");
        cmd.arg("-mno-fma");
    }
    if !cfg!(target_os = "windows") {
        cmd.arg("-ffast-math");
        cmd.arg("-fno-math-errno");
        cmd.arg("-fno-trapping-math");
        cmd.arg("-fomit-frame-pointer");
    }
    cmd.arg("-DGPTRS_C_PROFILE");
    let pid = std::process::id();
    let nonce = TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let tmp_out = out.with_file_name(format!(
        "{}.tmp.{}.{}",
        out.file_name()
            .ok_or_else(|| BackendError::execution("invalid output path"))?
            .to_string_lossy(),
        pid,
        nonce
    ));
    cmd.arg("-o").arg(&tmp_out).arg(src);

    if !cfg!(target_os = "windows") {
        cmd.arg("-lm");
    }

    let output = cmd
        .output()
        .map_err(|err| BackendError::execution(err.to_string()))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(BackendError::execution(format!(
            "C compiler failed: {stderr}"
        )));
    }
    match std::fs::rename(&tmp_out, out) {
        Ok(()) => Ok(()),
        Err(err) => {
            // If another thread finished compiling first, keep the existing library.
            if out.exists() {
                let _ = std::fs::remove_file(&tmp_out);
                Ok(())
            } else {
                Err(BackendError::execution(err.to_string()))
            }
        }
    }
}
