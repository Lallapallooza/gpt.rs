use std::collections::HashMap;
use std::ffi::CStr;
use std::os::raw::{c_char, c_void};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex, OnceLock};
use std::time::Duration;

use anyhow::{anyhow, ensure, Result};
use gpt_rs::backend::hashing::FingerprintHasher;
use gpt_rs::backend::{ParamStore, PortableBackend};
use gpt_rs_ir::host::HostTensor;
use gpt_rs_ir::module::Module;
use gpt_rs_ir::passes::Pipeline;
use gpt_rs_ir::types::TensorType;
use libloading::Library;

use crate::codegen::ENTRY_SYMBOL;
use crate::dtype::dtype_tag;
mod codegen;
mod dtype;
mod kernels;
mod optimizer;
mod targets;

pub use self::codegen::generate_c_module;

#[derive(Default)]
pub struct CBackend {
    params: ParamStore<HostTensor>,
    outputs: Mutex<OutputPool>,
}

/// How many runs a free buffer stays in the [`OutputPool`] when no run takes it.
const KEEP_RUNS: u64 = 4;

/// The output buffers of earlier runs. A run writes each result into a buffer of its type that no
/// other handle holds any more, such as the KV caches of the decode step before last, instead of
/// into fresh zero-filled pages. Every kernel writes all elements of its result, so a reused
/// buffer needs no clearing.
#[derive(Default)]
struct OutputPool {
    runs: u64,
    /// Each buffer and the last run that wrote it.
    buffers: Vec<(HostTensor, u64)>,
}

impl OutputPool {
    /// A buffer of `ty` for a result: a free one of the pool, else a new one.
    fn take(&mut self, ty: &TensorType) -> Result<HostTensor> {
        let free = self.buffers.iter().position(|(buffer, _)| {
            buffer.is_unique()
                && buffer.dtype() == ty.dtype
                && ty
                    .static_dims()
                    .is_some_and(|dims| buffer.dims[..] == dims[..])
        });
        match free {
            Some(index) => Ok(self.buffers.swap_remove(index).0),
            None => HostTensor::zeros(ty),
        }
    }

    /// Keeps the results of a finished run, and drops the free buffers that no run took for
    /// [`KEEP_RUNS`] runs.
    fn keep(&mut self, results: &[HostTensor]) {
        self.runs += 1;
        let run = self.runs;
        self.buffers
            .extend(results.iter().map(|result| (result.clone(), run)));
        self.buffers
            .retain(|(buffer, written)| !buffer.is_unique() || run - written < KEEP_RUNS);
    }
}

impl CBackend {
    pub fn new() -> Self {
        Self::default()
    }
}

/// A function compiled into a loaded library, with the types of its results.
pub struct CompiledFunction {
    module: Arc<CompiledModule>,
    outputs: Vec<TensorType>,
}

/// The compiled modules of this process, by fingerprint. A module stays loaded until the process
/// exits: unloading it can unload the OpenMP runtime it links while the runtime's idle threads
/// still run.
static COMPILED: LazyLock<Mutex<HashMap<u64, Arc<CompiledModule>>>> =
    LazyLock::new(Default::default);

/// One lock per fingerprint, held while that module compiles and loads.
static COMPILE_GATES: LazyLock<Mutex<HashMap<u64, Arc<Mutex<()>>>>> =
    LazyLock::new(Default::default);

impl PortableBackend for CBackend {
    type TensorHandle = HostTensor;
    type Executable = CompiledFunction;

    fn backend_name(&self) -> &str {
        "c"
    }

    fn pipeline(&self) -> Pipeline {
        optimizer::pipeline()
    }

    fn params(&self) -> &ParamStore<HostTensor> {
        &self.params
    }

    fn compile(&self, module: &Arc<Module>, func: &str) -> Result<CompiledFunction> {
        let inst = module
            .func(func)
            .ok_or_else(|| anyhow!("the module has no function @{func}"))?;
        let outputs = module
            .op(inst)
            .as_func()
            .expect("a function")
            .results
            .iter()
            .map(|ty| {
                ty.as_tensor()
                    .cloned()
                    .ok_or_else(|| anyhow!("@{func} returns a {ty}, not a tensor"))
            })
            .collect::<Result<Vec<_>>>()?;
        let source = {
            let _scope = gpt_rs::profiling::compile_scope("c_backend.codegen");
            generate_c_module(module, func)?
        };
        let module = get_or_compile(&source)?;
        Ok(CompiledFunction { module, outputs })
    }

    fn run(&self, exe: &CompiledFunction, args: &[HostTensor]) -> Result<Vec<HostTensor>> {
        let module = &exe.module;
        let mut outputs = Vec::with_capacity(exe.outputs.len());
        {
            let mut pool = self.outputs.lock().expect("output pool poisoned");
            for ty in &exe.outputs {
                outputs.push(pool.take(ty)?);
            }
        }
        // The extents of every argument, then of every output, as the entrypoint reads them.
        let rank = args.iter().chain(&outputs).map(|t| t.dims.len()).sum();
        let mut dims: Vec<i64> = Vec::with_capacity(rank);
        for tensor in args.iter().chain(&outputs) {
            dims.extend(tensor.dims.iter().map(|&d| d as i64));
        }
        let mut rest = dims.as_slice();
        let mut next_dims = |rank: usize| {
            let (dims, tail) = rest.split_at(rank);
            rest = tail;
            dims
        };
        let mut input_views = Vec::with_capacity(args.len());
        for tensor in args {
            let data = tensor.to_le_bytes().as_ptr() as *mut c_void;
            input_views.push(ptir_tensor(tensor, next_dims(tensor.dims.len()), data)?);
        }
        let mut output_views = Vec::with_capacity(outputs.len());
        for tensor in &mut outputs {
            let dims = next_dims(tensor.dims.len());
            let data = tensor.bytes_mut().as_mut_ptr() as *mut c_void;
            output_views.push(ptir_tensor(tensor, dims, data)?);
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
            (module.entry)(
                input_views.as_ptr(),
                input_views.len(),
                output_views.as_mut_ptr(),
                output_views.len(),
            )
        };
        ensure!(
            status == 0,
            "the compiled C entrypoint failed with status {status}"
        );

        if profile_enabled {
            if let Some(profile_ops) = module.profile_ops.as_ref() {
                record_profile(profile_ops);
            }
        }
        self.outputs
            .lock()
            .expect("output pool poisoned")
            .keep(&outputs);
        Ok(outputs)
    }

    fn upload(&self, host: HostTensor) -> Result<HostTensor> {
        Ok(host)
    }

    fn download(&self, tensor: &HostTensor) -> Result<HostTensor> {
        Ok(tensor.clone())
    }
}

/// Adds the per-op times that the library counted in its last run to the profiler.
fn record_profile(profile_ops: &CProfileOps) {
    let count = profile_ops.signatures.len();
    if count == 0 {
        return;
    }
    let mut ns = vec![0u64; count];
    let mut calls = vec![0u64; count];
    let available =
        unsafe { (profile_ops.snapshot)(ns.as_mut_ptr(), calls.as_mut_ptr(), ns.len()) };
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

/// The entrypoint's view of `tensor`, whose extents `dims` and bytes `data` outlive the call.
fn ptir_tensor(tensor: &HostTensor, dims: &[i64], data: *mut c_void) -> Result<PtirTensor> {
    Ok(PtirTensor {
        dtype: dtype_tag(tensor.dtype())?,
        rank: dims.len() as u32,
        dims: dims.as_ptr(),
        data,
    })
}

#[repr(C)]
struct PtirTensor {
    dtype: u32,
    rank: u32,
    dims: *const i64,
    data: *mut c_void,
}

type CEntrypoint = unsafe extern "C" fn(*const PtirTensor, usize, *mut PtirTensor, usize) -> i32;
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
    profile_ops: Option<CProfileOps>,
}

/// The library compiled from `source`, loaded: from this process, from the disk cache, or compiled
/// now.
fn get_or_compile(source: &str) -> Result<Arc<CompiledModule>> {
    let fingerprint = fingerprint(source);
    if c_cache_debug_enabled() {
        eprintln!("gpt-rs-backend-c: cache fingerprint={fingerprint:016x}");
    }
    if let Some(found) = COMPILED
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
        let mut guard = COMPILE_GATES.lock().expect("compiled gate cache poisoned");
        guard
            .entry(fingerprint)
            .or_insert_with(|| Arc::new(Mutex::new(())))
            .clone()
    };
    let _gate_lock = gate.lock().expect("compiled gate poisoned");
    if let Some(found) = COMPILED
        .lock()
        .expect("compiled cache poisoned")
        .get(&fingerprint)
        .cloned()
    {
        gpt_rs::profiling::cache_event("c_backend.module_hit_mem");
        return Ok(found);
    }

    let cache_dir = c_cache_dir();
    std::fs::create_dir_all(&cache_dir)?;

    let src_path = cache_dir.join(format!("program_{fingerprint:016x}.c"));
    let ext = lib_ext();
    let lib_path = cache_dir.join(format!("libgpt_rs_c_{fingerprint:016x}{ext}"));

    if !lib_path.exists() {
        gpt_rs::profiling::cache_event("c_backend.module_miss_disk");
        let _compile_scope = gpt_rs::profiling::compile_scope("c_backend.compile");
        compile_c(&src_path, source, &lib_path)?;
    } else {
        gpt_rs::profiling::cache_event("c_backend.module_hit_disk");
    }

    let _load_scope = gpt_rs::profiling::compile_scope("c_backend.dlopen");
    let lib = match unsafe { Library::new(&lib_path) } {
        Ok(lib) => lib,
        Err(_err) => {
            // A cached library can be corrupt, for example after a crash during an earlier
            // write. Recompile it once and retry the load.
            let _ = std::fs::remove_file(&lib_path);
            let _compile_scope = gpt_rs::profiling::compile_scope("c_backend.compile");
            compile_c(&src_path, source, &lib_path)?;
            unsafe { Library::new(&lib_path) }?
        }
    };
    let entry = unsafe {
        lib.get::<CEntrypoint>(ENTRY_SYMBOL.as_bytes())
            .map(|symbol| *symbol)
    }?;

    let profile_ops = {
        let count_fn = unsafe { lib.get::<CProfileOpCount>(b"gpt_rs_c_prof_op_count") }.ok();
        let snapshot_fn =
            unsafe { lib.get::<CProfileOpSnapshot>(b"gpt_rs_c_prof_op_snapshot") }.ok();
        let label_fn = unsafe { lib.get::<CProfileOpLabel>(b"gpt_rs_c_prof_op_label") }.ok();
        let sig_fn = unsafe { lib.get::<CProfileOpSignature>(b"gpt_rs_c_prof_op_signature") }.ok();
        let elements_fn = unsafe { lib.get::<CProfileOpU64>(b"gpt_rs_c_prof_op_elements") }.ok();
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
                        Some(gpt_rs::profiling::intern_label(&raw))
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

    let module = Arc::new(CompiledModule {
        _lib: lib,
        entry,
        profile_ops,
    });
    COMPILED
        .lock()
        .expect("compiled cache poisoned")
        .insert(fingerprint, Arc::clone(&module));
    Ok(module)
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

/// The fingerprint of the library that this process's compiler and flags build from `source`.
fn fingerprint(source: &str) -> u64 {
    let mut hasher = FingerprintHasher::new();
    hasher.write(&source);
    hasher.write_u8(u8::from(c_profile_enabled()));
    let command = compiler_command();
    hasher.write(&command.compiler);
    hasher.write(&command.flags);
    hasher.write(&command.version);
    hasher.write(&command.host_cpu);
    hasher.write(&command.host_defines);
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

/// Compiler, flags and host identity for generated modules. All of it is part of the cache
/// fingerprint, because modules are built for the host CPU (`-march=native`).
struct CompilerCommand {
    compiler: String,
    flags: Vec<&'static str>,
    version: String,
    host_cpu: String,
    host_defines: Vec<String>,
}

fn compiler_command() -> &'static CompilerCommand {
    static COMMAND: OnceLock<CompilerCommand> = OnceLock::new();
    COMMAND.get_or_init(|| {
        let compiler = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
        let version = compiler_version(&compiler);
        let mut flags = Vec::new();
        if cfg!(target_os = "macos") {
            flags.push("-dynamiclib");
        } else {
            flags.extend(["-shared", "-fPIC"]);
        }
        flags.push("-O3");
        if !cfg!(target_os = "windows") {
            flags.push("-march=native");
            // No -ffast-math and no cross-op FMA contraction, so fused kernels round like the
            // unfused ops. Vector math comes from libmvec.
            flags.extend([
                "-ffp-contract=off",
                "-fno-math-errno",
                "-fno-trapping-math",
                "-fomit-frame-pointer",
            ]);
            if cfg!(all(
                target_os = "linux",
                target_env = "gnu",
                target_arch = "x86_64"
            )) && version.contains("clang")
            {
                flags.push("-fveclib=libmvec");
            }
        }
        if compiler_accepts(&compiler, "-fopenmp") {
            flags.push("-fopenmp");
        } else if compiler_accepts(&compiler, "-fopenmp-simd") {
            flags.push("-fopenmp-simd");
        }
        CompilerCommand {
            compiler,
            flags,
            version,
            host_cpu: host_cpu_signature(),
            host_defines: host_defines(),
        }
    })
}

fn compiler_version(compiler: &str) -> String {
    Command::new(compiler)
        .arg("--version")
        .output()
        .map(|output| String::from_utf8_lossy(&output.stdout).into_owned())
        .unwrap_or_default()
}

fn compiler_accepts(compiler: &str, flag: &str) -> bool {
    let dir = std::env::temp_dir().join(format!("gpt_rs_c_probe_{}", std::process::id()));
    if std::fs::create_dir_all(&dir).is_err() {
        return false;
    }
    let src = dir.join("probe.c");
    let out = dir.join(format!("probe{}", lib_ext()));
    let ok = std::fs::write(&src, "int gpt_rs_c_probe(void) { return 0; }\n").is_ok()
        && Command::new(compiler)
            .args(if cfg!(target_os = "macos") {
                &["-dynamiclib"][..]
            } else {
                &["-shared", "-fPIC"][..]
            })
            .arg(flag)
            .arg("-o")
            .arg(&out)
            .arg(&src)
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false);
    let _ = std::fs::remove_dir_all(&dir);
    ok
}

fn host_defines() -> Vec<String> {
    let mut defines = Vec::new();
    if std::env::var_os("OMP_NUM_THREADS").is_none() {
        // One thread per physical core: a team that also uses SMT siblings stalls at every
        // barrier whenever another process is runnable.
        let allowed = std::thread::available_parallelism().map_or(1, |n| n.get());
        let threads = num_cpus::get_physical().clamp(1, allowed);
        defines.push(format!("-DGPTRS_NUM_THREADS={threads}"));
    }
    for level in [1, 2] {
        if let Some(bytes) = data_cache_bytes(level) {
            defines.push(format!("-DGPTRS_L{level}_BYTES={bytes}"));
        }
    }
    defines
}

/// Size of the level `level` data or unified cache of the first core.
fn data_cache_bytes(level: u32) -> Option<usize> {
    let caches = std::fs::read_dir("/sys/devices/system/cpu/cpu0/cache").ok()?;
    caches.flatten().find_map(|entry| {
        let read = |name: &str| std::fs::read_to_string(entry.path().join(name)).ok();
        if read("level")?.trim() != level.to_string() || read("type")?.trim() == "Instruction" {
            return None;
        }
        let size = read("size")?;
        let size = size.trim();
        let (digits, unit) = match size.strip_suffix('K') {
            Some(digits) => (digits, 1 << 10),
            None => match size.strip_suffix('M') {
                Some(digits) => (digits, 1 << 20),
                None => (size, 1),
            },
        };
        digits.parse::<usize>().ok().map(|n| n * unit)
    })
}

fn host_cpu_signature() -> String {
    let Ok(cpuinfo) = std::fs::read_to_string("/proc/cpuinfo") else {
        return String::new();
    };
    cpuinfo
        .lines()
        .filter(|line| {
            let key = line.split(':').next().unwrap_or("").trim();
            key == "model name" || key == "flags" || key == "Features"
        })
        .take(2)
        .collect::<Vec<_>>()
        .join("\n")
}

/// Compiles `source` into `out` and leaves the source at `src`. Each thread writes its own
/// temporary files and renames them into place, so no compiler reads a half-written file.
fn compile_c(src: &Path, source: &str, out: &Path) -> Result<()> {
    static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

    let command = compiler_command();
    let mut cmd = Command::new(&command.compiler);
    cmd.args(&command.flags);
    if c_profile_enabled() {
        cmd.arg("-DGPTRS_C_PROFILE");
    }
    cmd.args(&command.host_defines);
    let pid = std::process::id();
    let nonce = TMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let tmp_out = out.with_file_name(format!(
        "{}.tmp.{}.{}",
        out.file_name()
            .ok_or_else(|| anyhow!("invalid output path"))?
            .to_string_lossy(),
        pid,
        nonce
    ));
    let tmp_src = src.with_extension(format!("tmp.{pid}.{nonce}.c"));
    std::fs::write(&tmp_src, source)?;
    cmd.arg("-o").arg(&tmp_out).arg(&tmp_src);

    if !cfg!(target_os = "windows") {
        cmd.arg("-lm");
    }

    let output = cmd.output();
    let _ = std::fs::rename(&tmp_src, src);
    let output = output?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow!("C compiler failed: {stderr}"));
    }
    match std::fs::rename(&tmp_out, out) {
        Ok(()) => Ok(()),
        Err(err) => {
            // If another thread finished compiling first, keep the existing library.
            if out.exists() {
                let _ = std::fs::remove_file(&tmp_out);
                Ok(())
            } else {
                Err(err.into())
            }
        }
    }
}
