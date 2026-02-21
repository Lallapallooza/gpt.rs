use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex};

use gpt_rs::backend::spec::{BackendError, BackendResult};
use serde::Deserialize;

use crate::bundle::KernelSpec;

#[derive(Debug, Clone)]
pub struct CompiledKernel {
    pub fingerprint: u64,
    pub ptx: Arc<str>,
    pub symbol: Arc<str>,
}

#[derive(Default)]
pub struct KernelCompiler {
    compiled: Mutex<HashMap<u64, Arc<CompiledKernel>>>,
    compile_gates: Mutex<HashMap<u64, Arc<Mutex<()>>>>,
}

impl KernelCompiler {
    pub fn new() -> Self {
        Self {
            compiled: Mutex::new(HashMap::new()),
            compile_gates: Mutex::new(HashMap::new()),
        }
    }

    pub fn compile(&self, kernel: &KernelSpec) -> BackendResult<Arc<CompiledKernel>> {
        let fingerprint = kernel_fingerprint(kernel);

        if let Some(found) = self
            .compiled
            .lock()
            .expect("triton kernel cache poisoned")
            .get(&fingerprint)
            .cloned()
        {
            return Ok(found);
        }

        let gate = {
            let mut guard = self
                .compile_gates
                .lock()
                .expect("triton compile gate cache poisoned");
            guard
                .entry(fingerprint)
                .or_insert_with(|| Arc::new(Mutex::new(())))
                .clone()
        };
        let _gate_lock = gate.lock().expect("triton compile gate poisoned");

        if let Some(found) = self
            .compiled
            .lock()
            .expect("triton kernel cache poisoned")
            .get(&fingerprint)
            .cloned()
        {
            return Ok(found);
        }

        let cache_dir = triton_cache_dir();
        std::fs::create_dir_all(&cache_dir)
            .map_err(|err| BackendError::execution(err.to_string()))?;

        let source_path = cache_dir.join(format!("kernel_{fingerprint:016x}.triton"));
        let ptx_path = cache_dir.join(format!("kernel_{fingerprint:016x}.ptx"));
        let meta_path = cache_dir.join(format!("kernel_{fingerprint:016x}.meta.json"));

        if !ptx_path.exists() || !meta_path.exists() {
            std::fs::write(&source_path, &kernel.source)
                .map_err(|err| BackendError::execution(err.to_string()))?;
            run_tritoncc_compile(&source_path, &ptx_path, &meta_path)?;
        }

        let ptx = std::fs::read_to_string(&ptx_path)
            .map_err(|err| BackendError::execution(err.to_string()))?;
        let meta = std::fs::read_to_string(&meta_path)
            .map_err(|err| BackendError::execution(err.to_string()))?;
        let parsed: TritonCcMeta =
            serde_json::from_str(&meta).map_err(|err| BackendError::execution(err.to_string()))?;

        let compiled = Arc::new(CompiledKernel {
            fingerprint,
            ptx: Arc::from(ptx),
            symbol: Arc::from(parsed.kernel_symbol),
        });
        self.compiled
            .lock()
            .expect("triton kernel cache poisoned")
            .insert(fingerprint, Arc::clone(&compiled));
        Ok(compiled)
    }
}

#[derive(Debug, Deserialize)]
struct TritonCcMeta {
    kernel_symbol: String,
}

fn run_tritoncc_compile(source: &Path, ptx: &Path, meta: &Path) -> BackendResult<()> {
    let compiler = tritoncc_binary();
    let arch = triton_arch();

    let output = Command::new(&compiler)
        .arg("compile")
        .arg("--arch")
        .arg(arch)
        .arg("--in")
        .arg(source)
        .arg("--out")
        .arg(ptx)
        .arg("--meta")
        .arg(meta)
        .output()
        .map_err(|err| BackendError::execution(format!("failed to run {compiler}: {err}")))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        return Err(BackendError::execution(format!(
            "tritoncc compile failed (status={}): stdout='{}' stderr='{}'",
            output.status,
            stdout.trim(),
            stderr.trim()
        )));
    }

    Ok(())
}

fn triton_cache_dir() -> PathBuf {
    if let Ok(value) = std::env::var("GPTRS_TRITON_CACHE_DIR") {
        let trimmed = value.trim();
        if !trimmed.is_empty() {
            return PathBuf::from(trimmed);
        }
    }
    PathBuf::from(".cache/gptrs-triton")
}

fn tritoncc_binary() -> String {
    for key in ["GPTRS_TRITONCC", "TRITONCC"] {
        if let Ok(value) = std::env::var(key) {
            let trimmed = value.trim();
            if !trimmed.is_empty() {
                return trimmed.to_string();
            }
        }
    }

    // Preferred path in this repository.
    let local = Path::new("tools/tritoncc/target/debug/tritoncc");
    if local.exists() {
        return local.display().to_string();
    }

    "tritoncc".to_string()
}

fn triton_arch() -> String {
    if let Ok(value) = std::env::var("GPTRS_TRITON_ARCH") {
        let trimmed = value.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }
    "sm_80".to_string()
}

fn kernel_fingerprint(kernel: &KernelSpec) -> u64 {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(kernel.id.as_bytes());
    bytes.extend_from_slice(kernel.symbol.as_bytes());
    bytes.extend_from_slice(kernel.source.as_bytes());
    fnv1a(&bytes)
}

fn fnv1a(bytes: &[u8]) -> u64 {
    const OFFSET: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;

    let mut hash = OFFSET;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}
