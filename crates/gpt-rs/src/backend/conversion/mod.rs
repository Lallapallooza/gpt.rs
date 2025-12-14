//! PTIR conversion infrastructure (IR-to-IR).
//!
//! Conversion targets consume optimized PTIR and emit a single target IR module.

mod bufferize;
mod legality;
mod registry;
mod walker;

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::backend::spec::{Program, ProgramSerdeError};

pub use bufferize::{
    plan_buffers, plan_buffers_with, AliasKind, BufferKey, BufferPlan, BufferSlot, BufferSpec,
    BufferUsage, BufferizeError, BufferizeOptions, FunctionBufferPlan, LiveRange,
};
pub use legality::{check_program_legality, LegalityReport, LegalitySpec, OperationKind};
pub use registry::{
    get_target as get_conversion_target, list_targets as list_conversion_targets,
    register_target as register_conversion_target,
};
pub use walker::{walk_program, ProgramVisitor};

#[derive(Debug, Error, Clone)]
#[error("{message}")]
pub struct ConversionError {
    message: String,
}

impl ConversionError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl From<ProgramSerdeError> for ConversionError {
    fn from(err: ProgramSerdeError) -> Self {
        ConversionError::new(err.to_string())
    }
}

pub type ConversionResult<T> = Result<T, ConversionError>;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConversionOptions {
    pub entrypoint_override: Option<String>,
}

impl ConversionOptions {
    pub fn digest(&self) -> u64 {
        let bytes = bincode::serialize(self).unwrap_or_default();
        fnv_hash(&bytes)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConvertedEntrypoint {
    pub ptir: String,
    pub symbol: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConvertedIr {
    pub module: String,
    pub entrypoints: Vec<ConvertedEntrypoint>,
}

pub trait ConversionTarget: Send + Sync {
    fn name(&self) -> &str;
    fn version(&self) -> u64 {
        0
    }
    fn file_extension(&self) -> &str;
    fn check(&self, _program: &Program, _options: &ConversionOptions) -> ConversionResult<()> {
        Ok(())
    }
    fn convert(
        &self,
        program: &Program,
        options: &ConversionOptions,
    ) -> ConversionResult<ConvertedIr>;
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ConversionCacheKey {
    pub program_hash: u64,
    pub target: String,
    pub target_version: u64,
    pub options_hash: u64,
    pub device_caps_hash: Option<u64>,
}

impl ConversionCacheKey {
    pub fn new(
        program: &Program,
        target: &dyn ConversionTarget,
        options: &ConversionOptions,
        device_caps_hash: Option<u64>,
    ) -> ConversionResult<Self> {
        Ok(Self {
            program_hash: hash_program(program)?,
            target: target.name().to_string(),
            target_version: target.version(),
            options_hash: options.digest(),
            device_caps_hash,
        })
    }
}

type ConversionCacheEntry = Arc<OnceLock<ConversionResult<Arc<ConvertedIr>>>>;
type ConversionCacheMap = HashMap<ConversionCacheKey, ConversionCacheEntry>;

pub struct ConversionCache {
    entries: Mutex<ConversionCacheMap>,
}

impl ConversionCache {
    pub fn new() -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
        }
    }

    pub fn get_or_convert<F>(
        &self,
        key: ConversionCacheKey,
        build: F,
    ) -> ConversionResult<Arc<ConvertedIr>>
    where
        F: FnOnce() -> ConversionResult<ConvertedIr>,
    {
        let cell = {
            let mut guard = self.entries.lock().expect("conversion cache poisoned");
            guard
                .entry(key)
                .or_insert_with(|| Arc::new(OnceLock::new()))
                .clone()
        };

        if let Some(existing) = cell.get() {
            return existing.clone();
        }

        let built = build().map(Arc::new);
        let _ = cell.set(built.clone());
        match cell.get() {
            Some(stored) => stored.clone(),
            None => built,
        }
    }
}

impl Default for ConversionCache {
    fn default() -> Self {
        Self::new()
    }
}

pub fn hash_program(program: &Program) -> ConversionResult<u64> {
    let bytes = program.to_bincode_bytes()?;
    Ok(fnv_hash(&bytes))
}

pub fn sanitize_symbol(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for (idx, ch) in value.chars().enumerate() {
        let is_valid = ch.is_ascii_alphanumeric() || ch == '_';
        if idx == 0 && ch.is_ascii_digit() {
            out.push('_');
        }
        out.push(if is_valid { ch } else { '_' });
    }
    if out.is_empty() {
        out.push_str("entry");
    }
    out
}

pub fn default_entrypoint_name(program: &Program) -> ConversionResult<String> {
    let base = sanitize_symbol(&program.entry);
    let hash = hash_program(program)?;
    Ok(format!("{base}__{hash:016x}"))
}

fn fnv_hash(bytes: &[u8]) -> u64 {
    const OFFSET: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;

    let mut hash = OFFSET;
    for byte in bytes {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConversionStage {
    Optimize,
    Legalize,
    Bufferize,
    Convert,
    Codegen,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConversionDiagnostic {
    pub stage: ConversionStage,
    pub function: Option<String>,
    pub instruction_index: Option<usize>,
    pub message: String,
}

impl ConversionDiagnostic {
    pub fn new(
        stage: ConversionStage,
        function: Option<String>,
        instruction_index: Option<usize>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            stage,
            function,
            instruction_index,
            message: message.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TargetSpec {
    pub name: String,
    pub version: u64,
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub features: std::collections::BTreeMap<String, String>,
}

impl TargetSpec {
    pub fn new(name: impl Into<String>, version: u64) -> Self {
        Self {
            name: name.into(),
            version,
            features: std::collections::BTreeMap::new(),
        }
    }

    pub fn digest(&self) -> u64 {
        let bytes = bincode::serialize(self).unwrap_or_default();
        fnv_hash(&bytes)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceCaps {
    pub arch: Option<String>,
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub features: std::collections::BTreeMap<String, String>,
}

impl DeviceCaps {
    pub fn new(arch: Option<String>) -> Self {
        Self {
            arch,
            features: std::collections::BTreeMap::new(),
        }
    }

    pub fn digest(&self) -> u64 {
        let bytes = bincode::serialize(self).unwrap_or_default();
        fnv_hash(&bytes)
    }
}
