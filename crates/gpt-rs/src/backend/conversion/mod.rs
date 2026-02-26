//! PTIR conversion infrastructure (IR-to-IR).
//!
//! Conversion targets consume optimized PTIR and emit a single target IR module.

mod bufferize;
mod legality;
mod registry;
mod walker;

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, OnceLock};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::backend::spec::{
    Function, Instruction, Operand, Operation, Program, ProgramSerdeError, Region, RegionId,
    ValueId,
};

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
            program_hash: hash_program_canonical(program)?,
            target: target.name().to_string(),
            target_version: target.version(),
            options_hash: options.digest(),
            device_caps_hash,
        })
    }
}

type ConversionCacheEntry = Arc<OnceLock<ConversionResult<Arc<ConvertedIr>>>>;
type ConversionCacheMap = HashMap<ConversionCacheKey, ConversionCacheEntry>;
type CanonicalHashMemoMap = HashMap<CanonicalHashMemoKey, u64>;

const CANONICAL_HASH_MEMO_CAPACITY: usize = 4096;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct CanonicalHashMemoKey {
    raw_hash: u64,
    raw_len: usize,
}

static CANONICAL_HASH_MEMO: OnceLock<Mutex<CanonicalHashMemoMap>> = OnceLock::new();

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
    let bytes = serialized_program(program)?;
    Ok(fnv_hash(&bytes))
}

pub fn hash_program_canonical(program: &Program) -> ConversionResult<u64> {
    let raw_bytes = serialized_program(program)?;
    let memo_key = CanonicalHashMemoKey {
        raw_hash: fnv_hash(&raw_bytes),
        raw_len: raw_bytes.len(),
    };

    if let Some(cached) = lookup_canonical_hash_memo(memo_key)? {
        crate::profiling::cache_event("conversion_hash_canonical_memo_hit");
        return Ok(cached);
    }

    crate::profiling::cache_event("conversion_hash_canonical_memo_miss");
    let canonical = canonicalize_program_for_hash(program);
    let bytes = canonical.to_bincode_bytes()?;
    let canonical_hash = fnv_hash(&bytes);
    store_canonical_hash_memo(memo_key, canonical_hash)?;
    Ok(canonical_hash)
}

fn serialized_program(program: &Program) -> ConversionResult<Vec<u8>> {
    program.to_bincode_bytes().map_err(ConversionError::from)
}

fn canonical_hash_memo() -> &'static Mutex<CanonicalHashMemoMap> {
    CANONICAL_HASH_MEMO.get_or_init(|| Mutex::new(HashMap::new()))
}

fn lookup_canonical_hash_memo(key: CanonicalHashMemoKey) -> ConversionResult<Option<u64>> {
    let guard = canonical_hash_memo()
        .lock()
        .map_err(|_| ConversionError::new("canonical hash memo mutex poisoned"))?;
    Ok(guard.get(&key).copied())
}

fn store_canonical_hash_memo(key: CanonicalHashMemoKey, value: u64) -> ConversionResult<()> {
    let mut guard = canonical_hash_memo()
        .lock()
        .map_err(|_| ConversionError::new("canonical hash memo mutex poisoned"))?;
    if guard.len() >= CANONICAL_HASH_MEMO_CAPACITY {
        guard.clear();
        crate::profiling::cache_event("conversion_hash_canonical_memo_clear");
    }
    guard.insert(key, value);
    Ok(())
}

fn canonicalize_program_for_hash(program: &Program) -> Program {
    let mut canonical = program.clone();
    canonicalize_function_names_and_entry(&mut canonical);
    let region_map = canonicalize_region_ids(&mut canonical);

    for function in &mut canonical.functions {
        canonicalize_function_value_ids(function);
    }
    for region in &mut canonical.regions {
        canonicalize_region_value_ids(region);
    }
    remap_program_region_refs(&mut canonical, &region_map);
    canonical
}

fn canonicalize_function_names_and_entry(program: &mut Program) {
    let mut names = HashMap::<String, String>::new();
    for (index, function) in program.functions.iter_mut().enumerate() {
        let canonical_name = format!("f{index}");
        names.insert(function.name.clone(), canonical_name.clone());
        function.name = canonical_name;
    }
    if let Some(mapped) = names.get(&program.entry) {
        program.entry = mapped.clone();
    }
}

fn canonicalize_region_ids(program: &mut Program) -> HashMap<usize, usize> {
    let mut map = HashMap::new();
    for (index, region) in program.regions.iter_mut().enumerate() {
        map.insert(region.id.0, index);
        region.id = RegionId(index);
    }
    map
}

fn remap_program_region_refs(program: &mut Program, region_map: &HashMap<usize, usize>) {
    for function in &mut program.functions {
        for instruction in &mut function.body {
            remap_instruction_regions(instruction, region_map);
        }
        for hint in &mut function.hints {
            for instruction in &mut hint.body {
                remap_instruction_regions(instruction, region_map);
            }
        }
    }
    for region in &mut program.regions {
        for instruction in &mut region.body {
            remap_instruction_regions(instruction, region_map);
        }
    }
}

fn remap_instruction_regions(instruction: &mut Instruction, region_map: &HashMap<usize, usize>) {
    match &mut instruction.op {
        Operation::Cond(spec) => {
            if let Some(mapped) = region_map.get(&spec.true_region.0).copied() {
                spec.true_region = RegionId(mapped);
            }
            if let Some(mapped) = region_map.get(&spec.false_region.0).copied() {
                spec.false_region = RegionId(mapped);
            }
        }
        Operation::While(spec) => {
            if let Some(mapped) = region_map.get(&spec.cond_region.0).copied() {
                spec.cond_region = RegionId(mapped);
            }
            if let Some(mapped) = region_map.get(&spec.body_region.0).copied() {
                spec.body_region = RegionId(mapped);
            }
        }
        Operation::Scan(spec) => {
            if let Some(mapped) = region_map.get(&spec.body_region.0).copied() {
                spec.body_region = RegionId(mapped);
            }
        }
        _ => {}
    }
}

fn canonicalize_function_value_ids(function: &mut Function) {
    let mut value_map = HashMap::<ValueId, ValueId>::new();
    let mut next_value = 0u32;

    for parameter in &mut function.parameter_ids {
        let old = *parameter;
        let new = ValueId(next_value);
        next_value = next_value.saturating_add(1);
        value_map.insert(old, new);
        *parameter = new;
    }

    for instruction in &mut function.body {
        canonicalize_instruction_id(instruction, &mut value_map, &mut next_value);
    }

    for (hint_index, hint) in function.hints.iter_mut().enumerate() {
        hint.id = hint_index as u32;
        for instruction in &mut hint.body {
            canonicalize_instruction_id(instruction, &mut value_map, &mut next_value);
        }
    }

    for instruction in &mut function.body {
        remap_instruction_values(instruction, &value_map);
    }

    for hint in &mut function.hints {
        remap_value_ids(hint.inputs.as_mut_slice(), &value_map);
        remap_value_ids(hint.exports.as_mut_slice(), &value_map);
        for instruction in &mut hint.body {
            remap_instruction_values(instruction, &value_map);
        }
    }

    remap_value_ids(function.result_ids.as_mut_slice(), &value_map);
}

fn canonicalize_region_value_ids(region: &mut Region) {
    let mut value_map = HashMap::<ValueId, ValueId>::new();

    let inferred = infer_region_parameter_ids(region);
    for (index, old_value) in inferred
        .into_iter()
        .take(region.parameters.len())
        .enumerate()
    {
        value_map.insert(old_value, ValueId(index as u32));
    }

    let mut next_value = region.parameters.len() as u32;
    for instruction in &mut region.body {
        canonicalize_instruction_id(instruction, &mut value_map, &mut next_value);
    }
    for instruction in &mut region.body {
        remap_instruction_values(instruction, &value_map);
    }
}

fn infer_region_parameter_ids(region: &Region) -> Vec<ValueId> {
    let defined = region
        .body
        .iter()
        .map(|instruction| instruction.id)
        .collect::<HashSet<_>>();
    let mut used = Vec::<ValueId>::new();

    for instruction in &region.body {
        for operand in &instruction.operands {
            match operand {
                Operand::Value(value) => used.push(*value),
                Operand::TupleElement { tuple, .. } => used.push(*tuple),
                Operand::Literal(_) => {}
            }
        }
    }

    used.sort_by_key(|value| value.0);
    used.dedup();

    let mut params = used
        .into_iter()
        .filter(|value| !defined.contains(value))
        .collect::<Vec<_>>();
    if params.len() == region.parameters.len() {
        params
    } else {
        params.clear();
        for index in 0..region.parameters.len() {
            params.push(ValueId(index as u32));
        }
        params
    }
}

fn canonicalize_instruction_id(
    instruction: &mut Instruction,
    value_map: &mut HashMap<ValueId, ValueId>,
    next_value: &mut u32,
) {
    let old = instruction.id;
    let new = if let Some(existing) = value_map.get(&old).copied() {
        existing
    } else {
        let assigned = ValueId(*next_value);
        *next_value = next_value.saturating_add(1);
        value_map.insert(old, assigned);
        assigned
    };
    instruction.id = new;
}

fn remap_instruction_values(instruction: &mut Instruction, value_map: &HashMap<ValueId, ValueId>) {
    for operand in &mut instruction.operands {
        match operand {
            Operand::Value(value) => {
                if let Some(mapped) = value_map.get(value).copied() {
                    *value = mapped;
                }
            }
            Operand::TupleElement { tuple, .. } => {
                if let Some(mapped) = value_map.get(tuple).copied() {
                    *tuple = mapped;
                }
            }
            Operand::Literal(_) => {}
        }
    }
}

fn remap_value_ids(ids: &mut [ValueId], value_map: &HashMap<ValueId, ValueId>) {
    for id in ids {
        if let Some(mapped) = value_map.get(id).copied() {
            *id = mapped;
        }
    }
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
