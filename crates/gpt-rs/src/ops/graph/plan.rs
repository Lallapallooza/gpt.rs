//! Plan caching primitives for lazy graph execution.

use std::collections::{HashMap, VecDeque};
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::Mutex;

use anyhow::Result;
use lru::LruCache;
use once_cell::sync::Lazy;
use serde::Serialize;

use crate::backend::hashing::fnv1a_hash;
use crate::backend::optimizer::PlanInputs;
use crate::backend::spec::{Operand, Operation, Program, TensorLiteral, TensorSpec, ValueId};
use crate::tensor::InputRole;

/// Default number of cached plans retained per arena before LRU eviction kicks in.
pub(super) const DEFAULT_PLAN_CACHE_CAPACITY: usize = 64;

/// Global cache that reuses optimized PTIR programs across graph arenas with matching signatures.
///
/// This avoids rerunning the optimizer pipeline when the structural signature matches, which is
/// especially important for workloads that create fresh arenas repeatedly (e.g. autoregressive
/// decoding in lazy mode).
static PROGRAM_CACHE: Lazy<Mutex<LruCache<PlanKey, CachedProgram>>> = Lazy::new(|| {
    Mutex::new(LruCache::new(
        NonZeroUsize::new(DEFAULT_PLAN_CACHE_CAPACITY).unwrap(),
    ))
});

/// Stable cache key built from the arena version and a deterministic graph signature.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(super) struct PlanKey {
    pub(super) version: u64,
    pub(super) hash: u64,
}

impl PlanKey {
    pub(super) fn new(
        backend: &str,
        version: u64,
        inputs: &[InputSignature],
        exports: &[ValueId],
        targets: &[ValueId],
        nodes: &[PlanNode],
    ) -> Result<Self> {
        let signature = SignatureData::from_components(inputs, exports, targets, nodes);
        let mut bytes = Vec::new();
        bytes.extend_from_slice(backend.as_bytes());
        bytes.push(0);
        bytes.extend(bincode::serialize(&signature)?);
        let hash = fnv1a_hash(&bytes);
        Ok(PlanKey { version, hash })
    }
}

#[derive(Clone, Debug)]
pub(super) struct InputSignature {
    pub(super) role: InputRole,
    pub(super) stable_id: Option<u128>,
    pub(super) spec: TensorSpec,
}

/// Lightweight representation of a node recorded inside the graph arena.
#[derive(Clone, Debug)]
pub(super) struct PlanNode {
    pub(super) value: ValueId,
    pub(super) op: Operation,
    pub(super) operands: Vec<Operand>,
    pub(super) spec: TensorSpec,
}

/// Specification for a graph input that will become a PTIR parameter.
#[derive(Clone, Debug)]
pub(super) struct ParameterSpec {
    pub(super) value: ValueId,
    pub(super) spec: TensorSpec,
}

/// Fully baked, reusable plan stored in the arena cache.
#[derive(Debug)]
pub(super) struct CachedPlan {
    pub(super) key: PlanKey,
    pub(super) program: Arc<Program>,
    pub(super) program_cache_hit: bool,
    pub(super) inputs: PlanInputs,
    pub(super) parameter_specs: Vec<ParameterSpec>,
    pub(super) parameter_values: Vec<ValueId>,
    pub(super) outputs: Vec<ValueId>,
    pub(super) exports: Vec<ValueId>,
}

impl CachedPlan {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        key: PlanKey,
        program: Arc<Program>,
        program_cache_hit: bool,
        inputs: PlanInputs,
        parameter_specs: Vec<ParameterSpec>,
        parameter_values: Vec<ValueId>,
        outputs: Vec<ValueId>,
        exports: Vec<ValueId>,
    ) -> Self {
        CachedPlan {
            key,
            program,
            program_cache_hit,
            inputs,
            parameter_specs,
            parameter_values,
            outputs,
            exports,
        }
    }
}

#[derive(Clone)]
pub(super) struct CachedProgram {
    pub(super) program: Arc<Program>,
    pub(super) inputs: PlanInputs,
}

pub(super) fn get_cached_program(key: &PlanKey) -> Option<CachedProgram> {
    let mut cache = PROGRAM_CACHE.lock().expect("program cache poisoned");
    cache.get(key).cloned()
}

pub(super) fn insert_cached_program(key: PlanKey, program: Arc<Program>, inputs: PlanInputs) {
    let mut cache = PROGRAM_CACHE.lock().expect("program cache poisoned");
    cache.put(key, CachedProgram { program, inputs });
}

/// In-memory LRU cache keyed by [`PlanKey`].
pub(super) struct PlanCache {
    capacity: usize,
    entries: HashMap<PlanKey, Arc<CachedPlan>>,
    order: VecDeque<PlanKey>,
}

impl PlanCache {
    pub(super) fn new(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        PlanCache {
            capacity,
            entries: HashMap::new(),
            order: VecDeque::new(),
        }
    }

    pub(super) fn get(&mut self, key: &PlanKey) -> Option<Arc<CachedPlan>> {
        if let Some(entry) = self.entries.get(key).cloned() {
            self.touch(key);
            Some(entry)
        } else {
            None
        }
    }

    pub(super) fn insert(&mut self, plan: Arc<CachedPlan>) {
        let key = plan.key;
        if self.entries.insert(key, Arc::clone(&plan)).is_some() {
            self.remove_from_order(&key);
        }
        self.order.push_back(key);
        while self.order.len() > self.capacity {
            if let Some(evicted) = self.order.pop_front() {
                self.entries.remove(&evicted);
                crate::profiling::cache_event("plan_cache_evict");
            }
        }
    }

    fn touch(&mut self, key: &PlanKey) {
        self.remove_from_order(key);
        self.order.push_back(*key);
    }

    fn remove_from_order(&mut self, key: &PlanKey) {
        if let Some(pos) = self.order.iter().position(|candidate| candidate == key) {
            self.order.remove(pos);
        }
    }
}

#[derive(Serialize)]
struct SignatureData {
    inputs: Vec<SignatureInput>,
    exports: Vec<ValueId>,
    targets: Vec<ValueId>,
    nodes: Vec<SignatureNode>,
}

impl SignatureData {
    fn from_components(
        inputs: &[InputSignature],
        exports: &[ValueId],
        targets: &[ValueId],
        nodes: &[PlanNode],
    ) -> Self {
        Canonicalizer::canonicalize(inputs, exports, targets, nodes)
    }
}

#[derive(Serialize)]
struct SignatureInput {
    role: InputRole,
    stable_id: Option<u128>,
    spec: TensorSpec,
}

#[derive(Serialize)]
struct SignatureNode {
    value: ValueId,
    op: Operation,
    operands: Vec<SignatureOperand>,
    spec: TensorSpec,
}

#[derive(Serialize)]
enum SignatureOperand {
    Value(ValueId),
    TupleElement { tuple: ValueId, index: usize },
    Literal(SignatureLiteral),
}

#[derive(Serialize)]
struct SignatureLiteral {
    spec: TensorSpec,
    byte_len: usize,
    byte_hash: u64,
}

struct Canonicalizer {
    mapping: HashMap<ValueId, ValueId>,
    next: u32,
}

impl Canonicalizer {
    fn new() -> Self {
        Canonicalizer {
            mapping: HashMap::new(),
            next: 0,
        }
    }

    fn canon_value(&mut self, value: ValueId) -> ValueId {
        *self.mapping.entry(value).or_insert_with(|| {
            let v = self.next;
            self.next += 1;
            ValueId(v)
        })
    }

    fn canon_operand(&mut self, operand: &Operand) -> SignatureOperand {
        match operand {
            Operand::Value(v) => SignatureOperand::Value(self.canon_value(*v)),
            Operand::TupleElement { tuple, index } => SignatureOperand::TupleElement {
                tuple: self.canon_value(*tuple),
                index: *index,
            },
            Operand::Literal(lit) => SignatureOperand::Literal(Self::literal_signature(lit)),
        }
    }

    fn literal_signature(lit: &TensorLiteral) -> SignatureLiteral {
        SignatureLiteral {
            spec: lit.spec.clone(),
            byte_len: lit.bytes.len(),
            byte_hash: fnv1a_hash(lit.bytes.as_ref()),
        }
    }

    fn canonicalize(
        inputs: &[InputSignature],
        exports: &[ValueId],
        targets: &[ValueId],
        nodes: &[PlanNode],
    ) -> SignatureData {
        let mut canon = Canonicalizer::new();

        let mut sig_nodes = Vec::with_capacity(nodes.len());
        for node in nodes {
            let value = canon.canon_value(node.value);
            let operands = node
                .operands
                .iter()
                .map(|op| canon.canon_operand(op))
                .collect();
            sig_nodes.push(SignatureNode {
                value,
                op: node.op.clone(),
                operands,
                spec: node.spec.clone(),
            });
        }

        let exports = exports.iter().map(|v| canon.canon_value(*v)).collect();
        let targets = targets.iter().map(|v| canon.canon_value(*v)).collect();

        SignatureData {
            inputs: inputs
                .iter()
                .map(|input| SignatureInput {
                    role: input.role,
                    stable_id: input.stable_id,
                    spec: input.spec.clone(),
                })
                .collect(),
            exports,
            targets,
            nodes: sig_nodes,
        }
    }
}

pub(super) fn ensure_targets_sorted(targets: &mut Vec<ValueId>) {
    targets.sort_by_key(|value| value.0);
    targets.dedup();
}

pub(super) fn ensure_exports_sorted(exports: &mut Vec<ValueId>) {
    exports.sort_by_key(|value| value.0);
    exports.dedup();
}

pub(super) fn ensure_parameters_sorted(parameters: &mut Vec<ParameterSpec>) {
    parameters.sort_by_key(|param| param.value.0);
    let mut dedup = Vec::with_capacity(parameters.len());
    for parameter in parameters.drain(..) {
        if dedup
            .last()
            .map(|last: &ParameterSpec| last.value == parameter.value)
            == Some(true)
        {
            continue;
        }
        dedup.push(parameter);
    }
    *parameters = dedup;
}
