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
use crate::backend::op_signature::operation_kind;
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
    pub(super) graph_hash: u64,
    pub(super) specialization_hash: u64,
    pub(super) input_binding_hash: u64,
    pub(super) shape_hash: u64,
    pub(super) dtype_hash: u64,
    pub(super) layout_hash: u64,
    pub(super) literal_hash: u64,
    pub(super) kv_bucket_hash: u64,
    pub(super) backend_option_hash: u64,
    pub(super) hash: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum CacheMissReason {
    ShapeSpecializationChange,
    DTypeChange,
    LayoutChange,
    LiteralValueOnlyChange,
    KvBucketChange,
    BackendOptionChange,
    Unknown,
}

impl PlanKey {
    pub(super) fn new_from_views(
        backend: &str,
        version: u64,
        inputs: &[InputSignature],
        exports: &[ValueId],
        targets: &[ValueId],
        node_views: &[PlanNodeView<'_>],
    ) -> Result<Self> {
        let signature = SignatureData::from_views(inputs, exports, targets, node_views);
        let graph_hash = hash_serializable(&GraphSignatureData::from_signature(&signature))?;
        let specialization = SpecializationData::from_signature(backend, &signature)?;
        let mut combined = [0u8; 16];
        combined[..8].copy_from_slice(&graph_hash.to_le_bytes());
        combined[8..].copy_from_slice(&specialization.specialization_hash.to_le_bytes());
        let hash = fnv1a_hash(&combined);
        Ok(PlanKey {
            version,
            graph_hash,
            specialization_hash: specialization.specialization_hash,
            input_binding_hash: specialization.input_binding_hash,
            shape_hash: specialization.shape_hash,
            dtype_hash: specialization.dtype_hash,
            layout_hash: specialization.layout_hash,
            literal_hash: specialization.literal_hash,
            kv_bucket_hash: specialization.kv_bucket_hash,
            backend_option_hash: specialization.backend_option_hash,
            hash,
        })
    }

    pub(super) fn with_version(self, version: u64) -> Self {
        Self { version, ..self }
    }

    pub(super) fn classify_change_from(self, previous: Option<PlanKey>) -> CacheMissReason {
        let Some(previous) = previous else {
            return CacheMissReason::Unknown;
        };
        if self.backend_option_hash != previous.backend_option_hash {
            return CacheMissReason::BackendOptionChange;
        }
        if self.kv_bucket_hash != previous.kv_bucket_hash && self.shape_hash != previous.shape_hash
        {
            return CacheMissReason::KvBucketChange;
        }
        if self.shape_hash != previous.shape_hash {
            return CacheMissReason::ShapeSpecializationChange;
        }
        if self.dtype_hash != previous.dtype_hash {
            return CacheMissReason::DTypeChange;
        }
        if self.layout_hash != previous.layout_hash {
            return CacheMissReason::LayoutChange;
        }
        if self.literal_hash != previous.literal_hash {
            return CacheMissReason::LiteralValueOnlyChange;
        }
        CacheMissReason::Unknown
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

#[derive(Clone, Copy)]
pub(super) struct PlanNodeView<'a> {
    pub(super) value: ValueId,
    pub(super) op: &'a Operation,
    pub(super) operands: &'a [Operand],
    pub(super) spec: &'a TensorSpec,
}

impl<'a> PlanNodeView<'a> {
    pub(super) fn new(
        value: ValueId,
        op: &'a Operation,
        operands: &'a [Operand],
        spec: &'a TensorSpec,
    ) -> Self {
        Self {
            value,
            op,
            operands,
            spec,
        }
    }
}

impl<'a> From<&'a PlanNode> for PlanNodeView<'a> {
    fn from(node: &'a PlanNode) -> Self {
        Self::new(node.value, &node.op, node.operands.as_slice(), &node.spec)
    }
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
    fn from_views(
        inputs: &[InputSignature],
        exports: &[ValueId],
        targets: &[ValueId],
        nodes: &[PlanNodeView<'_>],
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

#[derive(Serialize, Clone)]
struct SignatureLiteral {
    spec: TensorSpec,
    byte_len: usize,
    byte_hash: u64,
}

#[derive(Serialize)]
struct GraphSignatureData {
    inputs: Vec<GraphSignatureInput>,
    exports: Vec<ValueId>,
    targets: Vec<ValueId>,
    nodes: Vec<GraphSignatureNode>,
}

impl GraphSignatureData {
    fn from_signature(signature: &SignatureData) -> Self {
        Self {
            inputs: signature
                .inputs
                .iter()
                .map(|input| GraphSignatureInput {
                    role: input.role,
                    has_stable_id: input.stable_id.is_some(),
                })
                .collect(),
            exports: signature.exports.clone(),
            targets: signature.targets.clone(),
            nodes: signature
                .nodes
                .iter()
                .map(|node| GraphSignatureNode {
                    value: node.value,
                    op_kind: operation_kind(&node.op),
                    operands: node
                        .operands
                        .iter()
                        .map(|operand| match operand {
                            SignatureOperand::Value(value) => GraphSignatureOperand::Value(*value),
                            SignatureOperand::TupleElement { tuple, index } => {
                                GraphSignatureOperand::TupleElement {
                                    tuple: *tuple,
                                    index: *index,
                                }
                            }
                            SignatureOperand::Literal(literal) => {
                                GraphSignatureOperand::Literal(literal.spec.clone())
                            }
                        })
                        .collect(),
                })
                .collect(),
        }
    }
}

#[derive(Serialize)]
struct GraphSignatureInput {
    role: InputRole,
    has_stable_id: bool,
}

#[derive(Serialize)]
struct GraphSignatureNode {
    value: ValueId,
    op_kind: &'static str,
    operands: Vec<GraphSignatureOperand>,
}

#[derive(Serialize)]
enum GraphSignatureOperand {
    Value(ValueId),
    TupleElement { tuple: ValueId, index: usize },
    Literal(TensorSpec),
}

#[derive(Clone, Copy)]
struct SpecializationData {
    specialization_hash: u64,
    input_binding_hash: u64,
    shape_hash: u64,
    dtype_hash: u64,
    layout_hash: u64,
    literal_hash: u64,
    kv_bucket_hash: u64,
    backend_option_hash: u64,
}

impl SpecializationData {
    fn from_signature(backend: &str, signature: &SignatureData) -> Result<Self> {
        let input_binding_hash = hash_serializable(
            &signature
                .inputs
                .iter()
                .map(|input| (input.role, input.stable_id))
                .collect::<Vec<_>>(),
        )?;
        let shape_hash = hash_serializable(&ShapeSignature::from_signature(signature))?;
        let dtype_hash = hash_serializable(&DTypeSignature::from_signature(signature))?;
        let layout_hash = hash_serializable(&LayoutSignature::from_signature(signature))?;
        let literal_hash = hash_serializable(&LiteralSignature::from_signature(signature))?;
        let kv_bucket_hash = hash_serializable(&KvBucketSignature::from_signature(signature))?;
        let op_hash = hash_serializable(
            &signature
                .nodes
                .iter()
                .map(|node| &node.op)
                .collect::<Vec<_>>(),
        )?;
        let backend_option_hash = fnv1a_hash(backend.as_bytes());
        let specialization_hash = hash_serializable(&[
            op_hash,
            input_binding_hash,
            shape_hash,
            dtype_hash,
            layout_hash,
            literal_hash,
            kv_bucket_hash,
            backend_option_hash,
        ])?;
        Ok(Self {
            specialization_hash,
            input_binding_hash,
            shape_hash,
            dtype_hash,
            layout_hash,
            literal_hash,
            kv_bucket_hash,
            backend_option_hash,
        })
    }
}

#[derive(Serialize)]
struct ShapeSignature {
    input_shapes: Vec<crate::backend::spec::Shape>,
    output_shapes: Vec<crate::backend::spec::Shape>,
    literal_shapes: Vec<crate::backend::spec::Shape>,
}

impl ShapeSignature {
    fn from_signature(signature: &SignatureData) -> Self {
        let mut literal_shapes = Vec::new();
        for node in &signature.nodes {
            for operand in &node.operands {
                if let SignatureOperand::Literal(literal) = operand {
                    literal_shapes.push(literal.spec.shape.clone());
                }
            }
        }
        Self {
            input_shapes: signature
                .inputs
                .iter()
                .map(|input| input.spec.shape.clone())
                .collect(),
            output_shapes: signature
                .nodes
                .iter()
                .map(|node| node.spec.shape.clone())
                .collect(),
            literal_shapes,
        }
    }
}

#[derive(Serialize)]
struct DTypeSignature {
    input_dtypes: Vec<crate::backend::spec::DType>,
    output_dtypes: Vec<crate::backend::spec::DType>,
    literal_dtypes: Vec<crate::backend::spec::DType>,
}

impl DTypeSignature {
    fn from_signature(signature: &SignatureData) -> Self {
        let mut literal_dtypes = Vec::new();
        for node in &signature.nodes {
            for operand in &node.operands {
                if let SignatureOperand::Literal(literal) = operand {
                    literal_dtypes.push(literal.spec.dtype);
                }
            }
        }
        Self {
            input_dtypes: signature
                .inputs
                .iter()
                .map(|input| input.spec.dtype)
                .collect(),
            output_dtypes: signature.nodes.iter().map(|node| node.spec.dtype).collect(),
            literal_dtypes,
        }
    }
}

#[derive(Serialize)]
struct LayoutSignature {
    transpose_perms: Vec<Vec<usize>>,
    dot_dims: Vec<DotLayoutSignature>,
}

impl LayoutSignature {
    fn from_signature(signature: &SignatureData) -> Self {
        let mut transpose_perms = Vec::new();
        let mut dot_dims = Vec::new();
        for node in &signature.nodes {
            match &node.op {
                Operation::Transpose(spec) => transpose_perms.push(spec.perm.clone()),
                Operation::DotGeneral(spec) => dot_dims.push(DotLayoutSignature {
                    batch_lhs: spec.batch_lhs.clone(),
                    batch_rhs: spec.batch_rhs.clone(),
                    contract_lhs: spec.contract_lhs.clone(),
                    contract_rhs: spec.contract_rhs.clone(),
                }),
                _ => {}
            }
        }
        Self {
            transpose_perms,
            dot_dims,
        }
    }
}

#[derive(Serialize)]
struct DotLayoutSignature {
    batch_lhs: Vec<usize>,
    batch_rhs: Vec<usize>,
    contract_lhs: Vec<usize>,
    contract_rhs: Vec<usize>,
}

#[derive(Serialize)]
struct LiteralSignature {
    literals: Vec<SignatureLiteral>,
}

impl LiteralSignature {
    fn from_signature(signature: &SignatureData) -> Self {
        let mut literals = Vec::new();
        for node in &signature.nodes {
            for operand in &node.operands {
                if let SignatureOperand::Literal(literal) = operand {
                    literals.push(literal.clone());
                }
            }
        }
        Self { literals }
    }
}

#[derive(Serialize)]
struct KvBucketSignature {
    buckets: Vec<usize>,
}

impl KvBucketSignature {
    fn from_signature(signature: &SignatureData) -> Self {
        let mut buckets = Vec::new();
        for input in &signature.inputs {
            collect_bucket_dims(input.spec.shape.dims(), &mut buckets);
        }
        for node in &signature.nodes {
            collect_bucket_dims(node.spec.shape.dims(), &mut buckets);
            for operand in &node.operands {
                if let SignatureOperand::Literal(literal) = operand {
                    collect_bucket_dims(literal.spec.shape.dims(), &mut buckets);
                }
            }
        }
        buckets.sort_unstable();
        buckets.dedup();
        Self { buckets }
    }
}

fn collect_bucket_dims(dims: &[crate::backend::spec::Dimension], out: &mut Vec<usize>) {
    for dim in dims {
        if let crate::backend::spec::Dimension::Static(value) = dim {
            if *value >= 8 && value.is_power_of_two() {
                out.push(*value);
            }
        }
    }
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
        nodes: &[PlanNodeView<'_>],
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

fn hash_serializable<T: Serialize>(value: &T) -> Result<u64> {
    let bytes = bincode::serialize(value)?;
    Ok(fnv1a_hash(&bytes))
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
