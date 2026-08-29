//! Plan caching primitives for lazy graph execution.

use std::collections::{HashMap, VecDeque};
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::Mutex;

use std::hash::{BuildHasher, Hash, Hasher};

use anyhow::Result;
use lru::LruCache;
use once_cell::sync::Lazy;

use crate::backend::optimizer::PlanInputs;
use crate::backend::spec::{Dimension, Operand, Operation, Program, Shape, TensorSpec, ValueId};
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
    GraphStructureChange,
    Unknown,
}

impl PlanKey {
    /// Computes the key of `graph` in one pass, without allocating. The key covers the graph
    /// structure (input roles, op kinds, operand topology) and every specialization component.
    /// The values that a call requests are not part of the key, because plans bind outputs by
    /// node rank.
    pub(super) fn new<'n, F>(backend: &str, version: u64, graph: &PlanGraph<'_, F>) -> Result<Self>
    where
        F: Fn(ValueId) -> Option<PlanNodeView<'n>>,
    {
        debug_assert!(graph.nodes.windows(2).all(|pair| pair[0].0 < pair[1].0));
        debug_assert!(graph
            .input_values
            .windows(2)
            .all(|pair| pair[0].0 < pair[1].0));
        debug_assert_eq!(graph.inputs.len(), graph.input_values.len());

        let mut body = key_hasher();
        let mut ops = key_hasher();
        let mut bindings = key_hasher();
        let mut shapes = key_hasher();
        let mut dtypes = key_hasher();
        let mut layouts = key_hasher();
        let mut literals = key_hasher();
        let mut buckets = 0u64;

        body.write_usize(graph.inputs.len());
        for input in graph.inputs {
            input.role.hash(&mut body);
            body.write_u8(u8::from(input.stable_id.is_some()));
            input.role.hash(&mut bindings);
            input.stable_id.hash(&mut bindings);
            input.spec.shape.hash(&mut shapes);
            input.spec.dtype.hash(&mut dtypes);
            buckets |= bucket_mask(&input.spec.shape);
        }

        body.write_usize(graph.nodes.len());
        for &node_value in graph.nodes {
            let node = (graph.node_view)(node_value)
                .ok_or_else(|| anyhow::anyhow!("missing node for value {node_value:?}"))?;
            std::mem::discriminant(node.op).hash(&mut body);
            body.write_usize(node.operands.len());
            for operand in node.operands {
                match operand {
                    Operand::Value(value) => {
                        body.write_u8(0);
                        write_ref(&mut body, graph.value_ref(*value));
                    }
                    Operand::TupleElement { tuple, index } => {
                        body.write_u8(1);
                        write_ref(&mut body, graph.value_ref(*tuple));
                        body.write_usize(*index);
                    }
                    Operand::Literal(literal) => {
                        body.write_u8(2);
                        literal.spec.hash(&mut body);
                        literal.spec.shape.hash(&mut shapes);
                        literal.spec.dtype.hash(&mut dtypes);
                        buckets |= bucket_mask(&literal.spec.shape);
                        literal.spec.hash(&mut literals);
                        literal.bytes.hash(&mut literals);
                    }
                }
            }
            node.op.hash(&mut ops);
            node.spec.shape.hash(&mut shapes);
            node.spec.dtype.hash(&mut dtypes);
            buckets |= bucket_mask(&node.spec.shape);
            match node.op {
                Operation::Transpose(spec) => spec.perm.hash(&mut layouts),
                Operation::DotGeneral(spec) => (
                    &spec.batch_lhs,
                    &spec.batch_rhs,
                    &spec.contract_lhs,
                    &spec.contract_rhs,
                )
                    .hash(&mut layouts),
                _ => {}
            }
        }

        let mut backend_hasher = key_hasher();
        backend.hash(&mut backend_hasher);
        let graph_hash = body.finish();
        let input_binding_hash = bindings.finish();
        let shape_hash = shapes.finish();
        let dtype_hash = dtypes.finish();
        let layout_hash = layouts.finish();
        let literal_hash = literals.finish();
        let backend_option_hash = backend_hasher.finish();
        let specialization_hash = combine(&[
            ops.finish(),
            input_binding_hash,
            shape_hash,
            dtype_hash,
            layout_hash,
            literal_hash,
            buckets,
            backend_option_hash,
        ]);
        Ok(PlanKey {
            version,
            graph_hash,
            specialization_hash,
            input_binding_hash,
            shape_hash,
            dtype_hash,
            layout_hash,
            literal_hash,
            kv_bucket_hash: buckets,
            backend_option_hash,
            hash: combine(&[graph_hash, specialization_hash]),
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
        if self.graph_hash != previous.graph_hash {
            return CacheMissReason::GraphStructureChange;
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

/// Borrowed view of a pending node: its operation, operands, and output spec.
#[derive(Clone, Copy)]
pub(super) struct PlanNodeView<'a> {
    pub(super) op: &'a Operation,
    pub(super) operands: &'a [Operand],
    pub(super) spec: &'a TensorSpec,
}

impl<'a> PlanNodeView<'a> {
    pub(super) fn new(op: &'a Operation, operands: &'a [Operand], spec: &'a TensorSpec) -> Self {
        Self { op, operands, spec }
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
    pub(super) requested_outputs: Vec<ValueId>,
    /// Values that receive the program results, in result order.
    pub(super) program_outputs: Vec<ValueId>,
    /// Positions of `program_outputs` in the pending nodes of the plan, followed by its parameters.
    /// Both lists are sorted by value id. The program addresses its outputs by these positions in
    /// any graph with the same key.
    pub(super) output_ranks: Vec<usize>,
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
        requested_outputs: Vec<ValueId>,
        program_outputs: Vec<ValueId>,
        output_ranks: Vec<usize>,
        exports: Vec<ValueId>,
    ) -> Self {
        CachedPlan {
            key,
            program,
            program_cache_hit,
            inputs,
            parameter_specs,
            parameter_values,
            requested_outputs,
            program_outputs,
            output_ranks,
            exports,
        }
    }
}

#[derive(Clone)]
pub(super) struct CachedProgram {
    pub(super) program: Arc<Program>,
    pub(super) inputs: PlanInputs,
    /// See [`CachedPlan::output_ranks`].
    pub(super) output_ranks: Vec<usize>,
}

pub(super) fn get_cached_program(key: &PlanKey) -> Option<CachedProgram> {
    let mut cache = PROGRAM_CACHE.lock().expect("program cache poisoned");
    cache.get(key).cloned()
}

pub(super) fn insert_cached_program(
    key: PlanKey,
    program: Arc<Program>,
    inputs: PlanInputs,
    output_ranks: Vec<usize>,
) {
    let mut cache = PROGRAM_CACHE.lock().expect("program cache poisoned");
    cache.put(
        key,
        CachedProgram {
            program,
            inputs,
            output_ranks,
        },
    );
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

/// Borrowed view of the pending graph that a plan key is computed from.
///
/// `nodes` and `input_values` are sorted by value id. The key canonicalizes each value by its
/// position in those slices. Program parameters and results use the same order.
pub(super) struct PlanGraph<'a, F> {
    /// Input signatures in parameter order (aligned with `input_values`).
    pub(super) inputs: &'a [InputSignature],
    pub(super) input_values: &'a [ValueId],
    pub(super) nodes: &'a [ValueId],
    /// Looks up a pending node by value id.
    pub(super) node_view: F,
}

/// Canonical reference to a value inside a [`PlanGraph`].
#[derive(Clone, Copy)]
enum ValueRef {
    Node(usize),
    Input(usize),
    /// Neither a pending node nor an input, for example an export that is already materialized.
    External,
}

impl<F> PlanGraph<'_, F> {
    fn value_ref(&self, value: ValueId) -> ValueRef {
        if let Ok(index) = self.nodes.binary_search_by_key(&value.0, |v| v.0) {
            ValueRef::Node(index)
        } else if let Ok(index) = self.input_values.binary_search_by_key(&value.0, |v| v.0) {
            ValueRef::Input(index)
        } else {
            ValueRef::External
        }
    }
}

const PLAN_KEY_SEED: u64 = 0x5054_4952_706c_616e;

/// Plan-key hasher: foldhash that keeps every write in stream order.
///
/// foldhash buffers integer writes but folds byte slices at once, so an integer write and a
/// following byte write could swap places without a change in the hash. Only `write` is
/// overridden, so integers also go through it as bytes.
#[derive(Clone)]
struct KeyHasher(foldhash::quality::FoldHasher);

fn key_hasher() -> KeyHasher {
    KeyHasher(foldhash::quality::FixedState::with_seed(PLAN_KEY_SEED).build_hasher())
}

impl Hasher for KeyHasher {
    fn write(&mut self, bytes: &[u8]) {
        self.0.write(bytes);
    }

    fn finish(&self) -> u64 {
        self.0.finish()
    }
}

fn write_ref(hasher: &mut KeyHasher, value: ValueRef) {
    match value {
        ValueRef::Node(index) => {
            hasher.write_u8(1);
            hasher.write_usize(index);
        }
        ValueRef::Input(index) => {
            hasher.write_u8(2);
            hasher.write_usize(index);
        }
        ValueRef::External => hasher.write_u8(3),
    }
}

/// Sets bit `k` when a static dimension equals `2^k` and is at least 8. The result is the set of
/// power-of-two buckets that a KV cache capacity can land in.
fn bucket_mask(shape: &Shape) -> u64 {
    shape.dims().iter().fold(0, |mask, dim| match dim {
        Dimension::Static(value) if *value >= 8 && value.is_power_of_two() => {
            mask | (1u64 << value.trailing_zeros())
        }
        _ => mask,
    })
}

fn combine(parts: &[u64]) -> u64 {
    let mut hasher = key_hasher();
    for part in parts {
        hasher.write_u64(*part);
    }
    hasher.finish()
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
