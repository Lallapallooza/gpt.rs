//! Shared arena that stores lazily constructed operation graphs.
//!
//! The graph arena is the central orchestrator for lazy tensor graph execution. It maintains
//! a mutable graph of pending operations, compiles them into backend programs on demand, and
//! caches compiled plans for reuse across multiple executions with matching signatures.
//!
//! ## Architecture
//!
//! ```text
//! DeviceTensor
//!      |
//!      | contains Arc<GraphArena>
//!      v
//! GraphArena
//!      |
//!      +-- GraphInner (nodes, parameters, exports)
//!      |
//!      +-- PlanCache (compiled program cache)
//!      |
//!      +-- Optimizer (graph rewriting)
//!      |
//!      +-- Backend (program execution)
//! ```
//!
//! ## Lazy Execution Model
//!
//! 1. **Capture**: Functional ops call `arena.capture()` to record graph nodes
//! 2. **Export**: Nodes are marked as exported when they flow into DeviceTensor results
//! 3. **Materialize**: When a tensor's data is accessed, the arena compiles a plan
//! 4. **Cache**: Compiled plans are cached by signature (parameter specs + exported nodes)
//! 5. **Execute**: Backend runs the program and stores results
//!
//! ## Version-Based Staleness
//!
//! The arena maintains a version counter that increments whenever new nodes are added.
//! Cached plans store the version they were compiled against. When a plan is reused,
//! the arena checks if the graph has been modified since compilation. If so, the plan
//! is recompiled to include any new intermediate results.
//!
//! This allows multiple operations to share graph state while invalidating stale caches
//! automatically.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::{
    atomic::{AtomicUsize, Ordering as AtomicOrdering},
    Arc, Mutex,
};
use std::time::Instant;

use anyhow::{anyhow, bail, Result};

use crate::backend::optimizer::{
    default_optimizer, OptimizeConfig, OptimizeContext, OptimizeServices, Optimizer,
};
use crate::backend::param_resolver::{InMemoryParamResolver, ParamResolver};
use crate::backend::spec::{
    Function, Operand, PortableBackend, Program, ProgramBuilder, TensorSpec, ValueId, ValueType,
};
use crate::ops::trace::{
    self, ProgramCacheInfo, ProgramContext, ProgramKind, ProgramStats, ProgramStatus,
};
use crate::tensor::{DeviceTensor, InputRole};

use super::builder::GraphBuilder;
use super::plan::{
    ensure_exports_sorted, ensure_parameters_sorted, ensure_targets_sorted, get_cached_program,
    insert_cached_program, CacheMissReason, CachedPlan, InputSignature, ParameterSpec, PlanCache,
    PlanKey, PlanNode, PlanNodeView, DEFAULT_PLAN_CACHE_CAPACITY,
};
use super::state::{GraphInner, NodeState, ParameterRecord};

/// Central storage for lazy tensor graphs built on top of a single backend instance.
pub struct GraphArena<B: PortableBackend + 'static> {
    backend: Arc<B>,
    pub(super) param_resolver: Arc<dyn ParamResolver<Handle = B::TensorHandle>>,
    inner: Mutex<GraphInner<B>>,
    plan_cache: PlanCacheState,
    optimizer: Option<Arc<dyn Optimizer<B>>>,
    miss_telemetry: Mutex<CacheMissTelemetry>,
    id: usize,
}

static ARENA_ID_COUNTER: AtomicUsize = AtomicUsize::new(0);

#[derive(Default)]
struct CacheMissTelemetry {
    last_plan_by_graph: HashMap<u64, PlanKey>,
    last_program_by_graph: HashMap<u64, PlanKey>,
    last_plan_any: Option<PlanKey>,
    last_program_any: Option<PlanKey>,
}

/// Configures how an arena caches compiled plans and rewrites graphs.
pub enum CachePolicy<B: PortableBackend + 'static> {
    Disabled,
    LazyPrograms {
        capacity: usize,
        optimizer: Option<Arc<dyn Optimizer<B>>>,
    },
}

impl<B: PortableBackend + 'static> CachePolicy<B> {
    fn into_components(self) -> (PlanCacheState, Option<Arc<dyn Optimizer<B>>>) {
        match self {
            CachePolicy::Disabled => (PlanCacheState::Disabled, None),
            CachePolicy::LazyPrograms {
                capacity,
                optimizer,
            } => {
                let cache = PlanCacheState::Enabled(Mutex::new(PlanCache::new(capacity)));
                (cache, optimizer)
            }
        }
    }
}

impl<B: PortableBackend + 'static> Default for CachePolicy<B> {
    fn default() -> Self {
        CachePolicy::LazyPrograms {
            capacity: DEFAULT_PLAN_CACHE_CAPACITY,
            optimizer: None,
        }
    }
}

/// Internal state for managing the plan cache.
///
/// Wraps an optional `PlanCache` to support both caching-enabled and caching-disabled
/// execution modes. The `Enabled` variant uses a mutex for thread-safe access.
enum PlanCacheState {
    /// No caching - every materialization recompiles the plan.
    Disabled,
    /// Caching enabled - compiled plans are stored and reused by signature.
    Enabled(Mutex<PlanCache>),
}

impl PlanCacheState {
    fn is_enabled(&self) -> bool {
        matches!(self, PlanCacheState::Enabled(_))
    }

    fn get(&self, key: &PlanKey) -> Option<Arc<CachedPlan>> {
        match self {
            PlanCacheState::Disabled => None,
            PlanCacheState::Enabled(cache) => {
                let mut guard = cache.lock().expect("plan cache poisoned");
                guard.get(key)
            }
        }
    }

    fn insert(&self, plan: Arc<CachedPlan>) {
        if let PlanCacheState::Enabled(cache) = self {
            let mut guard = cache.lock().expect("plan cache poisoned");
            guard.insert(plan);
        }
    }
}

impl<B: PortableBackend + 'static> GraphArena<B> {
    /// Creates a new arena wrapping the provided backend.
    pub fn new(backend: Arc<B>) -> Arc<Self> {
        Self::with_policy(backend, CachePolicy::default())
    }

    /// Creates a new arena with the provided cache policy.
    pub fn with_policy(backend: Arc<B>, policy: CachePolicy<B>) -> Arc<Self> {
        let id = ARENA_ID_COUNTER.fetch_add(1, AtomicOrdering::Relaxed);
        let (plan_cache, optimizer) = policy.into_components();
        let backend_pipeline = backend.pipeline();
        let optimizer = match &plan_cache {
            PlanCacheState::Enabled(_) => {
                optimizer.or_else(|| Some(default_optimizer(backend_pipeline)))
            }
            PlanCacheState::Disabled => None,
        };
        let param_resolver = backend.param_resolver().unwrap_or_else(|| {
            Arc::new(InMemoryParamResolver::<B::TensorHandle>::new())
                as Arc<dyn ParamResolver<Handle = B::TensorHandle>>
        });
        Arc::new(GraphArena {
            backend,
            param_resolver,
            inner: Mutex::new(GraphInner::new()),
            plan_cache,
            optimizer,
            miss_telemetry: Mutex::new(CacheMissTelemetry::default()),
            id,
        })
    }

    /// Returns the underlying backend handle.
    pub fn backend(&self) -> Arc<B> {
        Arc::clone(&self.backend)
    }

    /// Marks a value identifier as exported outside of a capture closure.
    pub fn export(&self, value: ValueId) {
        let mut inner = self.inner.lock().expect("graph arena poisoned");
        inner.exports.insert(value);
    }

    /// Removes a value identifier from the exported set if it is no longer needed.
    pub fn unexport(&self, value: ValueId) {
        let mut inner = self.inner.lock().expect("graph arena poisoned");
        inner.exports.remove(&value);
    }

    /// Attempts to retrieve the tensor specification recorded for the provided value identifier.
    pub fn tensor_spec_for(&self, value: ValueId) -> Option<TensorSpec> {
        let inner = self.inner.lock().expect("graph arena poisoned");
        if let Some(record) = inner.nodes.get(&value) {
            return Some(record.spec.clone());
        }
        inner.parameter(value).map(|param| param.spec.clone())
    }

    pub(crate) fn try_ready_handle(&self, value: ValueId) -> Option<B::TensorHandle> {
        let inner = self.inner.lock().expect("graph arena poisoned");
        if let Some(record) = inner.nodes.get(&value) {
            return match &record.state {
                NodeState::Ready(handle) => Some(handle.clone()),
                NodeState::Pending => None,
            };
        }
        inner
            .parameter(value)
            .and_then(|param| self.resolve_parameter_handle(&inner, param).ok())
    }

    fn resolve_param_handle_by_stable_id(
        &self,
        inner: &GraphInner<B>,
        stable_id: u128,
    ) -> Result<B::TensorHandle> {
        if let Some(handle) = self.param_resolver.get(stable_id) {
            return Ok(handle);
        }

        if let Some(param) = inner
            .parameters
            .iter()
            .find(|record| record.role == InputRole::Param && record.stable_id == Some(stable_id))
        {
            return self.resolve_parameter_handle(inner, param);
        }

        if let Some(source) = inner.param_sources.get(&stable_id) {
            return source.source.load(source.base_id);
        }

        Err(anyhow!(
            "missing param {:?} in resolver/source for backend {}",
            stable_id,
            self.backend.backend_name()
        ))
    }

    fn resolve_parameter_handle(
        &self,
        inner: &GraphInner<B>,
        param: &ParameterRecord<B>,
    ) -> Result<B::TensorHandle> {
        match param.role {
            InputRole::Arg => param
                .handle
                .clone()
                .ok_or_else(|| anyhow!("arg value {:?} missing materialized handle", param.value)),
            InputRole::Param => {
                let stable_id = param
                    .stable_id
                    .ok_or_else(|| anyhow!("param value {:?} missing stable id", param.value))?;
                if let Some(handle) = self.param_resolver.get(stable_id) {
                    return Ok(handle);
                }
                if let Some(handle) = param.handle.clone() {
                    self.param_resolver.set(stable_id, handle.clone());
                    return Ok(handle);
                }
                if let Some(source) = inner.param_sources.get(&stable_id) {
                    return source.source.load(source.base_id);
                }
                Err(anyhow!(
                    "missing param source for stable id {:?} on backend {}",
                    stable_id,
                    self.backend.backend_name()
                ))
            }
        }
    }

    fn collect_target_handles(
        &self,
        inner: &GraphInner<B>,
        targets: &[ValueId],
    ) -> Result<Vec<B::TensorHandle>> {
        let mut handles = Vec::with_capacity(targets.len());
        for value in targets {
            if let Some(node) = inner.nodes.get(value) {
                match &node.state {
                    NodeState::Ready(handle) => handles.push(handle.clone()),
                    NodeState::Pending => {
                        return Err(anyhow!("value {:?} pending after program execution", value));
                    }
                }
            } else if let Some(param) = inner.parameter(*value) {
                handles.push(self.resolve_parameter_handle(inner, param)?);
            } else {
                return Err(anyhow!("value {:?} not registered", value));
            }
        }
        Ok(handles)
    }

    /// Captures a sequence of graph edits, exposing a [`GraphBuilder`] to the caller.
    /// The edits remain isolated to this arena and can be replayed later when materialisation is requested.
    pub fn capture<R, F>(self: &Arc<Self>, f: F) -> Result<R>
    where
        F: FnOnce(&mut GraphBuilder<B>) -> Result<R>,
    {
        let mut inner = self.inner.lock().expect("graph arena poisoned");
        let mut builder = GraphBuilder {
            arena: Arc::clone(self),
            inner: &mut *inner,
        };
        f(&mut builder)
    }

    /// Materialises the requested value, executing pending nodes as necessary.
    /// Pending nodes are compiled into a single backend program so repeated materialisations stay efficient.
    pub fn flush_until(&self, value: ValueId) -> Result<<B as PortableBackend>::TensorHandle> {
        let mut handles = self.materialize_values(&[value])?;
        handles
            .pop()
            .ok_or_else(|| anyhow!("value missing after execution"))
    }

    /// Materialises all requested value identifiers, executing pending nodes at most once.
    pub fn materialize_values(
        &self,
        values: &[ValueId],
    ) -> Result<Vec<<B as PortableBackend>::TensorHandle>> {
        if values.is_empty() {
            return Ok(Vec::new());
        }

        loop {
            match self.prepare_plan(values)? {
                PrepareResult::AllReady(handles) => return Ok(handles),
                PrepareResult::NeedsPlan(context) => {
                    let context = *context;
                    let (plan, plan_cache_hit) = {
                        let _scope = crate::profiling::compile_scope(if values.len() == 1 {
                            "graph::compile(single)"
                        } else {
                            "graph::compile(multi)"
                        });
                        self.get_or_build_plan(context)?
                    };
                    let entry_inputs = self.collect_entry_inputs(&plan, None)?;
                    match self.try_execute_plan(&plan, values, entry_inputs, plan_cache_hit) {
                        Ok(handles) => return Ok(handles),
                        Err(err) => {
                            if err.downcast_ref::<StalePlanError>().is_some() {
                                crate::profiling::cache_event("stale_plan_retry");
                                continue;
                            }
                            return Err(err);
                        }
                    }
                }
            }
        }
    }

    pub fn compile(self: &Arc<Self>, values: &[ValueId]) -> Result<CompiledGraph<B>> {
        if values.is_empty() {
            bail!("cannot compile an empty value set");
        }

        match self.prepare_plan(values)? {
            PrepareResult::AllReady(_) => {
                bail!("requested values are already materialised; nothing to compile")
            }
            PrepareResult::NeedsPlan(context) => {
                let context = *context;
                let arena_version = context.arena_version;
                let (plan, _plan_cache_hit) = self.get_or_build_plan(context)?;
                Ok(CompiledGraph::new(
                    Arc::clone(self),
                    plan,
                    values.to_vec(),
                    arena_version,
                ))
            }
        }
    }

    /// Imports a backend [`Function`] into the graph, returning the realised output values.
    /// This is used by higher-level functionals to stitch precompiled programs into lazy graphs.
    pub fn capture_function(
        self: &Arc<Self>,
        function: &Function,
        params: &[&DeviceTensor<B>],
    ) -> Result<Vec<ValueId>> {
        if function.parameter_ids.len() != params.len() {
            bail!(
                "function parameter count mismatch: expected {}, got {}",
                function.parameter_ids.len(),
                params.len()
            );
        }

        self.capture(|ctx| {
            let mut value_map: HashMap<ValueId, ValueId> = HashMap::new();

            for (param_id, tensor) in function.parameter_ids.iter().zip(params.iter()) {
                let imported = ctx.import(tensor)?;
                value_map.insert(*param_id, imported);
            }

            for instruction in &function.body {
                let operands = instruction
                    .operands
                    .iter()
                    .map(|operand| match operand {
                        Operand::Value(id) => value_map
                            .get(id)
                            .copied()
                            .map(Operand::Value)
                            .ok_or_else(|| anyhow!("missing value mapping for {:?}", id)),
                        Operand::TupleElement { tuple, index } => value_map
                            .get(tuple)
                            .copied()
                            .map(|mapped| Operand::TupleElement {
                                tuple: mapped,
                                index: *index,
                            })
                            .ok_or_else(|| anyhow!("missing tuple mapping for {:?}", tuple)),
                        Operand::Literal(lit) => Ok(Operand::Literal(lit.clone())),
                    })
                    .collect::<Result<Vec<_>>>()?;

                let tensor_spec = match &instruction.output {
                    ValueType::Tensor(spec) => spec.clone(),
                    _ => bail!("non-tensor outputs are not supported in capture"),
                };

                let new_id = ctx.emit(instruction.op.clone(), operands, tensor_spec);
                value_map.insert(instruction.id, new_id);
            }

            let mut results = Vec::with_capacity(function.result_ids.len());
            for original in &function.result_ids {
                let mapped = value_map
                    .get(original)
                    .copied()
                    .ok_or_else(|| anyhow!("missing result mapping for {:?}", original))?;
                results.push(mapped);
            }
            Ok(results)
        })
    }

    /// Walks the dependency graph, determines which nodes need execution, and packages the
    /// result into an [`ExecutionPlan`]. The method:
    /// - short-circuits when the target value is already materialised;
    /// - computes the transitive closure of dependencies using [`collect_dependencies`];
    /// - assigns new parameter ids for ready inputs or captured parameters;
    /// - emits operations in insertion order so dependencies run before their consumers.
    fn prepare_plan(&self, targets: &[ValueId]) -> Result<PrepareResult<B>> {
        let _scope = crate::profiling::compile_scope(if targets.len() == 1 {
            "graph::prepare_plan(single)"
        } else {
            "graph::prepare_plan(multi)"
        });
        let inner = self.inner.lock().expect("graph arena poisoned");

        let mut ready_handles: Vec<Option<B::TensorHandle>> = Vec::with_capacity(targets.len());
        let mut has_pending = false;

        {
            let _scope = crate::profiling::compile_scope("graph::prepare_plan.ready_scan");
            for value in targets {
                if let Some(node) = inner.nodes.get(value) {
                    match &node.state {
                        NodeState::Ready(handle) => ready_handles.push(Some(handle.clone())),
                        NodeState::Pending => {
                            has_pending = true;
                            ready_handles.push(None);
                        }
                    }
                } else if let Some(param) = inner.parameter(*value) {
                    let handle = self.resolve_parameter_handle(&inner, param)?;
                    ready_handles.push(Some(handle));
                } else {
                    return Err(anyhow!("value {:?} not registered in graph", value));
                }
            }
        }

        if !has_pending {
            let handles = ready_handles
                .into_iter()
                .map(|entry| entry.expect("ready handle missing"))
                .collect();
            return Ok(PrepareResult::AllReady(handles));
        }

        let mut requested = Vec::new();
        let mut seen = HashSet::new();
        {
            let _scope = crate::profiling::compile_scope("graph::prepare_plan.requested");
            for value in inner.exports.iter() {
                if seen.insert(*value) {
                    requested.push(*value);
                }
            }
            for value in targets {
                if seen.insert(*value) {
                    requested.push(*value);
                }
            }
        }

        let mut pending_targets = Vec::new();
        {
            let _scope = crate::profiling::compile_scope("graph::prepare_plan.pending_targets");
            for value in &requested {
                if let Some(node) = inner.nodes.get(value) {
                    if matches!(node.state, NodeState::Pending) {
                        pending_targets.push(*value);
                    }
                }
            }
        }

        if pending_targets.is_empty() {
            let handles = ready_handles
                .into_iter()
                .map(|entry| entry.expect("ready handle missing"))
                .collect();
            return Ok(PrepareResult::AllReady(handles));
        }

        let mut pending = HashSet::new();
        let mut inputs = HashSet::new();
        {
            let _scope = crate::profiling::compile_scope("graph::prepare_plan.dependency_closure");
            for value in &requested {
                if inner.nodes.contains_key(value) {
                    collect_dependencies(&inner, *value, &mut pending, &mut inputs)?;
                }
            }
        }

        let mut input_values: Vec<_> = inputs.into_iter().collect();
        let mut bindings: Vec<(ValueId, TensorSpec, InputRole, Option<u128>)>;
        {
            let _scope = crate::profiling::compile_scope("graph::prepare_plan.input_bindings");
            input_values.sort_by_key(|value| value.0);

            bindings = Vec::with_capacity(input_values.len());
            for value in &input_values {
                if let Some(param) = inner.parameter(*value) {
                    let stable_id = if param.role == InputRole::Param {
                        param.stable_id
                    } else {
                        None
                    };
                    bindings.push((*value, param.spec.clone(), param.role, stable_id));
                    continue;
                }
                let node = inner
                    .nodes
                    .get(value)
                    .ok_or_else(|| anyhow!("input value {:?} not registered", value))?;
                match &node.state {
                    NodeState::Ready(_) => {}
                    NodeState::Pending => {
                        return Err(anyhow!("input value {:?} still pending", value));
                    }
                }
                bindings.push((*value, node.spec.clone(), InputRole::Arg, None));
            }
        }

        let (parameter_specs, input_signatures) = {
            let _scope = crate::profiling::compile_scope("graph::prepare_plan.input_signature");
            bindings.sort_by_key(|(value, _, _, _)| value.0);
            let mut parameter_specs: Vec<ParameterSpec> = bindings
                .iter()
                .map(|(value, spec, _, _)| ParameterSpec {
                    value: *value,
                    spec: spec.clone(),
                })
                .collect();
            ensure_parameters_sorted(&mut parameter_specs);

            let input_signatures: Vec<InputSignature> = bindings
                .iter()
                .map(|(_, spec, role, stable_id)| InputSignature {
                    role: *role,
                    stable_id: *stable_id,
                    spec: spec.clone(),
                })
                .collect();
            (parameter_specs, input_signatures)
        };

        let mut ordered: Vec<_> = pending.into_iter().collect();
        ordered.sort_by_key(|value| value.0);

        // Preserve a deterministic result list using creation order, but emit only the
        // requested values (exports + explicit targets). This prevents every SSA value
        // from appearing as a program output.
        let mut result_values: Vec<_> = requested
            .iter()
            .copied()
            .filter(|value| inner.nodes.contains_key(value))
            .collect();
        result_values.sort_by_key(|value| value.0);
        result_values.dedup();

        let mut node_views = Vec::with_capacity(ordered.len());
        {
            let _scope = crate::profiling::compile_scope("graph::prepare_plan.node_views");
            for value in &ordered {
                let node = inner
                    .nodes
                    .get(value)
                    .ok_or_else(|| anyhow!("missing node for value {:?}", value))?;
                node_views.push(PlanNodeView::new(
                    *value,
                    &node.op,
                    node.operands.as_slice(),
                    &node.spec,
                ));
            }
        }

        let mut exports: Vec<_> = inner.exports.iter().copied().collect();
        ensure_exports_sorted(&mut exports);

        let mut signature_targets: Vec<_> = targets.to_vec();
        ensure_targets_sorted(&mut signature_targets);

        // Plan cache keys intentionally ignore arena version so decode-style workloads can reuse
        // the same structural plan across successive graph mutations. Concrete ValueId bindings
        // are rebound per-context in `get_or_build_plan`.
        let (key, compile_key) = {
            let _scope = crate::profiling::compile_scope("graph::prepare_plan.key_hash");
            PlanKey::new_pair_from_views(
                self.backend.backend_name(),
                0,
                &input_signatures,
                &exports,
                &signature_targets,
                node_views.as_slice(),
            )?
        };

        let context = PlanContext {
            key,
            compile_key,
            ordered_values: ordered,
            input_signatures,
            parameter_specs,
            outputs: result_values,
            exports,
            arena_version: inner.version,
        };

        Ok(PrepareResult::NeedsPlan(Box::new(context)))
    }

    fn get_or_build_plan(&self, context: PlanContext) -> Result<(Arc<CachedPlan>, bool)> {
        if let Some(plan) = self.plan_cache.get(&context.key) {
            crate::profiling::cache_event("plan_cache_hit");
            self.record_plan_key(context.key, false);
            return self
                .rebind_plan_for_context(plan, context)
                .map(|plan| (plan, true));
        }
        crate::profiling::cache_event("plan_cache_miss");
        if self.plan_cache.is_enabled() {
            crate::profiling::cache_event("plan_cache_miss_lookup");
            let reason = self.record_plan_key(context.key, true);
            emit_cache_miss_reason(CacheKind::Plan, reason);
        } else {
            crate::profiling::cache_event("plan_cache_miss_disabled");
            emit_cache_miss_reason(CacheKind::Plan, CacheMissReason::BackendOptionChange);
        }

        let plan = {
            let _scope = crate::profiling::compile_scope("graph::build_plan");
            let start = Instant::now();
            let plan = self.build_plan_from_context(context)?;
            crate::ops::graph::timing::add_compile_time(start.elapsed());
            plan
        };
        self.plan_cache.insert(Arc::clone(&plan));
        Ok((plan, false))
    }

    fn rebind_plan_for_context(
        &self,
        plan: Arc<CachedPlan>,
        context: PlanContext,
    ) -> Result<Arc<CachedPlan>> {
        if plan.requested_outputs == context.outputs
            && plan.exports == context.exports
            && plan.parameter_specs.len() == context.parameter_specs.len()
            && plan
                .parameter_specs
                .iter()
                .zip(context.parameter_specs.iter())
                .all(|(cached, current)| {
                    cached.value == current.value && cached.spec == current.spec
                })
        {
            return Ok(plan);
        }

        let (plan_parameter_specs, plan_parameter_values) = {
            let _scope = crate::profiling::compile_scope("graph::bind_plan_inputs");
            build_arg_bindings(&plan.inputs, &context.parameter_specs)?
        };

        Ok(Arc::new(CachedPlan::new(
            context.key,
            Arc::clone(&plan.program),
            plan.program_cache_hit,
            plan.inputs.clone(),
            plan_parameter_specs,
            plan_parameter_values,
            context.outputs,
            plan.program_outputs.clone(),
            context.exports,
        )))
    }

    fn build_plan_from_context(&self, context: PlanContext) -> Result<Arc<CachedPlan>> {
        let PlanContext {
            key,
            compile_key,
            ordered_values,
            input_signatures,
            parameter_specs,
            outputs,
            exports,
            arena_version,
        } = context;

        let program_cache_key = compile_key.with_version(0);

        let mut cached_program_outputs: Option<Vec<ValueId>> = None;
        if let Some(cached) = get_cached_program(&program_cache_key) {
            if program_outputs_cover_requested(&cached.program_outputs, &outputs) {
                crate::profiling::cache_event("program_cache_hit");
                self.record_program_key(program_cache_key, false);
                let (plan_parameter_specs, plan_parameter_values) = {
                    let _scope = crate::profiling::compile_scope("graph::bind_plan_inputs");
                    build_arg_bindings(&cached.inputs, &parameter_specs)?
                };

                return Ok(Arc::new(CachedPlan::new(
                    key,
                    cached.program,
                    true,
                    cached.inputs,
                    plan_parameter_specs,
                    plan_parameter_values,
                    outputs,
                    cached.program_outputs,
                    exports,
                )));
            }
            crate::profiling::cache_event("program_cache_output_coverage_miss");
            cached_program_outputs = Some(cached.program_outputs);
        }
        crate::profiling::cache_event("program_cache_miss");
        crate::profiling::cache_event("program_cache_miss_lookup");
        let reason = self.record_program_key(program_cache_key, true);
        emit_cache_miss_reason(CacheKind::Program, reason);

        let (mut function, entry_params, program_outputs) = {
            let _scope = crate::profiling::compile_scope("graph::lower_to_program");
            let nodes = self.materialize_plan_nodes(ordered_values.as_slice(), arena_version)?;

            let mut builder = ProgramBuilder::new();
            let mut mapping: HashMap<ValueId, ValueId> = HashMap::new();

            for spec in &parameter_specs {
                let new_id = builder.add_parameter(ValueType::Tensor(spec.spec.clone()));
                mapping.insert(spec.value, new_id);
            }

            for node in &nodes {
                let mapped_operands = node
                    .operands
                    .iter()
                    .map(|operand| match operand {
                        Operand::Value(src) => mapping
                            .get(src)
                            .copied()
                            .map(Operand::Value)
                            .ok_or_else(|| anyhow!("missing operand mapping for value {:?}", src)),
                        Operand::TupleElement { tuple, index } => mapping
                            .get(tuple)
                            .copied()
                            .map(|mapped| Operand::TupleElement {
                                tuple: mapped,
                                index: *index,
                            })
                            .ok_or_else(|| anyhow!("missing tuple mapping for value {:?}", tuple)),
                        Operand::Literal(lit) => Ok(Operand::Literal(lit.clone())),
                    })
                    .collect::<Result<Vec<_>>>()?;

                let new_id = builder.emit_single(
                    node.op.clone(),
                    mapped_operands,
                    ValueType::Tensor(node.spec.clone()),
                );
                mapping.insert(node.value, new_id);
            }

            // Compile only values that are semantically required by this plan, plus any
            // previously-cached outputs for the same compile key. This avoids emitting invalid
            // result IDs for transient intermediates while still allowing output-set growth to
            // converge to a stable cached superset.
            let compile_outputs =
                merged_program_outputs(outputs.as_slice(), cached_program_outputs.as_deref());
            let result_ids = compile_outputs
                .iter()
                .map(|original| {
                    mapping
                        .get(original)
                        .copied()
                        .ok_or_else(|| anyhow!("missing mapping for output value {:?}", original))
                })
                .collect::<Result<Vec<_>>>()?;

            let mut entry_params = Vec::with_capacity(parameter_specs.len());
            {
                let inner = self.inner.lock().expect("graph arena poisoned");
                for (idx, spec) in parameter_specs.iter().enumerate() {
                    let role = input_signatures
                        .get(idx)
                        .map(|sig| sig.role)
                        .unwrap_or(InputRole::Arg);
                    let param_id = mapping
                        .get(&spec.value)
                        .copied()
                        .ok_or_else(|| anyhow!("missing parameter mapping for {:?}", spec.value))?;
                    let stable_id = match role {
                        InputRole::Arg => Some(u128::from(param_id.0)),
                        InputRole::Param => {
                            let record = inner.parameter(spec.value).ok_or_else(|| {
                                anyhow!("missing param record for {:?}", spec.value)
                            })?;
                            let stable_id = record.stable_id.ok_or_else(|| {
                                anyhow!("param input {:?} missing stable id", spec.value)
                            })?;
                            if let Some(handle) = record.handle.clone() {
                                self.param_resolver.set(stable_id, handle);
                            }
                            Some(stable_id)
                        }
                    };
                    entry_params.push(crate::backend::optimizer::EntryParam {
                        id: param_id,
                        ty: ValueType::Tensor(spec.spec.clone()),
                        role,
                        stable_id,
                    });
                }
            }

            let function = builder.finish("captured", result_ids);
            Ok::<_, anyhow::Error>((function, entry_params, compile_outputs))?
        };

        let services = OptimizeServices {
            params: Some(self.param_resolver.as_ref()),
        };
        let entry = crate::backend::optimizer::EntrySignature::new(entry_params);
        let cfg = OptimizeConfig::default();
        let mut cx = OptimizeContext::new(self.backend.as_ref(), services, entry, cfg);

        if let Some(optimizer) = self.optimizer.as_ref() {
            let _scope = crate::profiling::compile_scope("optimizer::optimize");
            let _ = optimizer.optimize(&mut function, &mut cx);
        }

        let inputs = cx.entry().plan_inputs();
        let (plan_parameter_specs, plan_parameter_values) = {
            let _scope = crate::profiling::compile_scope("graph::bind_plan_inputs");
            build_arg_bindings(&inputs, &parameter_specs)?
        };

        let program = Arc::new(Program::new("captured").with_functions(vec![function]));
        insert_cached_program(
            program_cache_key,
            Arc::clone(&program),
            inputs.clone(),
            program_outputs.clone(),
        );
        Ok(Arc::new(CachedPlan::new(
            key,
            program,
            false,
            inputs,
            plan_parameter_specs,
            plan_parameter_values,
            outputs,
            program_outputs,
            exports,
        )))
    }

    fn record_plan_key(&self, key: PlanKey, miss: bool) -> CacheMissReason {
        let mut telemetry = self
            .miss_telemetry
            .lock()
            .expect("graph arena miss telemetry poisoned");
        let previous_any = telemetry.last_plan_any;
        let previous = telemetry.last_plan_by_graph.insert(key.graph_hash, key);
        telemetry.last_plan_any = Some(key);
        if miss {
            key.classify_change_from(previous.or(previous_any))
        } else {
            CacheMissReason::Unknown
        }
    }

    fn record_program_key(&self, key: PlanKey, miss: bool) -> CacheMissReason {
        let mut telemetry = self
            .miss_telemetry
            .lock()
            .expect("graph arena miss telemetry poisoned");
        let previous_any = telemetry.last_program_any;
        let previous = telemetry.last_program_by_graph.insert(key.graph_hash, key);
        telemetry.last_program_any = Some(key);
        if miss {
            key.classify_change_from(previous.or(previous_any))
        } else {
            CacheMissReason::Unknown
        }
    }

    fn materialize_plan_nodes(
        &self,
        ordered_values: &[ValueId],
        arena_version: u64,
    ) -> Result<Vec<PlanNode>> {
        let inner = self.inner.lock().expect("graph arena poisoned");
        if inner.version != arena_version {
            return Err(StalePlanError.into());
        }

        let mut nodes = Vec::with_capacity(ordered_values.len());
        for value in ordered_values {
            let node = inner
                .nodes
                .get(value)
                .ok_or_else(|| anyhow!("missing node for value {:?}", value))?;
            nodes.push(PlanNode {
                value: *value,
                op: node.op.clone(),
                operands: node.operands.clone(),
                spec: node.spec.clone(),
            });
        }
        Ok(nodes)
    }

    fn try_execute_plan(
        &self,
        plan: &Arc<CachedPlan>,
        targets: &[ValueId],
        entry_inputs: Vec<B::TensorHandle>,
        plan_cache_hit: bool,
    ) -> Result<Vec<B::TensorHandle>> {
        let trace_sink = trace::current_sink();
        let trace_id = trace::next_trace_id();
        let context = ProgramContext {
            trace_id,
            graph_id: self.id,
            backend: self.backend.backend_name().to_string(),
            plan_hash: plan.key.hash,
            plan_graph_hash: plan.key.graph_hash,
            plan_specialization_hash: plan.key.specialization_hash,
            cache: ProgramCacheInfo {
                plan_cache_hit,
                program_cache_hit: plan.program_cache_hit,
            },
            targets: targets.to_vec(),
            outputs: plan.requested_outputs.clone(),
            exports: plan.exports.clone(),
            timestamp: std::time::SystemTime::now(),
            kind: ProgramKind::Materialize {
                values: targets.to_vec(),
            },
        };

        if let Some(ref sink) = trace_sink {
            sink.before_program(&context, &plan.program);
        }

        let start = Instant::now();
        let exec = self.backend.run_program(&plan.program, &entry_inputs);

        match exec {
            Ok(mut produced) => {
                if produced.len() != plan.program_outputs.len() {
                    if let Some(ref sink) = trace_sink {
                        sink.after_program(
                            &context,
                            &ProgramStats {
                                duration: start.elapsed(),
                                output_count: produced.len(),
                                status: ProgramStatus::Failure {
                                    message: format!(
                                        "backend returned {} outputs, expected {}",
                                        produced.len(),
                                        plan.program_outputs.len()
                                    ),
                                },
                            },
                        );
                    }
                    return Err(anyhow!(
                        "backend returned {} outputs, expected {}",
                        produced.len(),
                        plan.program_outputs.len()
                    ));
                }

                let final_handles = {
                    let mut inner = self.inner.lock().expect("graph arena poisoned");
                    if plan.key.version != 0 && inner.version != plan.key.version {
                        return Err(StalePlanError.into());
                    }

                    for (value_id, handle) in plan.program_outputs.iter().zip(produced.drain(..)) {
                        if let Some(node) = inner.nodes.get_mut(value_id) {
                            node.state = NodeState::Ready(handle);
                        }
                    }

                    self.collect_target_handles(&inner, targets)?
                };

                if let Some(ref sink) = trace_sink {
                    sink.after_program(
                        &context,
                        &ProgramStats {
                            duration: start.elapsed(),
                            output_count: plan.program_outputs.len(),
                            status: ProgramStatus::Success,
                        },
                    );
                }

                Ok(final_handles)
            }
            Err(err) => {
                if let Some(ref sink) = trace_sink {
                    sink.after_program(
                        &context,
                        &ProgramStats {
                            duration: start.elapsed(),
                            output_count: 0,
                            status: ProgramStatus::Failure {
                                message: err.to_string(),
                            },
                        },
                    );
                }
                Err(err.into())
            }
        }
    }

    fn collect_entry_inputs(
        &self,
        plan: &Arc<CachedPlan>,
        overrides: Option<&[B::TensorHandle]>,
    ) -> Result<Vec<B::TensorHandle>> {
        let mut inner = self.inner.lock().expect("graph arena poisoned");
        if plan.key.version != 0 && inner.version != plan.key.version {
            return Err(StalePlanError.into());
        }

        if let Some(overrides) = overrides {
            if overrides.len() != plan.parameter_values.len() {
                return Err(anyhow!(
                    "override handle count {} does not match parameter count {}",
                    overrides.len(),
                    plan.parameter_values.len()
                ));
            }
        }

        if plan.inputs.roles.len() != plan.inputs.stable_ids.len() {
            bail!(
                "plan inputs role/stable id length mismatch: {} vs {}",
                plan.inputs.roles.len(),
                plan.inputs.stable_ids.len()
            );
        }

        let mut arg_index = 0usize;
        let mut handles = Vec::with_capacity(plan.inputs.roles.len());
        for (role, stable_id) in plan.inputs.roles.iter().zip(plan.inputs.stable_ids.iter()) {
            match role {
                InputRole::Arg => {
                    let value = plan
                        .parameter_values
                        .get(arg_index)
                        .copied()
                        .ok_or_else(|| anyhow!("arg index {} out of range", arg_index))?;
                    if let Some(overrides) = overrides {
                        let handle = overrides[arg_index].clone();
                        if let Some(param) = inner.parameter_mut(value) {
                            param.handle = Some(handle.clone());
                        } else {
                            let node = inner
                                .nodes
                                .get_mut(&value)
                                .ok_or_else(|| anyhow!("arg value {:?} not registered", value))?;
                            node.state = NodeState::Ready(handle.clone());
                        }
                        handles.push(handle);
                        arg_index += 1;
                        continue;
                    }

                    if let Some(param) = inner.parameter(value) {
                        handles.push(self.resolve_parameter_handle(&inner, param)?);
                        arg_index += 1;
                        continue;
                    }

                    let node = inner
                        .nodes
                        .get(&value)
                        .ok_or_else(|| anyhow!("arg value {:?} not registered", value))?;
                    match &node.state {
                        NodeState::Ready(handle) => handles.push(handle.clone()),
                        NodeState::Pending => {
                            return Err(anyhow!(
                                "arg value {:?} pending while collecting entry inputs",
                                value
                            ));
                        }
                    }
                    arg_index += 1;
                }
                InputRole::Param => {
                    let stable_id =
                        stable_id.ok_or_else(|| anyhow!("param input missing stable id"))?;
                    let handle = self.resolve_param_handle_by_stable_id(&inner, stable_id)?;
                    handles.push(handle);
                }
            }
        }

        if arg_index != plan.parameter_values.len() {
            bail!(
                "plan arg count mismatch: consumed {} values but plan recorded {}",
                arg_index,
                plan.parameter_values.len()
            );
        }

        drop(inner);
        Ok(handles)
    }
}

#[derive(Debug)]
struct StalePlanError;

impl fmt::Display for StalePlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "cached plan is stale after graph mutation")
    }
}

impl std::error::Error for StalePlanError {}

enum CacheKind {
    Plan,
    Program,
}

const PLAN_CACHE_MISS_REASON_EVENTS: [&str; 8] = [
    "plan_cache_miss_reason.shape_specialization_change",
    "plan_cache_miss_reason.dtype_change",
    "plan_cache_miss_reason.layout_change",
    "plan_cache_miss_reason.literal_value_only_change",
    "plan_cache_miss_reason.kv_bucket_change",
    "plan_cache_miss_reason.backend_option_change",
    "plan_cache_miss_reason.graph_structure_change",
    "plan_cache_miss_reason.unknown",
];

const PROGRAM_CACHE_MISS_REASON_EVENTS: [&str; 8] = [
    "program_cache_miss_reason.shape_specialization_change",
    "program_cache_miss_reason.dtype_change",
    "program_cache_miss_reason.layout_change",
    "program_cache_miss_reason.literal_value_only_change",
    "program_cache_miss_reason.kv_bucket_change",
    "program_cache_miss_reason.backend_option_change",
    "program_cache_miss_reason.graph_structure_change",
    "program_cache_miss_reason.unknown",
];

fn cache_miss_reason_index(reason: CacheMissReason) -> usize {
    match reason {
        CacheMissReason::ShapeSpecializationChange => 0,
        CacheMissReason::DTypeChange => 1,
        CacheMissReason::LayoutChange => 2,
        CacheMissReason::LiteralValueOnlyChange => 3,
        CacheMissReason::KvBucketChange => 4,
        CacheMissReason::BackendOptionChange => 5,
        CacheMissReason::GraphStructureChange => 6,
        CacheMissReason::Unknown => 7,
    }
}

fn emit_cache_miss_reason(kind: CacheKind, reason: CacheMissReason) {
    let index = cache_miss_reason_index(reason);
    let event = match kind {
        CacheKind::Plan => PLAN_CACHE_MISS_REASON_EVENTS[index],
        CacheKind::Program => PROGRAM_CACHE_MISS_REASON_EVENTS[index],
    };
    crate::profiling::cache_event(event);
}

fn program_outputs_cover_requested(
    program_outputs: &[ValueId],
    requested_outputs: &[ValueId],
) -> bool {
    requested_outputs
        .iter()
        .all(|requested| program_outputs.contains(requested))
}

fn merged_program_outputs(
    requested_outputs: &[ValueId],
    cached_program_outputs: Option<&[ValueId]>,
) -> Vec<ValueId> {
    let mut merged = Vec::with_capacity(
        requested_outputs.len()
            + cached_program_outputs
                .map(|outputs| outputs.len())
                .unwrap_or_default(),
    );
    merged.extend_from_slice(requested_outputs);
    if let Some(outputs) = cached_program_outputs {
        merged.extend_from_slice(outputs);
    }
    merged.sort_by_key(|value| value.0);
    merged.dedup();
    merged
}

enum PrepareResult<B: PortableBackend + 'static> {
    AllReady(Vec<B::TensorHandle>),
    NeedsPlan(Box<PlanContext>),
}

fn build_arg_bindings(
    inputs: &crate::backend::optimizer::PlanInputs,
    parameter_specs: &[ParameterSpec],
) -> Result<(Vec<ParameterSpec>, Vec<ValueId>)> {
    if inputs.roles.len() != inputs.stable_ids.len() {
        bail!(
            "plan inputs role/stable id length mismatch: {} vs {}",
            inputs.roles.len(),
            inputs.stable_ids.len()
        );
    }

    let mut arg_specs = Vec::new();
    let mut arg_values = Vec::new();
    for (role, stable_id) in inputs.roles.iter().zip(inputs.stable_ids.iter()) {
        if *role != InputRole::Arg {
            continue;
        }
        let stable_id = stable_id.ok_or_else(|| anyhow!("arg input missing stable id"))?;
        let idx = usize::try_from(stable_id)
            .map_err(|_| anyhow!("arg stable id {} out of range", stable_id))?;
        let spec = parameter_specs
            .get(idx)
            .ok_or_else(|| anyhow!("arg stable id {} out of range", stable_id))?;
        arg_values.push(spec.value);
        arg_specs.push(ParameterSpec {
            value: spec.value,
            spec: spec.spec.clone(),
        });
    }

    Ok((arg_specs, arg_values))
}

struct PlanContext {
    key: PlanKey,
    compile_key: PlanKey,
    ordered_values: Vec<ValueId>,
    input_signatures: Vec<InputSignature>,
    parameter_specs: Vec<ParameterSpec>,
    outputs: Vec<ValueId>,
    exports: Vec<ValueId>,
    arena_version: u64,
}

pub struct CompiledGraph<B: PortableBackend + 'static> {
    arena: Arc<GraphArena<B>>,
    plan: Arc<CachedPlan>,
    targets: Vec<ValueId>,
    arena_version: u64,
}

impl<B: PortableBackend + 'static> CompiledGraph<B> {
    fn new(
        arena: Arc<GraphArena<B>>,
        plan: Arc<CachedPlan>,
        targets: Vec<ValueId>,
        arena_version: u64,
    ) -> Self {
        CompiledGraph {
            arena,
            plan,
            targets,
            arena_version,
        }
    }

    pub fn execute(&self) -> Result<Vec<B::TensorHandle>> {
        {
            let inner = self.arena.inner.lock().expect("graph arena poisoned");
            if inner.version != self.arena_version {
                bail!("compiled graph is stale after arena mutation");
            }

            let mut ready = Vec::with_capacity(self.targets.len());
            let mut pending = false;
            for value in &self.targets {
                if let Some(node) = inner.nodes.get(value) {
                    match &node.state {
                        NodeState::Ready(handle) => ready.push(handle.clone()),
                        NodeState::Pending => {
                            pending = true;
                            break;
                        }
                    }
                } else if let Some(param) = inner.parameter(*value) {
                    ready.push(self.arena.resolve_parameter_handle(&inner, param)?);
                } else {
                    bail!("value {:?} not registered in graph", value);
                }
            }

            if !pending {
                return Ok(ready);
            }
        }

        let entry_inputs = self.arena.collect_entry_inputs(&self.plan, None)?;
        match self
            .arena
            .try_execute_plan(&self.plan, &self.targets, entry_inputs, true)
        {
            Ok(handles) => Ok(handles),
            Err(err) => {
                if err.downcast_ref::<StalePlanError>().is_some() {
                    bail!("compiled graph is stale after arena mutation");
                }
                Err(err)
            }
        }
    }

    pub fn execute_with_inputs(&self, inputs: &[&DeviceTensor<B>]) -> Result<Vec<B::TensorHandle>> {
        {
            let inner = self.arena.inner.lock().expect("graph arena poisoned");
            if inner.version != self.arena_version {
                bail!("compiled graph is stale after arena mutation");
            }
        }

        if inputs.len() != self.plan.parameter_values.len() {
            bail!(
                "compiled graph expects {} inputs, got {}",
                self.plan.parameter_values.len(),
                inputs.len()
            );
        }

        let arena_backend = self.arena.backend();
        let mut overrides = Vec::with_capacity(inputs.len());

        for (idx, tensor) in inputs.iter().enumerate() {
            if !Arc::ptr_eq(&arena_backend, &tensor.backend()) {
                bail!("input {} is bound to a different backend", idx);
            }

            let expected = &self.plan.parameter_specs[idx].spec;
            if &tensor.tensor_spec() != expected {
                bail!("input {} does not match expected tensor spec", idx);
            }

            overrides.push(tensor.materialize()?);
        }

        let entry_inputs = self
            .arena
            .collect_entry_inputs(&self.plan, Some(&overrides))?;
        match self
            .arena
            .try_execute_plan(&self.plan, &self.targets, entry_inputs, true)
        {
            Ok(handles) => Ok(handles),
            Err(err) => {
                if err.downcast_ref::<StalePlanError>().is_some() {
                    bail!("compiled graph is stale after arena mutation");
                }
                Err(err)
            }
        }
    }

    pub fn targets(&self) -> &[ValueId] {
        &self.targets
    }

    pub fn parameters(&self) -> &[ValueId] {
        &self.plan.parameter_values
    }
}

/// Recursively collects pending dependencies for `value`, classifying which inputs can be fed as
/// parameters and which nodes must be executed.
fn collect_dependencies<B: PortableBackend + 'static>(
    inner: &GraphInner<B>,
    value: ValueId,
    pending: &mut HashSet<ValueId>,
    inputs: &mut HashSet<ValueId>,
) -> Result<()> {
    if pending.contains(&value) || inputs.contains(&value) {
        return Ok(());
    }

    let node = inner
        .nodes
        .get(&value)
        .ok_or_else(|| anyhow!("value {:?} not registered", value))?;

    if let NodeState::Ready(_) = node.state {
        inputs.insert(value);
        return Ok(());
    }

    pending.insert(value);

    for operand in &node.operands {
        if let Operand::Value(dep) = operand {
            if inner.nodes.contains_key(dep) {
                collect_dependencies(inner, *dep, pending, inputs)?;
            } else {
                inputs.insert(*dep);
            }
        }
    }

    Ok(())
}
