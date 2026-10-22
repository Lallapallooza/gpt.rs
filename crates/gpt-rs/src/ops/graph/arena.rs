//! Shared arena that stores lazily constructed operation graphs.
//!
//! The graph arena is the central orchestrator for lazy tensor graph execution. It maintains
//! a mutable graph of pending operations and compiles them into backend programs on demand. The
//! program cache reuses a compiled program for every graph with the same plan key.
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
//!      +-- GraphInner (values, ops, exports)
//!      |
//!      +-- Backend (program execution)
//! ```
//!
//! ## Lazy Execution Model
//!
//! 1. **Record**: Each op of `DeviceTensor` records a graph node through `arena.capture()`
//! 2. **Export**: Nodes are marked as exported when they flow into DeviceTensor results
//! 3. **Materialize**: When a tensor's data is accessed, the arena builds a plan of the pending
//!    nodes it needs
//! 4. **Cache**: The program cache holds compiled programs by plan key
//! 5. **Execute**: Backend runs the program and stores results
//!
//! ## Program Reuse
//!
//! Plan keys leave out value ids and the arena version, so a graph with the same structure in a
//! later arena reuses the program. A cached program is recompiled when it does not produce every
//! requested output. A build fails with a stale-plan error and retries when the graph changed
//! while it ran.

use std::any::Any;
use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{anyhow, Result};
use gpt_rs_ir::dialects::func::ParamId;
use gpt_rs_ir::ir::{Module, Type, Value};
use gpt_rs_ir::passes::{run_pass, PruneParams};

use super::builder::GraphBuilder;
use super::plan::{
    get_cached_program, insert_cached_program, CachedProgram, InputBinding, Plan, PlanGraph,
    PlanInput, PlanKey, PlanNode, PlanNodeView, ValueRef,
};
use super::state::{GraphInner, NodeState, ValueDef, ValueSet};
use crate::backend::PortableBackend;
use crate::ops::graph::ValueId;
use crate::ops::trace::{
    self, ExecutionTraceSink, ProgramContext, ProgramKind, ProgramStats, ProgramStatus,
};
use crate::tensor::{InputRole, Materialized};

/// Central storage for lazy tensor graphs built on top of a single backend instance.
pub struct GraphArena<B: PortableBackend + 'static> {
    backend: Arc<B>,
    inner: Mutex<GraphInner<B>>,
    id: usize,
    /// The programs it ran, when it records them for a trace.
    runs: Option<Mutex<Vec<ProgramRun>>>,
}

/// A program run of an arena that records its runs: what a traced function replays.
pub(super) struct ProgramRun {
    pub(super) key: PlanKey,
    pub(super) exe: Arc<dyn Any + Send + Sync>,
    pub(super) module: Arc<Module>,
    /// What each argument of the program was, in order.
    pub(super) inputs: Vec<RunInput>,
    /// The value of each result of the program, in order.
    pub(super) outputs: Vec<ValueId>,
    /// The values marked for export that the program also computed.
    pub(super) exports: Vec<ValueId>,
}

/// An argument of a program that an arena ran.
pub(super) enum RunInput {
    /// The parameter of this id.
    Param(u128),
    /// The tensor of this id that the graph captured, which the record does not hold, so that the
    /// run may take it.
    Captured(u128),
    /// A value that an earlier program of the arena computed.
    Computed,
}

static ARENA_ID_COUNTER: AtomicUsize = AtomicUsize::new(0);

impl<B: PortableBackend + 'static> GraphArena<B> {
    /// Creates a new arena wrapping the provided backend.
    pub fn new(backend: Arc<B>) -> Arc<Self> {
        Self::with_runs(backend, None)
    }

    /// A new arena that records the programs it runs, which [`Self::take_runs`] gives.
    pub(super) fn recording_runs(backend: Arc<B>) -> Arc<Self> {
        Self::with_runs(backend, Some(Mutex::default()))
    }

    fn with_runs(backend: Arc<B>, runs: Option<Mutex<Vec<ProgramRun>>>) -> Arc<Self> {
        let id = ARENA_ID_COUNTER.fetch_add(1, AtomicOrdering::Relaxed);
        Arc::new(GraphArena {
            backend,
            inner: Mutex::new(GraphInner::new()),
            id,
            runs,
        })
    }

    /// The programs that this arena ran since it was made, or since the last call, when it
    /// records them.
    pub(super) fn take_runs(&self) -> Vec<ProgramRun> {
        match &self.runs {
            Some(runs) => std::mem::take(&mut *runs.lock().expect("graph arena poisoned")),
            None => Vec::new(),
        }
    }

    /// The id of the arena, which trace sinks see as the graph of its programs.
    pub(super) fn id(&self) -> usize {
        self.id
    }

    /// Returns the underlying backend handle.
    pub fn backend(&self) -> Arc<B> {
        Arc::clone(&self.backend)
    }

    /// Whether this arena runs on the backend instance `backend`.
    pub fn runs_on(&self, backend: &Arc<B>) -> bool {
        Arc::ptr_eq(&self.backend, backend)
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

    pub(crate) fn try_ready_handle(&self, value: ValueId) -> Option<B::TensorHandle> {
        let inner = self.inner.lock().expect("graph arena poisoned");
        self.ready_handle(&inner, value).ok().flatten()
    }

    /// The handle of `value` when it is captured or computed, `None` while it is pending.
    fn ready_handle(
        &self,
        inner: &GraphInner<B>,
        value: ValueId,
    ) -> Result<Option<B::TensorHandle>> {
        match &inner.value(value)?.def {
            ValueDef::Node {
                state: NodeState::Ready(handle),
                ..
            } => Ok(Some(handle.clone())),
            ValueDef::Node {
                state: NodeState::Pending,
                ..
            } => Ok(None),
            ValueDef::Input(input) => self.input_handle(input).map(Some),
            ValueDef::Given => Err(anyhow!(
                "{value:?} went to a program that may have updated it in place"
            )),
        }
    }

    /// The handle of a captured tensor. A parameter's handle is the one in the parameter store,
    /// which [`DeviceTensor::as_param`](crate::tensor::DeviceTensor::as_param) filled.
    fn input_handle(&self, input: &Materialized<B>) -> Result<B::TensorHandle> {
        match input.role {
            InputRole::Param => self.param_handle(input.id),
            InputRole::Arg => Ok(input.tensor.clone()),
        }
    }

    fn param_handle(&self, id: u128) -> Result<B::TensorHandle> {
        self.backend.params().get(id).ok_or_else(|| {
            anyhow!(
                "parameter {id:#x} is not in the parameter store of backend {}",
                self.backend.backend_name()
            )
        })
    }

    fn collect_target_handles(
        &self,
        inner: &GraphInner<B>,
        targets: &[ValueId],
    ) -> Result<Vec<B::TensorHandle>> {
        targets
            .iter()
            .map(|&value| {
                self.ready_handle(inner, value)?
                    .ok_or_else(|| anyhow!("value {value:?} pending after program execution"))
            })
            .collect()
    }

    /// Records graph edits through a [`GraphBuilder`], holding the arena for their duration.
    /// The edits remain isolated to this arena and can be replayed later when materialization is requested.
    pub(crate) fn capture<R, F>(self: &Arc<Self>, edit: F) -> Result<R>
    where
        F: FnOnce(&mut GraphBuilder<B>) -> Result<R>,
    {
        let mut inner = self.inner.lock().expect("graph arena poisoned");
        edit(&mut GraphBuilder {
            arena: self,
            inner: &mut inner,
            location: None,
        })
    }

    /// Materializes the requested value, executing pending nodes as necessary.
    /// Pending nodes are compiled into a single backend program so repeated materializations stay efficient.
    pub fn flush_until(&self, value: ValueId) -> Result<<B as PortableBackend>::TensorHandle> {
        let mut handles = self.materialize_values(&[value])?;
        handles
            .pop()
            .ok_or_else(|| anyhow!("value missing after execution"))
    }

    /// Materializes all requested value identifiers, executing pending nodes at most once.
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
                    let plan = {
                        let _scope = crate::profiling::compile_scope("graph::plan");
                        self.plan(*context)?
                    };
                    let run_inputs = match &self.runs {
                        Some(_) => Some(self.run_inputs(&plan)?),
                        None => None,
                    };
                    let entry_inputs = self.collect_entry_inputs(&plan)?;
                    match self.try_execute_plan(&plan, values, entry_inputs) {
                        Ok(handles) => {
                            if let (Some(runs), Some(inputs)) = (&self.runs, run_inputs) {
                                runs.lock().expect("graph arena poisoned").push(ProgramRun {
                                    key: plan.key,
                                    exe: Arc::clone(&plan.exe),
                                    module: Arc::clone(&plan.module),
                                    inputs,
                                    outputs: plan.program_outputs.clone(),
                                    exports: plan.exports.clone(),
                                });
                            }
                            return Ok(handles);
                        }
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

    /// Walks the dependency graph, determines which nodes need execution, and packages the
    /// result into a [`PlanContext`]. The method:
    /// - short-circuits when every target is already materialized;
    /// - finds the pending nodes that the targets and exports need ([`plan_refs`]);
    /// - binds ready nodes and captured tensors as the plan's inputs;
    /// - orders the pending nodes by value id, so dependencies run before their consumers.
    fn prepare_plan(&self, targets: &[ValueId]) -> Result<PrepareResult<B>> {
        let _scope = crate::profiling::compile_scope("graph::prepare_plan");
        let inner = self.inner.lock().expect("graph arena poisoned");

        let mut ready_handles = Vec::with_capacity(targets.len());
        for &value in targets {
            match self.ready_handle(&inner, value)? {
                Some(handle) => ready_handles.push(handle),
                None => break,
            }
        }
        if ready_handles.len() == targets.len() {
            return Ok(PrepareResult::AllReady(ready_handles));
        }

        // The program also computes every export, such as updated caches, so later programs read
        // them instead of recomputing.
        let mut requested = Vec::with_capacity(inner.exports.len() + targets.len());
        requested.extend(
            inner
                .exports
                .iter()
                .chain(targets)
                .copied()
                .filter(|&value| inner.node(value).is_some()),
        );
        requested.sort_by_key(|value| value.0);
        requested.dedup();

        let (refs, node_count, input_count) = plan_refs(&inner, &requested)?;
        let (ordered, inputs) = plan_values(&inner, &refs, node_count, input_count)?;

        let mut exports: Vec<_> = inner.exports.iter().copied().collect();
        exports.sort_by_key(|value| value.0);

        let graph = PlanGraph {
            inputs: &inputs,
            nodes: &ordered,
            refs: &refs,
            node_view: |value: ValueId| {
                let (op, result) = inner.node(value)?;
                Some(PlanNodeView {
                    op: &op.op,
                    operands: inner.operands(op),
                    first: op.first,
                    result,
                })
            },
        };
        let key = PlanKey::new(
            self.backend.backend_name(),
            self.backend.compile_identity(),
            &graph,
        )?;

        Ok(PrepareResult::NeedsPlan(Box::new(PlanContext {
            key,
            ordered_values: ordered,
            inputs,
            outputs: requested,
            exports,
            arena_version: inner.version,
        })))
    }

    /// The program of `context`: the cached one when it computes every requested output, else a
    /// new one, compiled with the cached program's outputs added.
    fn plan(&self, context: PlanContext) -> Result<Plan> {
        let PlanContext {
            key,
            ordered_values,
            inputs: plan_inputs,
            outputs,
            exports,
            arena_version,
        } = context;

        let mut compile_outputs = outputs.clone();
        if let Some(cached) = get_cached_program(&key) {
            let program_outputs =
                values_at_ranks(&ordered_values, &plan_inputs, &cached.output_ranks)?;
            if outputs.iter().all(|value| program_outputs.contains(value)) {
                crate::profiling::cache_event("program_cache_hit");
                return Ok(Plan {
                    key,
                    args: arg_values(&cached.inputs, &plan_inputs)?,
                    exe: cached.exe,
                    module: cached.module,
                    program_cache_hit: true,
                    inputs: cached.inputs,
                    requested_outputs: outputs,
                    program_outputs,
                    exports,
                });
            }
            // Compile a superset, so the outputs that the cached program serves stay served.
            compile_outputs.extend(program_outputs);
            compile_outputs.sort_by_key(|value| value.0);
            compile_outputs.dedup();
        }
        crate::profiling::cache_event("program_cache_miss");

        let _scope = crate::profiling::compile_scope("graph::compile");
        let nodes = self.materialize_plan_nodes(&ordered_values, arena_version)?;
        let params: Vec<Option<ParamId>> = plan_inputs
            .iter()
            .map(|input| input.stable_id.map(ParamId))
            .collect();
        let mut built = build_module(&plan_inputs, &params, &nodes, &compile_outputs)?;
        {
            let _pipeline_scope = crate::profiling::compile_scope("optimizer::pass_pipeline");
            trace::run_pipeline(&mut built, &self.backend.pipeline())?;
        }
        let inputs = prune_params(&mut built)?;
        let module = Arc::new(built);
        self.load_static_params(&module)?;
        let compiled = {
            let _backend_scope = crate::profiling::compile_scope("backend::compile");
            self.backend.compile(&module, "captured")?
        };
        let exe: Arc<dyn Any + Send + Sync> = Arc::new(compiled);
        let output_ranks = ranks_of(&ordered_values, &plan_inputs, &compile_outputs)?;
        insert_cached_program(
            key,
            CachedProgram {
                exe: Arc::clone(&exe),
                module: Arc::clone(&module),
                inputs: inputs.clone(),
                output_ranks,
            },
        );
        Ok(Plan {
            key,
            args: arg_values(&inputs, &plan_inputs)?,
            exe,
            module,
            program_cache_hit: false,
            inputs,
            requested_outputs: outputs,
            program_outputs: compile_outputs,
            exports,
        })
    }

    /// Runs on the backend each load-time function of `module` whose results the parameter store
    /// does not hold yet, and stores the results there.
    fn load_static_params(&self, module: &Arc<Module>) -> Result<()> {
        for func in module.funcs() {
            let attrs = func.attrs();
            if !attrs.is_load_time() {
                continue;
            }
            let computed: Vec<u128> = attrs.result_params.iter().map(|p| p.0).collect();
            let params = self.backend.params();
            if params.get_all(computed.iter().copied()).is_some() {
                continue;
            }
            let leaves = (0..attrs.args.len())
                .map(|position| {
                    let id = attrs
                        .param(position)
                        .expect("a load-time function reads parameters");
                    self.param_handle(id.0)
                })
                .collect::<Result<Vec<_>>>()?;
            let _suspend = crate::profiling::suspend();
            let handles = self
                .backend
                .compile(module, &attrs.name)
                .and_then(|exe| self.backend.run(&exe, leaves))
                .map_err(|err| anyhow!("load-time function @{}: {err:#}", attrs.name))?;
            for (id, handle) in computed.into_iter().zip(handles) {
                params.set(id, handle);
            }
        }
        Ok(())
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

        ordered_values
            .iter()
            .map(|&value| {
                let (op, result) = inner
                    .node(value)
                    .ok_or_else(|| anyhow!("missing node for value {value:?}"))?;
                Ok(PlanNode {
                    value,
                    op: op.op.clone(),
                    operands: inner.operands(op).to_vec(),
                    first: op.first,
                    result,
                    location: op.location.clone(),
                })
            })
            .collect()
    }

    fn try_execute_plan(
        &self,
        plan: &Plan,
        targets: &[ValueId],
        entry_inputs: Vec<B::TensorHandle>,
    ) -> Result<Vec<B::TensorHandle>> {
        // Build the program context only when a trace sink will read it.
        let traced = trace::current_sink().map(|sink| (sink, self.program_context(plan, targets)));

        if let Some((sink, context)) = &traced {
            sink.before_program(context, &plan.module);
        }

        let start = Instant::now();
        let exe = plan
            .exe
            .downcast_ref::<B::Executable>()
            .ok_or_else(|| anyhow!("the plan was compiled by another backend type"))?;
        let exec = self.backend.run(exe, entry_inputs);

        match exec {
            Ok(produced) => {
                if produced.len() != plan.program_outputs.len() {
                    let message = format!(
                        "backend returned {} outputs, expected {}",
                        produced.len(),
                        plan.program_outputs.len()
                    );
                    let status = ProgramStatus::Failure {
                        message: message.clone(),
                    };
                    report_program(traced.as_ref(), start, produced.len(), status);
                    return Err(anyhow!(message));
                }

                let final_handles = self.store_outputs(plan, targets, produced)?;
                let output_count = plan.program_outputs.len();
                report_program(traced.as_ref(), start, output_count, ProgramStatus::Success);
                Ok(final_handles)
            }
            Err(err) => {
                let status = ProgramStatus::Failure {
                    message: err.to_string(),
                };
                report_program(traced.as_ref(), start, 0, status);
                Err(err)
            }
        }
    }

    /// What a trace sink learns about the run of `plan` that materializes `targets`.
    fn program_context(&self, plan: &Plan, targets: &[ValueId]) -> ProgramContext {
        ProgramContext {
            trace_id: trace::next_trace_id(),
            graph_id: self.id,
            backend: self.backend.backend_name().to_string(),
            plan_hash: plan.key.0,
            program_cache_hit: plan.program_cache_hit,
            targets: targets.to_vec(),
            outputs: plan.requested_outputs.clone(),
            exports: plan.exports.clone(),
            timestamp: std::time::SystemTime::now(),
            kind: ProgramKind::Materialize {
                values: targets.to_vec(),
            },
        }
    }

    /// Stores each handle that the run of `plan` `produced` as the value of its output, and returns
    /// the handles of `targets`.
    fn store_outputs(
        &self,
        plan: &Plan,
        targets: &[ValueId],
        produced: Vec<B::TensorHandle>,
    ) -> Result<Vec<B::TensorHandle>> {
        let mut inner = self.inner.lock().expect("graph arena poisoned");
        for (value, handle) in plan.program_outputs.iter().zip(produced) {
            if let Some(ValueDef::Node { state, .. }) = inner
                .values
                .get_mut(value.0 as usize)
                .map(|record| &mut record.def)
            {
                *state = NodeState::Ready(handle);
            }
        }
        self.collect_target_handles(&inner, targets)
    }

    /// The handles of the inputs of `plan`, in the order of its program's parameters. The
    /// parameters come from the parameter store under one lock.
    fn collect_entry_inputs(&self, plan: &Plan) -> Result<Vec<B::TensorHandle>> {
        let param_ids = plan.inputs.iter().filter_map(|binding| match binding {
            InputBinding::Param(id) => Some(*id),
            InputBinding::Arg(_) => None,
        });
        let mut params = self
            .backend
            .params()
            .get_all(param_ids)
            .ok_or_else(|| anyhow!("a parameter of the program is not in the parameter store"))?
            .into_iter();
        let mut inner = self.inner.lock().expect("graph arena poisoned");
        let given = given_inputs(&inner, plan);
        let mut args = plan.args.iter();
        let mut handles = Vec::with_capacity(plan.inputs.len());
        for binding in &plan.inputs {
            let handle = match binding {
                InputBinding::Param(_) => params.next().expect("one handle per parameter"),
                InputBinding::Arg(_) => {
                    let value = *args
                        .next()
                        .ok_or_else(|| anyhow!("plan binds more arguments than it records"))?;
                    match given.contains(&value) {
                        true => give(&mut inner, value),
                        false => self.arg_handle(&inner, value)?,
                    }
                }
            };
            handles.push(handle);
        }
        Ok(handles)
    }

    /// What each argument of `plan` is, in the order of its program's parameters.
    fn run_inputs(&self, plan: &Plan) -> Result<Vec<RunInput>> {
        let inner = self.inner.lock().expect("graph arena poisoned");
        let mut args = plan.args.iter();
        let mut inputs = Vec::with_capacity(plan.inputs.len());
        for binding in &plan.inputs {
            let input = match binding {
                InputBinding::Param(id) => RunInput::Param(*id),
                InputBinding::Arg(_) => {
                    let value = *args
                        .next()
                        .ok_or_else(|| anyhow!("plan binds more arguments than it records"))?;
                    match &inner.value(value)?.def {
                        ValueDef::Input(input) if input.role == InputRole::Arg => {
                            RunInput::Captured(input.id)
                        }
                        _ => RunInput::Computed,
                    }
                }
            };
            inputs.push(input);
        }
        Ok(inputs)
    }

    fn arg_handle(&self, inner: &GraphInner<B>, value: ValueId) -> Result<B::TensorHandle> {
        self.ready_handle(inner, value)?
            .ok_or_else(|| anyhow!("arg value {value:?} pending while collecting entry inputs"))
    }
}

/// The run-time arguments of `plan` whose handles its run may take, so the backend may update
/// their memory in place. Each is a captured tensor that:
/// - no tensor holds any more;
/// - the plan binds once;
/// - is read by no pending value that a caller may still ask for, directly or through other
///   pending values.
///
/// A caller may ask for a value while a tensor of it is alive, unless the run computes it.
fn given_inputs<B: PortableBackend + 'static>(inner: &GraphInner<B>, plan: &Plan) -> ValueSet {
    let mut given = ValueSet::default();
    for &value in &plan.args {
        let ValueDef::Input(input) = &inner.values[value.0 as usize].def else {
            continue;
        };
        let bound_once = plan.args.iter().filter(|&&arg| arg == value).count() == 1;
        if input.role == InputRole::Arg && Arc::strong_count(input) == 1 && bound_once {
            given.insert(value);
        }
    }
    if given.is_empty() {
        return given;
    }
    let computed: ValueSet = plan.requested_outputs.iter().copied().collect();
    let mut stack = Vec::new();
    for (index, record) in inner.values.iter().enumerate() {
        let value = ValueId(index as u32);
        let pending = matches!(
            record.def,
            ValueDef::Node {
                state: NodeState::Pending,
                ..
            }
        );
        if pending && record.tensors.strong_count() > 0 && !computed.contains(&value) {
            stack.push(value);
        }
    }
    let mut seen = ValueSet::default();
    while let Some(value) = stack.pop() {
        if !seen.insert(value) {
            continue;
        }
        let record = &inner.values[value.0 as usize];
        match &record.def {
            ValueDef::Node {
                op,
                state: NodeState::Pending,
                ..
            } => stack.extend_from_slice(inner.operands(&inner.ops[*op as usize])),
            ValueDef::Input(_) => {
                given.remove(&value);
            }
            _ => {}
        }
    }
    given
}

/// Takes the handle of the captured tensor `value` for a run, which nothing else holds.
fn give<B: PortableBackend + 'static>(
    inner: &mut GraphInner<B>,
    value: ValueId,
) -> B::TensorHandle {
    let def = std::mem::replace(&mut inner.values[value.0 as usize].def, ValueDef::Given);
    let ValueDef::Input(input) = def else {
        unreachable!("only captured tensors are given");
    };
    match Arc::try_unwrap(input) {
        Ok(owned) => owned.tensor,
        Err(_) => unreachable!("the graph holds the only handle of a given tensor"),
    }
}

/// Tells the trace sink of `traced`, when there is one, how the run that began at `start` ended.
pub(super) fn report_program(
    traced: Option<&(Arc<dyn ExecutionTraceSink>, ProgramContext)>,
    start: Instant,
    output_count: usize,
    status: ProgramStatus,
) {
    if let Some((sink, context)) = traced {
        sink.after_program(
            context,
            &ProgramStats {
                duration: start.elapsed(),
                output_count,
                status,
            },
        );
    }
}

#[derive(Debug)]
struct StalePlanError;

impl fmt::Display for StalePlanError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "cached plan is stale after graph mutation")
    }
}

impl std::error::Error for StalePlanError {}

/// Returns the rank of each program output. The rank of a pending node is its position in
/// `context.ordered_values`. The rank of an input is the pending-node count plus its position among
/// the plan inputs. A materialized value that is requested again passes straight through as an
/// input.
fn ranks_of(
    context_nodes: &[ValueId],
    params: &[PlanInput],
    values: &[ValueId],
) -> Result<Vec<usize>> {
    values
        .iter()
        .map(|value| {
            context_nodes
                .binary_search_by_key(&value.0, |node| node.0)
                .or_else(|_| {
                    params
                        .binary_search_by_key(&value.0, |param| param.value.0)
                        .map(|index| context_nodes.len() + index)
                })
                .map_err(|_| anyhow!("program output {value:?} is neither a node nor an input"))
        })
        .collect()
}

fn value_at_rank(context_nodes: &[ValueId], params: &[PlanInput], rank: usize) -> Option<ValueId> {
    match rank.checked_sub(context_nodes.len()) {
        None => Some(context_nodes[rank]),
        Some(index) => params.get(index).map(|param| param.value),
    }
}

fn values_at_ranks(
    context_nodes: &[ValueId],
    params: &[PlanInput],
    ranks: &[usize],
) -> Result<Vec<ValueId>> {
    let mut values = Vec::with_capacity(ranks.len());
    for &rank in ranks {
        let value = value_at_rank(context_nodes, params, rank)
            .ok_or_else(|| anyhow!("program output rank {rank} is out of range"))?;
        values.push(value);
    }
    Ok(values)
}

enum PrepareResult<B: PortableBackend + 'static> {
    AllReady(Vec<B::TensorHandle>),
    NeedsPlan(Box<PlanContext>),
}

/// The values that the run-time arguments of `inputs` bind, in order.
fn arg_values(inputs: &[InputBinding], plan_inputs: &[PlanInput]) -> Result<Vec<ValueId>> {
    let mut values = Vec::with_capacity(inputs.len());
    for binding in inputs {
        let &InputBinding::Arg(index) = binding else {
            continue;
        };
        let input = plan_inputs
            .get(index)
            .ok_or_else(|| anyhow!("arg position {index} out of range"))?;
        values.push(input.value);
    }
    Ok(values)
}

struct PlanContext {
    key: PlanKey,
    ordered_values: Vec<ValueId>,
    inputs: Vec<PlanInput>,
    outputs: Vec<ValueId>,
    exports: Vec<ValueId>,
    arena_version: u64,
}

/// The module whose function `@captured` computes `outputs` from the plan inputs `inputs`, bound
/// to `params`, with the ops `nodes`.
fn build_module(
    inputs: &[PlanInput],
    params: &[Option<ParamId>],
    nodes: &[PlanNode],
    outputs: &[ValueId],
) -> Result<Module> {
    let mut module = Module::new();
    let types = inputs
        .iter()
        .map(|input| Type::Tensor(input.ty.clone()))
        .collect();
    module.build_func(
        "captured",
        types,
        params.to_vec(),
        Vec::new(),
        |builder, args| {
            let mut values: HashMap<ValueId, Value> = inputs
                .iter()
                .map(|input| input.value)
                .zip(args.iter().copied())
                .collect();
            // The op of each `first` value, built once for all of its results.
            let mut insts = HashMap::new();
            for node in nodes {
                let inst = match insts.get(&node.first) {
                    Some(&inst) => inst,
                    None => {
                        let operands = node
                            .operands
                            .iter()
                            .map(|operand| {
                                values
                                    .get(operand)
                                    .copied()
                                    .ok_or_else(|| anyhow!("{operand:?} is not computed"))
                            })
                            .collect::<Result<Vec<_>>>()?;
                        builder.set_location(node.location.clone());
                        let inst = builder.create_op(node.op.clone(), operands, None, Vec::new());
                        insts.insert(node.first, inst);
                        inst
                    }
                };
                values.insert(node.value, builder.module.results(inst)[node.result]);
            }
            outputs
                .iter()
                .map(|output| {
                    values
                        .get(output)
                        .copied()
                        .ok_or_else(|| anyhow!("output {output:?} is not computed"))
                })
                .collect()
        },
    )?;
    Ok(module)
}

/// Removes from the functions of `module` the parameter arguments that nothing reads. Returns the
/// binding of each argument `@captured` keeps: its parameter when it binds one, or else its
/// position among the plan inputs.
fn prune_params(module: &mut Module) -> Result<Vec<InputBinding>> {
    let func = module.func("captured").expect("the captured function");
    let attrs = func.attrs();
    let args = module.args(func.body().id());
    let bindings = args
        .iter()
        .enumerate()
        .filter_map(|(index, &arg)| match attrs.param(index) {
            Some(param) => (!module.uses(arg).is_empty()).then_some(InputBinding::Param(param.0)),
            None => Some(InputBinding::Arg(index)),
        })
        .collect();
    run_pass(&PruneParams, module)?;
    Ok(bindings)
}

/// The pending nodes of `refs` in value order, and the plan inputs it binds: ready nodes and
/// captured tensors.
fn plan_values<B: PortableBackend + 'static>(
    inner: &GraphInner<B>,
    refs: &[ValueRef],
    node_count: usize,
    input_count: usize,
) -> Result<(Vec<ValueId>, Vec<PlanInput>)> {
    let mut ordered = Vec::with_capacity(node_count);
    let mut inputs = Vec::with_capacity(input_count);
    for (index, value_ref) in refs.iter().enumerate() {
        let value = ValueId(index as u32);
        match value_ref {
            ValueRef::Node(_) => ordered.push(value),
            ValueRef::Input(_) => inputs.push(plan_input(inner, value)?),
            ValueRef::External => {}
        }
    }
    Ok((ordered, inputs))
}

/// The plan input that binds `value`: a parameter by its stable id, or a run-time argument.
fn plan_input<B: PortableBackend + 'static>(
    inner: &GraphInner<B>,
    value: ValueId,
) -> Result<PlanInput> {
    let record = &inner.values[value.0 as usize];
    let (role, stable_id) = match &record.def {
        ValueDef::Input(input) => (
            input.role,
            (input.role == InputRole::Param).then_some(input.id),
        ),
        ValueDef::Node { .. } => (InputRole::Arg, None),
        ValueDef::Given => {
            return Err(anyhow!(
                "{value:?} went to a program that may have updated it in place"
            ))
        }
    };
    Ok(PlanInput {
        value,
        role,
        stable_id,
        ty: record.tensor_type().clone(),
    })
}

/// What each value of `inner` is in the plan that computes `requested`, by value id: a pending
/// node that the plan runs, an input that it reads (a captured tensor or a computed node), or
/// neither. Nodes and inputs are numbered in value order. Also returns how many of each there
/// are.
fn plan_refs<B: PortableBackend + 'static>(
    inner: &GraphInner<B>,
    requested: &[ValueId],
) -> Result<(Vec<ValueRef>, usize, usize)> {
    let mut refs = vec![ValueRef::External; inner.values.len()];
    let mut stack = requested.to_vec();
    while let Some(value) = stack.pop() {
        let slot = refs
            .get_mut(value.0 as usize)
            .ok_or_else(|| anyhow!("value {value:?} not registered"))?;
        if *slot != ValueRef::External {
            continue;
        }
        match &inner.values[value.0 as usize].def {
            ValueDef::Node {
                op,
                state: NodeState::Pending,
                ..
            } => {
                *slot = ValueRef::Node(0);
                stack.extend_from_slice(inner.operands(&inner.ops[*op as usize]));
            }
            _ => *slot = ValueRef::Input(0),
        }
    }
    let (mut nodes, mut inputs) = (0, 0);
    for slot in &mut refs {
        match slot {
            ValueRef::Node(index) => {
                *index = nodes;
                nodes += 1;
            }
            ValueRef::Input(index) => {
                *index = inputs;
                inputs += 1;
            }
            ValueRef::External => {}
        }
    }
    Ok((refs, nodes as usize, inputs as usize))
}
