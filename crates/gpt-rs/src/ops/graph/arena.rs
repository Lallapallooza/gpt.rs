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
use gpt_rs_ir::dialects::func::{Func, ParamId};
use gpt_rs_ir::ir::{Module, Type, Value};
use gpt_rs_ir::passes::{self, Pass, Pipeline, PruneParams};

use super::builder::GraphBuilder;
use super::plan::{
    get_cached_program, insert_cached_program, CachedProgram, InputBinding, Plan, PlanGraph,
    PlanInput, PlanKey, PlanNode, PlanNodeView, ValueRef,
};
use super::state::{GraphInner, NodeState, ValueDef};
use crate::backend::PortableBackend;
use crate::ops::graph::ValueId;
use crate::ops::trace::{self, PassRun, ProgramContext, ProgramKind, ProgramStats, ProgramStatus};
use crate::tensor::{InputRole, Materialized};

/// Central storage for lazy tensor graphs built on top of a single backend instance.
pub struct GraphArena<B: PortableBackend + 'static> {
    backend: Arc<B>,
    inner: Mutex<GraphInner<B>>,
    id: usize,
}

static ARENA_ID_COUNTER: AtomicUsize = AtomicUsize::new(0);

impl<B: PortableBackend + 'static> GraphArena<B> {
    /// Creates a new arena wrapping the provided backend.
    pub fn new(backend: Arc<B>) -> Arc<Self> {
        let id = ARENA_ID_COUNTER.fetch_add(1, AtomicOrdering::Relaxed);
        Arc::new(GraphArena {
            backend,
            inner: Mutex::new(GraphInner::new()),
            id,
        })
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
    pub(crate) fn capture<R, F>(self: &Arc<Self>, f: F) -> Result<R>
    where
        F: FnOnce(&mut GraphBuilder<B>) -> Result<R>,
    {
        let mut inner = self.inner.lock().expect("graph arena poisoned");
        f(&mut GraphBuilder {
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
                    let entry_inputs = self.collect_entry_inputs(&plan)?;
                    match self.try_execute_plan(&plan, values, entry_inputs) {
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

        let (refs, nodes, inputs) = plan_refs(&inner, &requested)?;
        let mut ordered = Vec::with_capacity(nodes);
        let mut inputs = Vec::with_capacity(inputs);
        for (index, value_ref) in refs.iter().enumerate() {
            let value = ValueId(index as u32);
            match value_ref {
                ValueRef::Node(_) => ordered.push(value),
                ValueRef::Input(_) => {
                    let record = &inner.values[index];
                    let (role, stable_id) = match &record.def {
                        ValueDef::Input(input) => (
                            input.role,
                            (input.role == InputRole::Param).then_some(input.id),
                        ),
                        ValueDef::Node { .. } => (InputRole::Arg, None),
                    };
                    inputs.push(PlanInput {
                        value,
                        role,
                        stable_id,
                        ty: record.tensor_type().clone(),
                    });
                }
                ValueRef::External => {}
            }
        }

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
                    ty: inner.values[value.0 as usize].tensor_type(),
                    first: op.first,
                    result,
                })
            },
        };
        let key = PlanKey::new(self.backend.backend_name(), &graph)?;

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

        let cached = get_cached_program(&key);
        let mut compile_outputs = outputs.clone();
        if let Some(cached) = cached {
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
        let mut module = build_module(&plan_inputs, &params, &nodes, &compile_outputs)?;
        {
            let _scope = crate::profiling::compile_scope("optimizer::pass_pipeline");
            run_traced(&mut module, &self.backend.pipeline())?;
        }
        let inputs = prune_params(&mut module)?;
        let module = Arc::new(module);
        self.load_static_params(&module)?;
        let exe = {
            let _scope = crate::profiling::compile_scope("backend::compile");
            self.backend.compile(&module, "captured")?
        };
        let exe: Arc<dyn Any + Send + Sync> = Arc::new(exe);
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
            let attrs = module.op(func).get::<Func>().expect("a function");
            if !attrs.is_load_time() {
                continue;
            }
            let computed: Vec<u128> = attrs.result_params.iter().map(|p| p.0).collect();
            let params = self.backend.params();
            if params.get_all(computed.iter().copied()).is_some() {
                continue;
            }
            let leaves = (0..attrs.args.len())
                .map(|i| {
                    let id = attrs
                        .param(i)
                        .expect("a load-time function reads parameters");
                    self.param_handle(id.0)
                })
                .collect::<Result<Vec<_>>>()?;
            let _suspend = crate::profiling::suspend();
            let handles = self
                .backend
                .compile(module, &attrs.name)
                .and_then(|exe| self.backend.run(&exe, &leaves))
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
        let traced = trace::current_sink().map(|sink| {
            let context = ProgramContext {
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
            };
            (sink, context)
        });

        if let Some((sink, context)) = &traced {
            sink.before_program(context, &plan.module);
        }

        let start = Instant::now();
        let exe = plan
            .exe
            .downcast_ref::<B::Executable>()
            .ok_or_else(|| anyhow!("the plan was compiled by another backend type"))?;
        let exec = self.backend.run(exe, &entry_inputs);

        match exec {
            Ok(mut produced) => {
                if produced.len() != plan.program_outputs.len() {
                    if let Some((sink, context)) = &traced {
                        sink.after_program(
                            context,
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
                    for (value, handle) in plan.program_outputs.iter().zip(produced.drain(..)) {
                        if let Some(ValueDef::Node { state, .. }) = inner
                            .values
                            .get_mut(value.0 as usize)
                            .map(|record| &mut record.def)
                        {
                            *state = NodeState::Ready(handle);
                        }
                    }

                    self.collect_target_handles(&inner, targets)?
                };

                if let Some((sink, context)) = &traced {
                    sink.after_program(
                        context,
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
                if let Some((sink, context)) = &traced {
                    sink.after_program(
                        context,
                        &ProgramStats {
                            duration: start.elapsed(),
                            output_count: 0,
                            status: ProgramStatus::Failure {
                                message: err.to_string(),
                            },
                        },
                    );
                }
                Err(err)
            }
        }
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
        let inner = self.inner.lock().expect("graph arena poisoned");
        let mut args = plan.args.iter();
        let mut handles = Vec::with_capacity(plan.inputs.len());
        for binding in &plan.inputs {
            let handle = match binding {
                InputBinding::Param(_) => params.next().expect("one handle per parameter"),
                InputBinding::Arg(_) => {
                    let value = *args
                        .next()
                        .ok_or_else(|| anyhow!("plan binds more arguments than it records"))?;
                    self.arg_handle(&inner, value)?
                }
            };
            handles.push(handle);
        }
        Ok(handles)
    }

    fn arg_handle(&self, inner: &GraphInner<B>, value: ValueId) -> Result<B::TensorHandle> {
        self.ready_handle(inner, value)?
            .ok_or_else(|| anyhow!("arg value {value:?} pending while collecting entry inputs"))
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
                .binary_search_by_key(&value.0, |v| v.0)
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

/// Runs the passes of `pipeline` on `module`, verifying after each. A trace sink receives the
/// module after each pass.
fn run_traced(module: &mut Module, pipeline: &Pipeline) -> Result<()> {
    let sink = trace::current_sink();
    for pass in pipeline.passes() {
        let _scope = crate::profiling::compile_pass_scope(pass.name());
        let changed = passes::run_pass(pass.as_ref(), module)?;
        let stats = passes::take_stats();
        if let Some(sink) = &sink {
            let run = PassRun {
                name: pass.name(),
                options: &pass.options(),
                changed,
                stats: &stats,
            };
            sink.on_pass(&run, module);
        }
    }
    Ok(())
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
    module.build_func("captured", types, params.to_vec(), |b, args| {
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
                        .map(|v| {
                            values
                                .get(v)
                                .copied()
                                .ok_or_else(|| anyhow!("{v:?} is not computed"))
                        })
                        .collect::<Result<Vec<_>>>()?;
                    b.set_location(node.location.clone());
                    let inst = b.create_op(node.op.clone(), operands, None, Vec::new());
                    insts.insert(node.first, inst);
                    inst
                }
            };
            values.insert(node.value, b.module.results(inst)[node.result]);
        }
        outputs
            .iter()
            .map(|v| {
                values
                    .get(v)
                    .copied()
                    .ok_or_else(|| anyhow!("output {v:?} is not computed"))
            })
            .collect()
    })?;
    Ok(module)
}

/// Removes from the functions of `module` the parameter arguments that nothing reads. Returns the
/// binding of each argument `@captured` keeps: its parameter when it binds one, or else its
/// position among the plan inputs.
fn prune_params(module: &mut Module) -> Result<Vec<InputBinding>> {
    let func = module.func("captured").expect("the captured function");
    let attrs = module.op(func).get::<Func>().expect("a function");
    let args = module.args(module.regions(func)[0]);
    let bindings = args
        .iter()
        .enumerate()
        .filter_map(|(index, &arg)| match attrs.param(index) {
            Some(param) => (!module.uses(arg).is_empty()).then_some(InputBinding::Param(param.0)),
            None => Some(InputBinding::Arg(index)),
        })
        .collect();
    PruneParams.run(module)?;
    Ok(bindings)
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
