//! Functions that run as one compiled program, recorded once for each signature of their inputs.
//!
//! A tensor op records into a graph every time it runs, so a forward of hundreds of ops costs
//! hundreds of recordings before its program runs. A [`Traced`] function records its graph on the
//! first call with inputs of some signature, and on later calls with the same one runs the program
//! of that graph on the new inputs directly. The code it runs stays ordinary tensor code: the
//! runtime traces its fixed entry points, such as the forward of an image model.

use std::any::Any;
use std::cell::Cell;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime};

use anyhow::{anyhow, ensure, Result};
use gpt_rs_ir::ir::{Module, TensorType};

use super::arena::{report_program, ProgramRun, RunInput};
use super::context::push_default_arena;
use super::plan::PlanKey;
use super::GraphArena;
use crate::backend::PortableBackend;
use crate::nn::capture;
use crate::ops::graph::ValueId;
use crate::ops::trace::{self, ProgramContext, ProgramKind, ProgramStatus};
use crate::tensor::{DType, DeviceTensor, LazyHandle, Shape};

thread_local! {
    /// While a trace records on this thread, whether the function it records materialized a
    /// tensor: ran its program, or read its values back to the host.
    static MATERIALIZED: Cell<Option<bool>> = const { Cell::new(None) };
}

/// Notes that a tensor was materialized, which a trace that records on this thread cannot replay:
/// what the function does next may depend on the tensor's values.
pub(crate) fn note_materialized() {
    MATERIALIZED.with(|materialized| {
        if materialized.get().is_some() {
            materialized.set(Some(true));
        }
    });
}

/// The function that a [`Traced`] runs: tensors in, tensors out.
pub type TracedFn<B> = dyn Fn(&[DeviceTensor<B>]) -> Vec<DeviceTensor<B>> + Send + Sync;

/// A function of tensors that [`Traces`] are kept for, given at each call.
pub type TracesFn<'f, B> = dyn Fn(&[DeviceTensor<B>]) -> Vec<DeviceTensor<B>> + 'f;

/// The results of a function, recorded but not computed, and the value of each in the arena that
/// recorded it.
type Recorded<B> = (Vec<DeviceTensor<B>>, Vec<Option<ValueId>>);

/// What a trace is recorded for: the dtype and extents of each input, and for each input the
/// position of the first input that is the same tensor, which the recorded graph reads as one.
#[derive(Clone, PartialEq, Eq, Hash)]
struct Signature {
    types: Vec<TensorType>,
    first_alias: Vec<usize>,
}

impl Signature {
    fn of<B: PortableBackend + 'static>(inputs: &[DeviceTensor<B>]) -> Signature {
        let mut types = Vec::with_capacity(inputs.len());
        let mut first_alias = Vec::with_capacity(inputs.len());
        for (position, input) in inputs.iter().enumerate() {
            types.push(input.tensor_type());
            let mut first = position;
            for (earlier, other) in inputs[..position].iter().enumerate() {
                if input.same_tensor(other) {
                    first = earlier;
                    break;
                }
            }
            first_alias.push(first);
        }
        Signature { types, first_alias }
    }
}

/// A function of tensors, run on one backend, that records its graph once for each signature of
/// its inputs: their dtypes, their extents, and which of them are the same tensor. Later calls
/// with the same signature run the program of that graph on their inputs, without recording.
///
/// The graph it records must depend on nothing but the signature, so state that changes between
/// calls, such as a cache, must be an input. A trace that cannot be replayed records every call
/// instead, as the function would without a trace. Such a trace, while it records:
/// - materializes a tensor or reads its values back, so that the graph may depend on them;
/// - reads a tensor that is neither an input nor a parameter, such as one made before the trace or
///   one made from host data;
/// - does not run exactly one program, returns a tensor that the program does not compute, or
///   marks for export a value that it does not return.
///
/// Calls also record while debugging asks for every op as it runs (`GPTRS_EAGER`), or while
/// module outputs are captured. With `GPTRS_TRACE_CHECK` set, every call records and fails when
/// its graph is not the one the trace replays. It keeps one trace for each signature it is called
/// with ([`Traces`]).
pub struct Traced<B: PortableBackend + 'static> {
    function: Box<TracedFn<B>>,
    traces: Traces<B>,
}

/// The traces of one function, one for each signature of its inputs, as [`Traced`] keeps them for
/// the function it owns. The function is given at each call, so that it may borrow what it runs,
/// such as the model of a decoding state. Each call must give the same function.
pub struct Traces<B: PortableBackend + 'static> {
    backend: Arc<B>,
    traces: Mutex<HashMap<Signature, Arc<Trace>>>,
}

/// How a traced function runs on inputs of one signature.
enum Trace {
    /// The program of its graph.
    Replay(Replay),
    /// Recorded every call.
    Recorded,
}

/// The program of a traced function's graph, and where its arguments and results are.
struct Replay {
    key: PlanKey,
    exe: Arc<dyn Any + Send + Sync>,
    module: Arc<Module>,
    /// The arena that recorded it, and the value of each result there, which trace sinks see.
    graph_id: usize,
    values: Vec<ValueId>,
    /// Where each argument of the program comes from.
    args: Vec<ReplayArg>,
    /// The result of the program that is each result of the function, and its shape and dtype.
    results: Vec<(usize, Shape, DType)>,
}

/// An argument of a replayed program.
enum ReplayArg {
    /// The input of the function at this position.
    Input(usize),
    /// The parameter with this id, read from the backend's parameter store at each run.
    Param(u128),
}

impl<B: PortableBackend + 'static> Traced<B> {
    /// `function` on `backend`, traced.
    pub fn new(
        backend: Arc<B>,
        function: impl Fn(&[DeviceTensor<B>]) -> Vec<DeviceTensor<B>> + Send + Sync + 'static,
    ) -> Self {
        Traced {
            function: Box::new(function),
            traces: Traces::new(backend),
        }
    }

    /// The results of the function on `inputs`, computed: by the program of the trace of their
    /// signature, recording that trace on its first call. A run may update in place an input that
    /// nothing else holds.
    pub fn call(&self, inputs: Vec<DeviceTensor<B>>) -> Result<Vec<DeviceTensor<B>>> {
        self.traces.call(inputs, self.function.as_ref())
    }
}

impl<B: PortableBackend + 'static> Traces<B> {
    /// No traces yet, of a function that runs on `backend`.
    pub fn new(backend: Arc<B>) -> Self {
        Traces {
            backend,
            traces: Mutex::default(),
        }
    }

    /// The results of `function` on `given`, computed: by the program of the trace of their
    /// signature, recording that trace on its first call. A run may update in place an input that
    /// nothing else holds.
    pub fn call(
        &self,
        given: Vec<DeviceTensor<B>>,
        function: &TracesFn<'_, B>,
    ) -> Result<Vec<DeviceTensor<B>>> {
        let backend = &self.backend;
        let inputs = own_inputs(backend, given)?;
        if crate::env::eager_enabled() || capture::capturing() {
            let arena = GraphArena::new(Arc::clone(backend));
            return compute(backend, record(&arena, inputs, function).0);
        }
        let signature = Signature::of(&inputs);
        let cached = self.lock().get(&signature).cloned();
        let Some(trace) = cached else {
            let (results, recorded) = trace_of(backend, inputs, function)?;
            self.lock().insert(signature, Arc::new(recorded));
            return Ok(results);
        };
        match &*trace {
            Trace::Recorded => {
                let arena = GraphArena::new(Arc::clone(backend));
                compute(backend, record(&arena, inputs, function).0)
            }
            Trace::Replay(replay) if crate::env::trace_check_enabled() => {
                replay.check(backend, inputs, function)
            }
            Trace::Replay(replay) => replay.run(backend, inputs),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<Signature, Arc<Trace>>> {
        self.traces.lock().expect("traces poisoned")
    }
}

/// `inputs`, each a tensor of its own. An input that another program computed is read through its
/// computed handle, so that a trace takes it as an input rather than as a value of that program,
/// which its graph would have to import. Inputs that are one tensor stay one.
fn own_inputs<B: PortableBackend + 'static>(
    backend: &Arc<B>,
    inputs: Vec<DeviceTensor<B>>,
) -> Result<Vec<DeviceTensor<B>>> {
    let mut own: Vec<DeviceTensor<B>> = Vec::with_capacity(inputs.len());
    for (position, input) in inputs.iter().enumerate() {
        let alias = inputs[..position]
            .iter()
            .position(|other| input.same_tensor(other));
        if let Some(earlier) = alias {
            own.push(own[earlier].clone());
            continue;
        }
        own.push(match input.lazy_handle() {
            LazyHandle::Input(_) => input.clone(),
            LazyHandle::Node { .. } => DeviceTensor::from_handle(
                Arc::clone(backend),
                input.shape().clone(),
                input.dtype(),
                input.materialize()?,
            ),
        });
    }
    Ok(own)
}

/// The results of `function` on `inputs`, recorded into `arena`, and the value of each result
/// there. It drops `inputs` before it returns, so that the run of the results may take an input
/// that nothing else holds.
fn record<B: PortableBackend + 'static>(
    arena: &Arc<GraphArena<B>>,
    inputs: Vec<DeviceTensor<B>>,
    function: &TracesFn<'_, B>,
) -> Recorded<B> {
    let _arena = push_default_arena(Arc::clone(arena));
    let results = function(&inputs);
    drop(inputs);
    let mut values = Vec::with_capacity(results.len());
    for result in &results {
        let value = match result.graph_value() {
            Some((graph, value)) if Arc::ptr_eq(&graph, arena) => Some(value),
            _ => None,
        };
        values.push(value);
    }
    (results, values)
}

/// `results`, computed.
fn compute<B: PortableBackend + 'static>(
    backend: &Arc<B>,
    results: Vec<DeviceTensor<B>>,
) -> Result<Vec<DeviceTensor<B>>> {
    let mut lazy = Vec::with_capacity(results.len());
    for result in &results {
        lazy.push(result);
    }
    let handles = DeviceTensor::materialize_many(&lazy)?;
    let mut computed = Vec::with_capacity(results.len());
    for (result, handle) in results.iter().zip(handles) {
        computed.push(DeviceTensor::from_handle(
            Arc::clone(backend),
            result.shape().clone(),
            result.dtype(),
            handle,
        ));
    }
    Ok(computed)
}

/// The results of `function` on `inputs`, recorded once and computed, and the trace of their
/// signature.
fn trace_of<B: PortableBackend + 'static>(
    backend: &Arc<B>,
    inputs: Vec<DeviceTensor<B>>,
    function: &TracesFn<'_, B>,
) -> Result<(Vec<DeviceTensor<B>>, Trace)> {
    let mut input_ids = Vec::with_capacity(inputs.len());
    for input in &inputs {
        input_ids.push(match input.lazy_handle() {
            LazyHandle::Input(captured) => Some(captured.id),
            LazyHandle::Node { .. } => None,
        });
    }
    let arena = GraphArena::recording_runs(Arc::clone(backend));
    let around = MATERIALIZED.replace(Some(false));
    let (results, values) = record(&arena, inputs, function);
    let materialized = MATERIALIZED.replace(around) == Some(true);
    let computed = compute(backend, results)?;
    let runs = arena.take_runs();
    let replay = if materialized {
        None
    } else {
        replay_of(&arena, runs, &input_ids, &computed, &values)
    };
    let trace = replay.map_or(Trace::Recorded, Trace::Replay);
    Ok((computed, trace))
}

/// The replay of the one program that `arena` ran for a trace of inputs whose captured tensors
/// have the ids `input_ids`, when it computed every result, reads only the inputs and parameters,
/// and exports only results, which the replay returns.
fn replay_of<B: PortableBackend + 'static>(
    arena: &GraphArena<B>,
    runs: Vec<ProgramRun>,
    input_ids: &[Option<u128>],
    results: &[DeviceTensor<B>],
    values: &[Option<ValueId>],
) -> Option<Replay> {
    let [run] = <[ProgramRun; 1]>::try_from(runs).ok()?;
    for export in &run.exports {
        if !values.contains(&Some(*export)) {
            return None;
        }
    }
    if input_ids.contains(&None) {
        return None;
    }
    let mut args = Vec::with_capacity(run.inputs.len());
    for input in &run.inputs {
        let arg = match input {
            RunInput::Param(id) => ReplayArg::Param(*id),
            RunInput::Captured(captured) => {
                let position = input_ids.iter().position(|&id| id == Some(*captured))?;
                ReplayArg::Input(position)
            }
            RunInput::Computed => return None,
        };
        args.push(arg);
    }
    let mut computed = Vec::with_capacity(values.len());
    let mut traced_values = Vec::with_capacity(values.len());
    for (&result_value, result) in values.iter().zip(results) {
        let value = result_value?;
        let position = run.outputs.iter().position(|&output| output == value)?;
        computed.push((position, result.shape().clone(), result.dtype()));
        traced_values.push(value);
    }
    Some(Replay {
        key: run.key,
        exe: run.exe,
        module: run.module,
        graph_id: arena.id(),
        values: traced_values,
        args,
        results: computed,
    })
}

impl Replay {
    /// The results of the program on `inputs`.
    fn run<B: PortableBackend + 'static>(
        &self,
        backend: &Arc<B>,
        inputs: Vec<DeviceTensor<B>>,
    ) -> Result<Vec<DeviceTensor<B>>> {
        let mut given: Vec<Option<DeviceTensor<B>>> = inputs.into_iter().map(Some).collect();
        let mut args = Vec::with_capacity(self.args.len());
        for arg in &self.args {
            let handle = match arg {
                ReplayArg::Input(position) => given[*position]
                    .take()
                    .ok_or_else(|| anyhow!("the traced program takes input {position} twice"))?
                    .into_handle()?,
                ReplayArg::Param(id) => backend.params().get(*id).ok_or_else(|| {
                    anyhow!("the traced program reads parameter {id}, not stored")
                })?,
            };
            args.push(handle);
        }
        let exe = self
            .exe
            .downcast_ref::<B::Executable>()
            .ok_or_else(|| anyhow!("the trace was compiled by another backend type"))?;
        let traced = trace::current_sink().map(|sink| {
            let context = ProgramContext {
                trace_id: trace::next_trace_id(),
                graph_id: self.graph_id,
                backend: backend.backend_name().to_string(),
                plan_hash: self.key.0,
                program_cache_hit: true,
                targets: self.values.clone(),
                outputs: self.values.clone(),
                exports: Vec::new(),
                timestamp: SystemTime::now(),
                kind: ProgramKind::Replay,
            };
            sink.before_program(&context, &self.module);
            (sink, context)
        });
        let start = Instant::now();
        let outcome = backend.run(exe, args);
        let status = match &outcome {
            Ok(_) => ProgramStatus::Success,
            Err(err) => ProgramStatus::Failure {
                message: err.to_string(),
            },
        };
        let output_count = outcome.as_ref().map_or(0, Vec::len);
        report_program(traced.as_ref(), start, output_count, status);
        let produced = outcome?;
        let mut results = Vec::with_capacity(self.results.len());
        for (position, shape, dtype) in &self.results {
            let handle = produced
                .get(*position)
                .ok_or_else(|| anyhow!("the program gives {} results", produced.len()))?;
            results.push(DeviceTensor::from_handle(
                Arc::clone(backend),
                shape.clone(),
                *dtype,
                handle.clone(),
            ));
        }
        Ok(results)
    }

    /// The results of `function` on `inputs`, recorded again, after checking that it records the
    /// graph of this replay.
    fn check<B: PortableBackend + 'static>(
        &self,
        backend: &Arc<B>,
        inputs: Vec<DeviceTensor<B>>,
        function: &TracesFn<'_, B>,
    ) -> Result<Vec<DeviceTensor<B>>> {
        let arena = GraphArena::recording_runs(Arc::clone(backend));
        let (results, _) = record(&arena, inputs, function);
        let computed = compute(backend, results)?;
        let runs = arena.take_runs();
        let keys: Vec<PlanKey> = runs.iter().map(|run| run.key).collect();
        ensure!(
            keys == [self.key],
            "a traced function recorded a graph other than the one its trace replays: its graph \
             depends on more than the dtypes and extents of its inputs"
        );
        Ok(computed)
    }
}
