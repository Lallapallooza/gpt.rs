//! A traced function records its graph on the first call with inputs of some signature: their
//! dtypes, their extents, and which of them are the same tensor. Later calls with that signature
//! run the program of that graph on their inputs. A function whose graph may depend on more than
//! that records every call.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::{self, ThreadId};

use anyhow::Result;
use gpt_rs::backend::PortableBackend;
use gpt_rs::gpt_rs_ir::ir::Module;
use gpt_rs::ops::graph::Traced;
use gpt_rs::ops::trace::{self, ExecutionTraceSink, ProgramContext, ProgramStats};
use gpt_rs::tensor::{DeviceTensor, HostTensor};
use gpt_rs_backend_ref_cpu::CpuPortableBackend;

type Tensor = DeviceTensor<CpuPortableBackend>;

/// The kind of each program run, with the thread that ran it.
#[derive(Default)]
struct ProgramKinds {
    runs: Mutex<Vec<(ThreadId, String)>>,
}

impl ExecutionTraceSink for ProgramKinds {
    fn before_program(&self, context: &ProgramContext, _module: &Module) {
        let run = (thread::current().id(), context.kind.label().to_string());
        self.runs.lock().unwrap().push(run);
    }

    fn after_program(&self, _context: &ProgramContext, _stats: &ProgramStats) {}
}

/// The sink of every test here, installed once: each test runs on a thread of its own and reads
/// the programs of its thread.
fn programs() -> &'static ProgramKinds {
    static SINK: OnceLock<Arc<ProgramKinds>> = OnceLock::new();
    SINK.get_or_init(|| {
        let sink = Arc::new(ProgramKinds::default());
        let installed =
            trace::install_global_sink(Arc::clone(&sink) as Arc<dyn ExecutionTraceSink>);
        std::mem::forget(installed);
        sink
    })
}

/// The kinds of the programs this thread ran.
fn kinds_here() -> Vec<String> {
    let here = thread::current().id();
    let runs = programs().runs.lock().unwrap();
    let mut kinds = Vec::new();
    for (thread, kind) in runs.iter() {
        if *thread == here {
            kinds.push(kind.clone());
        }
    }
    kinds
}

/// What `run` returns, and the kind of each program that it runs.
fn kinds_of<T>(run: impl FnOnce() -> Result<T>) -> Result<(T, Vec<String>)> {
    let before = kinds_here().len();
    let value = run()?;
    Ok((value, kinds_here()[before..].to_vec()))
}

fn tensor(backend: &Arc<CpuPortableBackend>, values: &[f32]) -> Tensor {
    let host = HostTensor::f32([values.len()], values.to_vec()).unwrap();
    DeviceTensor::from_host(Arc::clone(backend), host).unwrap()
}

fn values(tensor: &Tensor) -> Vec<f32> {
    tensor.to_host().unwrap().as_f32().unwrap().to_vec()
}

/// `2x + 1` of `x`, traced.
fn affine(inputs: &[Tensor]) -> Vec<Tensor> {
    vec![&inputs[0] * 2.0 + 1.0]
}

#[test]
fn a_traced_function_runs_the_program_of_its_trace_on_new_inputs() -> Result<()> {
    let backend = Arc::new(CpuPortableBackend::new());
    let traced = Traced::new(Arc::clone(&backend), affine);
    let first = tensor(&backend, &[1.0, 2.0, 3.0]);
    let (first_results, first_kinds) = kinds_of(|| traced.call(vec![first]))?;
    assert_eq!(values(&first_results[0]), [3.0, 5.0, 7.0]);
    assert_eq!(first_kinds, ["materialize"]);

    let second = tensor(&backend, &[4.0, 5.0, 6.0]);
    let (second_results, second_kinds) = kinds_of(|| traced.call(vec![second]))?;
    assert_eq!(values(&second_results[0]), [9.0, 11.0, 13.0]);
    assert_eq!(second_kinds, ["replay"]);
    Ok(())
}

#[test]
fn inputs_of_another_shape_record_a_trace_of_their_own() -> Result<()> {
    let backend = Arc::new(CpuPortableBackend::new());
    let traced = Traced::new(Arc::clone(&backend), affine);
    let three = tensor(&backend, &[1.0, 2.0, 3.0]);
    let (_, three_kinds) = kinds_of(|| traced.call(vec![three]))?;
    assert_eq!(three_kinds, ["materialize"]);

    let four = tensor(&backend, &[1.0, 2.0, 3.0, 4.0]);
    let (four_results, four_kinds) = kinds_of(|| traced.call(vec![four]))?;
    assert_eq!(values(&four_results[0]), [3.0, 5.0, 7.0, 9.0]);
    assert_eq!(four_kinds, ["materialize"]);

    let three_again = tensor(&backend, &[5.0, 6.0, 7.0]);
    let (three_again_results, three_again_kinds) = kinds_of(|| traced.call(vec![three_again]))?;
    assert_eq!(values(&three_again_results[0]), [11.0, 13.0, 15.0]);
    assert_eq!(three_again_kinds, ["replay"]);
    Ok(())
}

#[test]
fn a_function_that_reads_values_back_while_it_records_records_every_call() -> Result<()> {
    let backend = Arc::new(CpuPortableBackend::new());
    // Shifts `x` by its first element, read back on the host.
    let traced = Traced::new(Arc::clone(&backend), |inputs: &[Tensor]| {
        let first = values(&inputs[0])[0];
        vec![&inputs[0] + first]
    });
    let first = tensor(&backend, &[1.0, 2.0, 3.0]);
    let (first_results, _) = kinds_of(|| traced.call(vec![first]))?;
    assert_eq!(values(&first_results[0]), [2.0, 3.0, 4.0]);

    let second = tensor(&backend, &[5.0, 6.0, 7.0]);
    let (second_results, second_kinds) = kinds_of(|| traced.call(vec![second]))?;
    assert_eq!(values(&second_results[0]), [10.0, 11.0, 12.0]);
    assert_eq!(second_kinds, ["materialize"]);
    Ok(())
}

#[test]
fn a_function_that_makes_tensors_from_host_data_records_every_call() -> Result<()> {
    let backend = Arc::new(CpuPortableBackend::new());
    // Adds the number of the call, uploaded as a tensor of its own.
    let calls = AtomicUsize::new(0);
    let uploads_to = Arc::clone(&backend);
    let traced = Traced::new(Arc::clone(&backend), move |inputs: &[Tensor]| {
        let call = calls.fetch_add(1, Ordering::Relaxed) + 1;
        let offset = tensor(&uploads_to, &[call as f32; 3]);
        vec![&inputs[0] + &offset]
    });
    let first = tensor(&backend, &[1.0, 2.0, 3.0]);
    let (first_results, _) = kinds_of(|| traced.call(vec![first]))?;
    assert_eq!(values(&first_results[0]), [2.0, 3.0, 4.0]);

    let second = tensor(&backend, &[1.0, 2.0, 3.0]);
    let (second_results, second_kinds) = kinds_of(|| traced.call(vec![second]))?;
    assert_eq!(values(&second_results[0]), [3.0, 4.0, 5.0]);
    assert_eq!(second_kinds, ["materialize"]);
    Ok(())
}

#[test]
fn a_function_that_materializes_a_tensor_while_it_records_records_every_call() -> Result<()> {
    let backend = Arc::new(CpuPortableBackend::new());
    // Doubles `x`, and adds 1 unless the first element of the double, downloaded from the
    // materialized tensor, is negative.
    let downloads_from = Arc::clone(&backend);
    let traced = Traced::new(Arc::clone(&backend), move |inputs: &[Tensor]| {
        let doubled = &inputs[0] * 2.0;
        let handle = doubled.materialize().unwrap();
        let host = downloads_from.download(&handle).unwrap();
        match host.as_f32().unwrap()[0] < 0.0 {
            true => vec![doubled],
            false => vec![doubled + 1.0],
        }
    });
    let negative = tensor(&backend, &[-1.0, 2.0, 3.0]);
    let negative_results = traced.call(vec![negative])?;
    assert_eq!(values(&negative_results[0]), [-2.0, 4.0, 6.0]);

    let positive = tensor(&backend, &[1.0, 2.0, 3.0]);
    let positive_results = traced.call(vec![positive])?;
    assert_eq!(values(&positive_results[0]), [3.0, 5.0, 7.0]);
    Ok(())
}

#[test]
fn one_tensor_passed_twice_and_two_tensors_have_traces_of_their_own() -> Result<()> {
    let backend = Arc::new(CpuPortableBackend::new());
    let traced = Traced::new(Arc::clone(&backend), |inputs: &[Tensor]| {
        vec![&inputs[0] + &inputs[1] * 2.0]
    });
    let both = tensor(&backend, &[1.0, 2.0, 3.0]);
    let (twice_results, twice_kinds) = kinds_of(|| traced.call(vec![both.clone(), both]))?;
    assert_eq!(values(&twice_results[0]), [3.0, 6.0, 9.0]);
    assert_eq!(twice_kinds, ["materialize"]);

    let first = tensor(&backend, &[1.0, 2.0, 3.0]);
    let second = tensor(&backend, &[10.0, 20.0, 30.0]);
    let (pair_results, pair_kinds) = kinds_of(|| traced.call(vec![first, second]))?;
    assert_eq!(values(&pair_results[0]), [21.0, 42.0, 63.0]);
    assert_eq!(pair_kinds, ["materialize"]);
    Ok(())
}
