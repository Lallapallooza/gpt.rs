use std::sync::{Arc, Mutex};

use gpt_rs::backend::{ParamStore, PortableBackend};
use gpt_rs::gpt_rs_ir::module::Module;
use gpt_rs::ops::graph::context::with_default_arena;
use gpt_rs::ops::graph::GraphArena;
use gpt_rs::ops::trace::{self, ExecutionTraceSink, ProgramContext, ProgramStats};
use gpt_rs::tensor::{DType, DeviceTensor, HostTensor};
use gpt_rs_backend_ref_cpu::CpuPortableBackend;
use gpt_rs_backend_tests::tensor_as;

type CpuHandle = <CpuPortableBackend as PortableBackend>::TensorHandle;
static TRACE_TEST_MUTEX: Mutex<()> = Mutex::new(());

struct NamedCpuBackend {
    name: &'static str,
    inner: Arc<CpuPortableBackend>,
}

impl NamedCpuBackend {
    fn new(name: &'static str) -> Self {
        Self {
            name,
            inner: Arc::new(CpuPortableBackend::new()),
        }
    }
}

impl PortableBackend for NamedCpuBackend {
    type TensorHandle = CpuHandle;
    type Executable = <CpuPortableBackend as PortableBackend>::Executable;

    fn backend_name(&self) -> &str {
        self.name
    }

    fn params(&self) -> &ParamStore<Self::TensorHandle> {
        self.inner.params()
    }

    fn compile(&self, module: &Arc<Module>, func: &str) -> anyhow::Result<Self::Executable> {
        self.inner.compile(module, func)
    }

    fn run(
        &self,
        exe: &Self::Executable,
        args: &[Self::TensorHandle],
    ) -> anyhow::Result<Vec<Self::TensorHandle>> {
        self.inner.run(exe, args)
    }

    fn upload(&self, host: HostTensor) -> anyhow::Result<Self::TensorHandle> {
        self.inner.upload(host)
    }

    fn download(&self, tensor: &Self::TensorHandle) -> anyhow::Result<HostTensor> {
        self.inner.download(tensor)
    }
}

struct ContextSink {
    contexts: Mutex<Vec<ProgramContext>>,
}

impl ExecutionTraceSink for ContextSink {
    fn before_program(&self, context: &ProgramContext, _module: &Module) {
        self.contexts
            .lock()
            .expect("trace context sink mutex poisoned")
            .push(context.clone());
    }

    fn after_program(&self, _context: &ProgramContext, _stats: &ProgramStats) {}
}

fn tensor_from_data_with_shape(
    backend: &Arc<NamedCpuBackend>,
    shape: &[usize],
    data: &[f32],
) -> anyhow::Result<DeviceTensor<NamedCpuBackend>> {
    let host = HostTensor::f32(shape, data.to_vec())?;
    DeviceTensor::from_host(Arc::clone(backend), host)
}

fn tensor_from_data(
    backend: &Arc<NamedCpuBackend>,
    data: &[f32],
) -> anyhow::Result<DeviceTensor<NamedCpuBackend>> {
    tensor_from_data_with_shape(backend, &[2, 2], data)
}

fn contexts_after_two_runs(contexts: &[ProgramContext]) -> (&ProgramContext, &ProgramContext) {
    let len = contexts.len();
    assert!(
        len >= 2,
        "expected at least two program executions, got {len}"
    );
    (&contexts[len - 2], &contexts[len - 1])
}

#[test]
fn literal_value_change_does_not_reuse_program_cache() -> anyhow::Result<()> {
    let _serial_guard = TRACE_TEST_MUTEX.lock().expect("trace test mutex poisoned");
    let backend = Arc::new(NamedCpuBackend::new("cpu-literal-change-cache-test"));
    let sink = Arc::new(ContextSink {
        contexts: Mutex::new(Vec::new()),
    });
    let _trace_guard = trace::install_global_sink(sink.clone() as Arc<dyn ExecutionTraceSink>);

    let input1 = tensor_from_data(&backend, &[1.0, 2.0, 3.0, 4.0])?;
    let out1 = (&input1 + 1.0).to_host()?;
    assert_eq!(out1.as_f32().unwrap(), &[2.0, 3.0, 4.0, 5.0]);

    let input2 = tensor_from_data(&backend, &[1.0, 2.0, 3.0, 4.0])?;
    let out2 = (&input2 + 2.0).to_host()?;
    assert_eq!(out2.as_f32().unwrap(), &[3.0, 4.0, 5.0, 6.0]);

    let contexts = sink
        .contexts
        .lock()
        .expect("trace context sink mutex poisoned")
        .clone();
    let (first, second) = contexts_after_two_runs(&contexts);
    assert!(
        !first.program_cache_hit,
        "first execution unexpectedly hit program cache"
    );
    assert!(
        !second.program_cache_hit,
        "program cache was reused across different literal values"
    );
    assert_ne!(first.plan_hash, second.plan_hash);

    Ok(())
}

#[test]
fn identical_literal_graph_reuses_program_cache() -> anyhow::Result<()> {
    let _serial_guard = TRACE_TEST_MUTEX.lock().expect("trace test mutex poisoned");
    let backend = Arc::new(NamedCpuBackend::new("cpu-identical-literal-cache-test"));
    let sink = Arc::new(ContextSink {
        contexts: Mutex::new(Vec::new()),
    });
    let _trace_guard = trace::install_global_sink(sink.clone() as Arc<dyn ExecutionTraceSink>);

    let input1 = tensor_from_data(&backend, &[5.0, 6.0, 7.0, 8.0])?;
    let _ = (&input1 + 1.5).to_host()?;

    let input2 = tensor_from_data(&backend, &[1.0, 2.0, 3.0, 4.0])?;
    let _ = (&input2 + 1.5).to_host()?;

    let contexts = sink
        .contexts
        .lock()
        .expect("trace context sink mutex poisoned")
        .clone();
    let (first, second) = contexts_after_two_runs(&contexts);
    assert!(
        !first.program_cache_hit,
        "first execution unexpectedly hit program cache"
    );
    assert!(
        second.program_cache_hit,
        "expected second identical graph to hit program cache"
    );
    assert_eq!(first.plan_hash, second.plan_hash);

    Ok(())
}

#[test]
fn shape_change_does_not_reuse_program_cache() -> anyhow::Result<()> {
    let _serial_guard = TRACE_TEST_MUTEX.lock().expect("trace test mutex poisoned");
    let backend = Arc::new(NamedCpuBackend::new("cpu-shape-specialization-cache-test"));
    let sink = Arc::new(ContextSink {
        contexts: Mutex::new(Vec::new()),
    });
    let _trace_guard = trace::install_global_sink(sink.clone() as Arc<dyn ExecutionTraceSink>);

    let input1 = tensor_from_data_with_shape(&backend, &[2, 2], &[1.0, 2.0, 3.0, 4.0])?;
    let _ = (&input1 + 1.0).to_host()?;

    let input2 = tensor_from_data_with_shape(
        &backend,
        &[4, 4],
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ],
    )?;
    let _ = (&input2 + 1.0).to_host()?;

    let contexts = sink
        .contexts
        .lock()
        .expect("trace context sink mutex poisoned")
        .clone();
    let (first, second) = contexts_after_two_runs(&contexts);
    assert!(
        !second.program_cache_hit,
        "shape specialization change should not reuse cached program"
    );
    assert_ne!(first.plan_hash, second.plan_hash);

    Ok(())
}

// `plan_hash` covers the graph structure and every specialization: shapes, dtypes, literals and
// bound parameters.

type Dt = DeviceTensor<NamedCpuBackend>;

fn input(backend: &Arc<NamedCpuBackend>, shape: &[usize], dtype: DType) -> Dt {
    let values: Vec<f32> = (0..shape.iter().product::<usize>())
        .map(|i| i as f32 * 0.25 - 1.0)
        .collect();
    let host = tensor_as(shape, &values, dtype);
    DeviceTensor::from_host(Arc::clone(backend), host).unwrap()
}

/// Runs `build` in a fresh arena, materializes its result and returns the traced context of the
/// program that computed it.
fn traced(build: impl FnOnce(&Arc<NamedCpuBackend>) -> anyhow::Result<Dt>) -> ProgramContext {
    let _serial_guard = TRACE_TEST_MUTEX.lock().expect("trace test mutex poisoned");
    let backend = Arc::new(NamedCpuBackend::new("cpu-plan-key-test"));
    let sink = Arc::new(ContextSink {
        contexts: Mutex::new(Vec::new()),
    });
    let _trace_guard = trace::install_global_sink(sink.clone() as Arc<dyn ExecutionTraceSink>);
    with_default_arena(GraphArena::new(Arc::clone(&backend)), || {
        build(&backend)?.to_host()
    })
    .unwrap();
    let contexts = sink.contexts.lock().unwrap();
    contexts.last().expect("a program ran").clone()
}

fn small_graph(backend: &Arc<NamedCpuBackend>, swap: bool) -> Dt {
    let x = input(backend, &[2, 4], DType::F32);
    let y = input(backend, &[2, 4], DType::F32);
    let diff = if swap { &y - &x } else { &x - &y };
    -(diff * x)
}

#[test]
fn value_id_offsets_do_not_change_the_key_but_topology_does() {
    let fresh = traced(|backend| Ok(small_graph(backend, false)));
    let offset = traced(|backend| {
        // Materialized values in front of the graph shift every value id it gets.
        input(backend, &[3], DType::F32).neg().abs().to_host()?;
        Ok(small_graph(backend, false))
    });
    assert_eq!(fresh.plan_hash, offset.plan_hash);
    let swapped = traced(|backend| Ok(small_graph(backend, true)));
    assert_ne!(fresh.plan_hash, swapped.plan_hash);
}

#[test]
fn dtypes_and_parameter_ids_are_part_of_the_key() {
    // Only the input dtype differs: both graphs are one cast to f32.
    let widen =
        |x: fn(&Arc<NamedCpuBackend>) -> Dt| traced(move |backend| Ok(x(backend).cast(DType::F32)));
    let from_bf16 = widen(|backend| input(backend, &[4], DType::Bf16));
    let from_i32 = widen(|backend| {
        let host = HostTensor::si32([4], vec![1, -2, 3, 4]).unwrap();
        DeviceTensor::from_host(Arc::clone(backend), host).unwrap()
    });
    assert_ne!(from_bf16.plan_hash, from_i32.plan_hash);

    let with_param = |stable_id| {
        traced(move |backend| {
            let w = input(backend, &[2, 4], DType::F32).as_param(stable_id)?;
            Ok(input(backend, &[2, 4], DType::F32) + w)
        })
    };
    assert_ne!(with_param(7).plan_hash, with_param(8).plan_hash);
}

#[test]
fn nodes_and_inputs_in_swapped_positions_are_distinguished() {
    let build = |node_first: bool| {
        traced(move |backend| {
            let x = input(backend, &[4], DType::F32);
            let y = input(backend, &[4], DType::F32);
            let node = x.neg();
            Ok(if node_first { node - y } else { y - node })
        })
    };
    assert_ne!(build(true).plan_hash, build(false).plan_hash);
}

#[test]
fn distant_operands_in_sparse_graphs_are_distinguished() {
    let build = |far: usize| {
        traced(move |backend| {
            let x = input(backend, &[4], DType::F32);
            let mut chain = vec![x.neg()];
            for step in 0..20 {
                if step >= 12 {
                    // A pending value outside the result's dependencies leaves a gap in the ids.
                    let _unused = x.abs();
                }
                chain.push(chain.last().unwrap().neg());
            }
            Ok(chain.last().unwrap() - &chain[far])
        })
    };
    assert_ne!(build(10).plan_hash, build(11).plan_hash);
}

#[test]
fn repeated_graph_in_one_arena_reuses_the_program_for_its_own_values() -> anyhow::Result<()> {
    let _serial_guard = TRACE_TEST_MUTEX.lock().expect("trace test mutex poisoned");
    let backend = Arc::new(NamedCpuBackend::new("cpu-plan-rebind-test"));
    let sink = Arc::new(ContextSink {
        contexts: Mutex::new(Vec::new()),
    });
    let _trace_guard = trace::install_global_sink(sink.clone() as Arc<dyn ExecutionTraceSink>);

    with_default_arena(
        GraphArena::new(Arc::clone(&backend)),
        || -> anyhow::Result<()> {
            for (a, b) in [(1.0f32, 2.0f32), (10.0, 20.0)] {
                let x = tensor_from_data_with_shape(&backend, &[2], &[a, b])?;
                let y = &x * &x + &x;
                assert_eq!(y.to_host()?.as_f32()?, [a * a + a, b * b + b]);
            }
            Ok(())
        },
    )?;
    let hits: Vec<bool> = sink
        .contexts
        .lock()
        .unwrap()
        .iter()
        .map(|c| c.program_cache_hit)
        .collect();
    assert_eq!(hits, [false, true]);
    Ok(())
}
