use std::sync::{Arc, Mutex};

use gpt_rs::backend::spec::{PortableBackend, Program};
use gpt_rs::capture;
use gpt_rs::ops::functional;
use gpt_rs::ops::graph::context::with_default_arena;
use gpt_rs::ops::graph::GraphArena;
use gpt_rs::ops::trace::{self, ExecutionTraceSink, ProgramContext, ProgramStats};
use gpt_rs::tensor::{DType, DeviceTensor, DeviceTensorOps, Shape, Tensor};
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

    fn backend_name(&self) -> &str {
        self.name
    }

    fn materialize(
        &self,
        init: gpt_rs::backend::spec::TensorInit,
    ) -> gpt_rs::backend::spec::BackendResult<Self::TensorHandle> {
        self.inner.materialize(init)
    }

    fn to_literal(
        &self,
        tensor: &Self::TensorHandle,
    ) -> gpt_rs::backend::spec::BackendResult<gpt_rs::backend::spec::TensorLiteral> {
        self.inner.to_literal(tensor)
    }

    fn execute_instruction(
        &self,
        instruction: &gpt_rs::backend::spec::Instruction,
        inputs: &[Self::TensorHandle],
    ) -> gpt_rs::backend::spec::BackendResult<Vec<Self::TensorHandle>> {
        self.inner.execute_instruction(instruction, inputs)
    }

    fn run_program(
        &self,
        program: &Program,
        entry_inputs: &[Self::TensorHandle],
    ) -> gpt_rs::backend::spec::BackendResult<Vec<Self::TensorHandle>> {
        self.inner.run_program(program, entry_inputs)
    }
}

struct ContextSink {
    contexts: Mutex<Vec<ProgramContext>>,
}

impl ExecutionTraceSink for ContextSink {
    fn before_program(&self, context: &ProgramContext, _program: &Program) {
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
    let host = Tensor::from_vec(Shape::new(shape.to_vec()), data.to_vec())?;
    DeviceTensor::from_host(Arc::clone(backend), host)
}

fn tensor_from_data(
    backend: &Arc<NamedCpuBackend>,
    data: &[f32],
) -> anyhow::Result<DeviceTensor<NamedCpuBackend>> {
    tensor_from_data_with_shape(backend, &[2, 2], data)
}

fn add_scalar_literal(
    input: &DeviceTensor<NamedCpuBackend>,
    scalar: f32,
) -> anyhow::Result<DeviceTensor<NamedCpuBackend>> {
    capture!(session, |input| {
        let scalar_tensor = session.scalar(scalar).broadcast_like(&input);
        input + scalar_tensor
    })
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
    let out1 = add_scalar_literal(&input1, 1.0)?.to_host()?;
    assert_eq!(out1.data(), &[2.0, 3.0, 4.0, 5.0]);

    let input2 = tensor_from_data(&backend, &[1.0, 2.0, 3.0, 4.0])?;
    let out2 = add_scalar_literal(&input2, 2.0)?.to_host()?;
    assert_eq!(out2.data(), &[3.0, 4.0, 5.0, 6.0]);

    let contexts = sink
        .contexts
        .lock()
        .expect("trace context sink mutex poisoned")
        .clone();
    let (first, second) = contexts_after_two_runs(&contexts);
    assert!(
        !first.cache.program_cache_hit,
        "first execution unexpectedly hit program cache"
    );
    assert!(
        !second.cache.program_cache_hit,
        "program cache was reused across different literal values"
    );
    assert_eq!(
        first.plan_graph_hash, second.plan_graph_hash,
        "stable graph hash should match across literal-only changes"
    );
    assert_ne!(
        first.plan_specialization_hash, second.plan_specialization_hash,
        "specialization hash should differ across literal-only changes"
    );

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
    let _ = add_scalar_literal(&input1, 1.5)?.to_host()?;

    let input2 = tensor_from_data(&backend, &[1.0, 2.0, 3.0, 4.0])?;
    let _ = add_scalar_literal(&input2, 1.5)?.to_host()?;

    let contexts = sink
        .contexts
        .lock()
        .expect("trace context sink mutex poisoned")
        .clone();
    let (first, second) = contexts_after_two_runs(&contexts);
    assert!(
        !first.cache.program_cache_hit,
        "first execution unexpectedly hit program cache"
    );
    assert!(
        second.cache.program_cache_hit,
        "expected second identical graph to hit program cache"
    );
    assert_eq!(
        first.plan_graph_hash, second.plan_graph_hash,
        "stable graph hash should match identical graph executions"
    );
    assert_eq!(
        first.plan_specialization_hash, second.plan_specialization_hash,
        "specialization hash should match identical graph executions"
    );

    Ok(())
}

#[test]
fn shape_change_keeps_graph_hash_but_changes_specialization() -> anyhow::Result<()> {
    let _serial_guard = TRACE_TEST_MUTEX.lock().expect("trace test mutex poisoned");
    let backend = Arc::new(NamedCpuBackend::new("cpu-shape-specialization-cache-test"));
    let sink = Arc::new(ContextSink {
        contexts: Mutex::new(Vec::new()),
    });
    let _trace_guard = trace::install_global_sink(sink.clone() as Arc<dyn ExecutionTraceSink>);

    let input1 = tensor_from_data_with_shape(&backend, &[2, 2], &[1.0, 2.0, 3.0, 4.0])?;
    let _ = add_scalar_literal(&input1, 1.0)?.to_host()?;

    let input2 = tensor_from_data_with_shape(
        &backend,
        &[4, 4],
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ],
    )?;
    let _ = add_scalar_literal(&input2, 1.0)?.to_host()?;

    let contexts = sink
        .contexts
        .lock()
        .expect("trace context sink mutex poisoned")
        .clone();
    let (first, second) = contexts_after_two_runs(&contexts);
    assert!(
        !second.cache.program_cache_hit,
        "shape specialization change should not reuse cached program"
    );
    assert_eq!(
        first.plan_graph_hash, second.plan_graph_hash,
        "stable graph hash should ignore shape specialization changes"
    );
    assert_ne!(
        first.plan_specialization_hash, second.plan_specialization_hash,
        "specialization hash should change with shape specialization"
    );

    Ok(())
}

// `plan_graph_hash` covers the graph structure. `plan_hash` also covers every specialization:
// shapes, dtypes, literals and bound parameters.

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

fn small_graph(backend: &Arc<NamedCpuBackend>, swap: bool) -> anyhow::Result<Dt> {
    let x = input(backend, &[2, 4], DType::F32);
    let y = input(backend, &[2, 4], DType::F32);
    let diff = if swap { y.sub(&x)? } else { x.sub(&y)? };
    diff.mul(&x)?.neg()
}

#[test]
fn value_id_offsets_do_not_change_the_key_but_topology_does() {
    let fresh = traced(|backend| small_graph(backend, false));
    let offset = traced(|backend| {
        // Materialized values in front of the graph shift every value id it gets.
        input(backend, &[3], DType::F32).neg()?.abs()?.to_host()?;
        small_graph(backend, false)
    });
    assert_eq!(fresh.plan_hash, offset.plan_hash);
    let swapped = traced(|backend| small_graph(backend, true));
    assert_ne!(fresh.plan_graph_hash, swapped.plan_graph_hash);
}

#[test]
fn dtypes_and_parameter_ids_are_part_of_the_key() {
    // Only the input dtype differs: both graphs are one cast to f32.
    let widen = |x: fn(&Arc<NamedCpuBackend>) -> Dt| {
        traced(move |backend| functional::cast(&x(backend), DType::F32))
    };
    let from_bf16 = widen(|backend| input(backend, &[4], DType::BF16));
    let from_i32 = widen(|backend| {
        let host = Tensor::from_i32(Shape::new([4]), vec![1, -2, 3, 4]).unwrap();
        DeviceTensor::from_host(Arc::clone(backend), host).unwrap()
    });
    assert_eq!(from_bf16.plan_graph_hash, from_i32.plan_graph_hash);
    assert_ne!(from_bf16.plan_hash, from_i32.plan_hash);

    let with_param = |stable_id| {
        traced(move |backend| {
            let w = input(backend, &[2, 4], DType::F32).as_param_with_id(stable_id)?;
            input(backend, &[2, 4], DType::F32).add(&w)
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
            let node = x.neg()?;
            if node_first {
                node.sub(&y)
            } else {
                y.sub(&node)
            }
        })
    };
    assert_ne!(build(true).plan_graph_hash, build(false).plan_graph_hash);
}

#[test]
fn distant_operands_in_sparse_graphs_are_distinguished() {
    let build = |far: usize| {
        traced(move |backend| {
            let x = input(backend, &[4], DType::F32);
            let mut chain = vec![x.neg()?];
            for step in 0..20 {
                if step >= 12 {
                    // A pending value outside the result's dependencies leaves a gap in the ids.
                    let _unused = x.abs()?;
                }
                chain.push(chain.last().unwrap().neg()?);
            }
            chain.last().unwrap().sub(&chain[far])
        })
    };
    assert_ne!(build(10).plan_graph_hash, build(11).plan_graph_hash);
}

#[test]
fn repeated_graph_in_one_arena_reuses_the_plan_for_its_own_values() -> anyhow::Result<()> {
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
                let y = x.mul(&x)?.add(&x)?;
                assert_eq!(y.to_host()?.data(), [a * a + a, b * b + b]);
            }
            Ok(())
        },
    )?;
    let hits: Vec<bool> = sink
        .contexts
        .lock()
        .unwrap()
        .iter()
        .map(|c| c.cache.plan_cache_hit)
        .collect();
    assert_eq!(hits, [false, true]);
    Ok(())
}
