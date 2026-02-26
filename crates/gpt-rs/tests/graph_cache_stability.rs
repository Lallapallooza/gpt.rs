use std::sync::{Arc, Mutex};

use gpt_rs::backend::spec::{PortableBackend, Program};
use gpt_rs::ops::functional::CaptureIntoDeviceTensor;
use gpt_rs::ops::trace::{self, ExecutionTraceSink, ProgramContext, ProgramStats};
use gpt_rs::tensor::{DeviceTensor, Shape, Tensor};
use gpt_rs_backend_ref_cpu::CpuPortableBackend;
use gpt_rs_macros::capture_ptir;

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
    capture_ptir!({ x = input }, |session| {
        let scalar_tensor = session.scalar(scalar).broadcast_like(&x);
        Ok((x + scalar_tensor).id())
    })?
    .into_device_tensor()
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
