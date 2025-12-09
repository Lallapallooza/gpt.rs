use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use anyhow::Result;
use gpt_rs::backend::spec::{PortableBackend, Program};
use gpt_rs::tensor::{DeviceTensor, DeviceTensorOps, Shape, Tensor};
use gpt_rs_backend_ref_cpu::CpuPortableBackend;

struct CountingBackend {
    inner: Arc<CpuPortableBackend>,
    runs: AtomicUsize,
}

impl CountingBackend {
    fn new() -> Self {
        CountingBackend {
            inner: Arc::new(CpuPortableBackend::new()),
            runs: AtomicUsize::new(0),
        }
    }

    fn calls(&self) -> usize {
        self.runs.load(Ordering::SeqCst)
    }
}

type CpuHandle = <CpuPortableBackend as PortableBackend>::TensorHandle;

type BackendHandle = CpuHandle;

impl PortableBackend for CountingBackend {
    type TensorHandle = BackendHandle;

    fn backend_name(&self) -> &str {
        "cpu-counting"
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
        self.runs.fetch_add(1, Ordering::SeqCst);
        self.inner.run_program(program, entry_inputs)
    }
}

fn tensor_from_data(
    backend: &Arc<CountingBackend>,
    data: &[f32],
) -> Result<DeviceTensor<CountingBackend>> {
    let host = Tensor::from_vec(Shape::new([2, 2]), data.to_vec())?;
    DeviceTensor::from_host(Arc::clone(backend), host)
}

#[test]
fn incremental_flush_skips_executed_nodes() -> Result<()> {
    let backend = Arc::new(CountingBackend::new());

    let a = tensor_from_data(&backend, &[1.0, 2.0, 3.0, 4.0])?;
    let b = tensor_from_data(&backend, &[5.0, 6.0, 7.0, 8.0])?;

    let c = a.add(&b)?; // lazy node
    assert_eq!(backend.calls(), 0);

    let _ = c.materialize()?;
    assert_eq!(backend.calls(), 1);

    let d = c.add(&a)?; // reuses graph
    let e = d.mul(&b)?; // another node
    assert_eq!(backend.calls(), 1);

    let _ = e.materialize()?; // should execute only new nodes
    assert_eq!(backend.calls(), 2);

    let _ = e.materialize()?; // already materialized, no new execution
    assert_eq!(backend.calls(), 2);

    Ok(())
}

#[test]
fn compiled_graph_executes_without_retracing() -> Result<()> {
    let backend = Arc::new(CountingBackend::new());

    let a = tensor_from_data(&backend, &[1.0, 2.0, 3.0, 4.0])?;
    let b = tensor_from_data(&backend, &[5.0, 6.0, 7.0, 8.0])?;

    let c = a.add(&b)?;
    let compiled = c.compile()?;
    assert_eq!(backend.calls(), 0);

    let first = compiled.execute()?;
    assert_eq!(backend.calls(), 1);
    assert_eq!(first.len(), 1);

    let second = compiled.execute()?;
    assert_eq!(backend.calls(), 1, "execution should reuse cached handles");
    assert_eq!(second.len(), 1);

    let a2 = tensor_from_data(&backend, &[2.0, 4.0, 6.0, 8.0])?;
    let b2 = tensor_from_data(&backend, &[1.0, 1.0, 1.0, 1.0])?;

    let third = compiled.execute_with_inputs(&[&a2, &b2])?;
    assert_eq!(backend.calls(), 2, "new inputs should trigger execution");
    assert_eq!(third.len(), 1);

    let literal = backend.inner.to_literal(&third[0])?;
    let tensor = Tensor::from_literal(&literal)?;
    assert_eq!(tensor.data(), &[3.0, 5.0, 7.0, 9.0]);

    Ok(())
}

#[test]
fn compiled_graph_detects_stale_plan() -> Result<()> {
    let backend = Arc::new(CountingBackend::new());

    let a = tensor_from_data(&backend, &[1.0, 2.0, 3.0, 4.0])?;
    let b = tensor_from_data(&backend, &[5.0, 6.0, 7.0, 8.0])?;

    let c = a.add(&b)?;
    let compiled = c.compile()?;

    // Mutate the graph after compilation.
    let _ = c.mul(&a)?;

    let err = compiled
        .execute()
        .err()
        .expect("stale compiled plan should report failure");
    assert!(err.to_string().contains("stale"));

    Ok(())
}
