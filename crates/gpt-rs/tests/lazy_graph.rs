use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use anyhow::Result;
use gpt_rs::backend::{ParamStore, PortableBackend};
use gpt_rs::gpt_rs_ir::ir::Module;
use gpt_rs::tensor::{DeviceTensor, HostTensor};
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
    type Executable = <CpuPortableBackend as PortableBackend>::Executable;

    fn backend_name(&self) -> &str {
        "cpu-counting"
    }

    fn params(&self) -> &ParamStore<Self::TensorHandle> {
        self.inner.params()
    }

    fn compile(&self, module: &Arc<Module>, func: &str) -> Result<Self::Executable> {
        self.inner.compile(module, func)
    }

    fn run(
        &self,
        exe: &Self::Executable,
        args: &[Self::TensorHandle],
    ) -> Result<Vec<Self::TensorHandle>> {
        self.runs.fetch_add(1, Ordering::SeqCst);
        self.inner.run(exe, args)
    }

    fn upload(&self, host: HostTensor) -> Result<Self::TensorHandle> {
        self.inner.upload(host)
    }

    fn download(&self, tensor: &Self::TensorHandle) -> Result<HostTensor> {
        self.inner.download(tensor)
    }
}

fn tensor_from_data(
    backend: &Arc<CountingBackend>,
    data: &[f32],
) -> Result<DeviceTensor<CountingBackend>> {
    let host = HostTensor::f32([2, 2], data.to_vec())?;
    DeviceTensor::from_host(Arc::clone(backend), host)
}

#[test]
fn incremental_flush_skips_executed_nodes() -> Result<()> {
    let backend = Arc::new(CountingBackend::new());

    let a = tensor_from_data(&backend, &[1.0, 2.0, 3.0, 4.0])?;
    let b = tensor_from_data(&backend, &[5.0, 6.0, 7.0, 8.0])?;

    let c = &a + &b; // lazy node
    assert_eq!(backend.calls(), 0);

    let _ = c.materialize()?;
    assert_eq!(backend.calls(), 1);

    let d = &c + &a; // reuses graph
    let e = d * &b; // another node
    assert_eq!(backend.calls(), 1);

    let _ = e.materialize()?; // should execute only new nodes
    assert_eq!(backend.calls(), 2);

    let _ = e.materialize()?; // already materialized, no new execution
    assert_eq!(backend.calls(), 2);

    Ok(())
}
