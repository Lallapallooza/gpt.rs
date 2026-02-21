use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use anyhow::Result;
use gpt_rs::backend::spec::{PortableBackend, TensorInit};
use gpt_rs::params::{BaseParamId, ParamSource};
use gpt_rs::tensor::{DType, DeviceTensor, DeviceTensorOps, Shape, Tensor};
use gpt_rs_backend_ref_cpu::CpuPortableBackend;

struct CountingParamSource {
    backend: Arc<CpuPortableBackend>,
    host: Tensor,
    loads: AtomicUsize,
}

impl CountingParamSource {
    fn new(backend: Arc<CpuPortableBackend>, host: Tensor) -> Self {
        Self {
            backend,
            host,
            loads: AtomicUsize::new(0),
        }
    }

    fn loads(&self) -> usize {
        self.loads.load(Ordering::SeqCst)
    }
}

impl ParamSource<CpuPortableBackend> for CountingParamSource {
    fn load(
        &self,
        _base_id: BaseParamId,
    ) -> Result<<CpuPortableBackend as PortableBackend>::TensorHandle> {
        self.loads.fetch_add(1, Ordering::SeqCst);
        let literal = self.host.to_literal();
        Ok(self.backend.materialize(TensorInit::Literal(literal))?)
    }
}

fn tensor_from_data(
    backend: &Arc<CpuPortableBackend>,
    dims: [usize; 2],
    data: &[f32],
) -> Result<DeviceTensor<CpuPortableBackend>> {
    let host = Tensor::from_vec(Shape::new(dims), data.to_vec())?;
    DeviceTensor::from_host(Arc::clone(backend), host)
}

fn streamed_param(
    backend: &Arc<CpuPortableBackend>,
    source: Arc<CountingParamSource>,
) -> DeviceTensor<CpuPortableBackend> {
    DeviceTensor::lazy_param(
        Arc::clone(backend),
        Shape::new([2, 2]),
        DType::F32,
        0xABCD,
        BaseParamId(0x1234),
        source,
    )
}

#[test]
fn streamed_param_without_cache_loads_from_source_every_time() -> Result<()> {
    let backend = Arc::new(CpuPortableBackend::new());
    let source = Arc::new(CountingParamSource::new(
        Arc::clone(&backend),
        Tensor::from_vec(Shape::new([2, 2]), vec![1.0, 2.0, 3.0, 4.0])?,
    ));

    let param = streamed_param(&backend, Arc::clone(&source));
    assert_eq!(source.loads(), 0);

    let _ = param.materialize()?;
    assert_eq!(source.loads(), 1);

    let _ = param.materialize()?;
    assert_eq!(source.loads(), 2);

    Ok(())
}

#[test]
fn compiled_graph_reloads_streamed_param_for_new_runtime_inputs() -> Result<()> {
    let backend = Arc::new(CpuPortableBackend::new());
    let source = Arc::new(CountingParamSource::new(
        Arc::clone(&backend),
        Tensor::from_vec(Shape::new([2, 2]), vec![10.0, 20.0, 30.0, 40.0])?,
    ));

    let input = tensor_from_data(&backend, [2, 2], &[1.0, 2.0, 3.0, 4.0])?;
    let param = streamed_param(&backend, Arc::clone(&source));
    let out = input.add(&param)?;
    let compiled = out.compile()?;

    assert_eq!(source.loads(), 0);

    let _ = compiled.execute()?;
    assert_eq!(source.loads(), 1);

    let _ = compiled.execute()?;
    assert_eq!(
        source.loads(),
        1,
        "cached graph outputs should not trigger extra param loads"
    );

    let input2 = tensor_from_data(&backend, [2, 2], &[4.0, 3.0, 2.0, 1.0])?;
    let outputs = compiled.execute_with_inputs(&[&input2])?;
    assert_eq!(outputs.len(), 1);
    assert_eq!(source.loads(), 2);

    Ok(())
}
