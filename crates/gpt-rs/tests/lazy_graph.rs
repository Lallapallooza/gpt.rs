use std::sync::Arc;

use anyhow::Result;
use gpt_rs::tensor::{DeviceTensor, HostTensor};
use gpt_rs_backend_ref_cpu::CpuPortableBackend;
use gpt_rs_backend_tests::recording_backend::RecordingBackend;

type Backend = RecordingBackend<CpuPortableBackend>;

fn tensor_from_data(backend: &Arc<Backend>, data: &[f32]) -> Result<DeviceTensor<Backend>> {
    let host = HostTensor::f32([2, 2], data.to_vec())?;
    DeviceTensor::from_host(Arc::clone(backend), host)
}

#[test]
fn incremental_flush_skips_executed_nodes() -> Result<()> {
    let backend = Arc::new(RecordingBackend::new(CpuPortableBackend::new()));

    let lhs = tensor_from_data(&backend, &[1.0, 2.0, 3.0, 4.0])?;
    let rhs = tensor_from_data(&backend, &[5.0, 6.0, 7.0, 8.0])?;

    let sum = &lhs + &rhs; // lazy node
    assert_eq!(backend.runs().len(), 0);

    let _ = sum.materialize()?;
    assert_eq!(backend.runs().len(), 1);

    let shifted = &sum + &lhs; // reuses graph
    let product = shifted * &rhs; // another node
    assert_eq!(backend.runs().len(), 1);

    let _ = product.materialize()?; // should execute only new nodes
    assert_eq!(backend.runs().len(), 2);

    let _ = product.materialize()?; // already materialized, no new execution
    assert_eq!(backend.runs().len(), 2);

    Ok(())
}
