//! Helpers shared by the tests that run functionals.

use std::sync::Arc;

use gpt_rs::backend::PortableBackend;
use gpt_rs::tensor::{DeviceTensor, HostTensor};

/// An f32 device tensor of `shape` with every element `value`.
pub fn filled<B: PortableBackend + 'static>(
    backend: &Arc<B>,
    shape: &[usize],
    value: f32,
) -> DeviceTensor<B> {
    let host = HostTensor::f32(shape, vec![value; shape.iter().product()]).unwrap();
    DeviceTensor::from_host(Arc::clone(backend), host).unwrap()
}
