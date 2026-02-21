use std::sync::Arc;

use gpt_rs::backend::spec::TensorSpec;

use crate::device::DeviceBuffer;

/// GPU-resident tensor handle for the Triton backend.
#[derive(Clone, Debug)]
pub struct TritonTensor {
    pub spec: TensorSpec,
    pub buffer: Arc<DeviceBuffer>,
}

impl TritonTensor {
    pub fn new(spec: TensorSpec, buffer: Arc<DeviceBuffer>) -> Self {
        Self { spec, buffer }
    }
}
