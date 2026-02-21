use std::sync::Arc;

use gpt_rs::backend::spec::TensorSpec;

/// Placeholder tensor handle for the Triton backend roadmap.
///
/// M1 keeps execution delegated to CPU tensors; this type exists so M2 can
/// establish ownership and metadata patterns before CUDA allocations land.
#[derive(Clone, Debug)]
pub struct TritonTensor {
    pub spec: TensorSpec,
    pub storage: TritonStorage,
}

#[derive(Clone, Debug)]
pub enum TritonStorage {
    /// Host-backed bytes used by bootstrap paths before device allocation is
    /// implemented.
    HostBytes(Arc<[u8]>),

    /// Reserved layout for future device pointers.
    DevicePtr { addr: u64, bytes: usize },
}

impl TritonTensor {
    pub fn host_backed(spec: TensorSpec, bytes: Arc<[u8]>) -> Self {
        Self {
            spec,
            storage: TritonStorage::HostBytes(bytes),
        }
    }
}
