use std::sync::Arc;

use crate::backend::spec::{PortableBackend, TensorSpec};
use crate::ops::graph::GraphArena;
use crate::tensor::{spec_utils, DeviceTensor};

pub fn tensor_spec_from_device<B: PortableBackend + 'static>(
    tensor: &DeviceTensor<B>,
) -> TensorSpec {
    TensorSpec::new(
        spec_utils::backend_dtype(tensor.dtype()),
        spec_utils::backend_shape_from_shape(tensor.shape()),
    )
}

pub fn resolve_graph_from_tensors<B: PortableBackend + 'static>(
    tensors: &[&DeviceTensor<B>],
) -> Option<Arc<GraphArena<B>>> {
    tensors.iter().find_map(|tensor| tensor.graph())
}
