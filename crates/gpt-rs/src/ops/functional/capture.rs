//! Runtime support of [`capture!`](crate::capture): the graph a capture records into, and the
//! conversion of the PTIR tensors its body returns into lazy device tensors.

use std::sync::Arc;

use anyhow::{anyhow, Result};

use crate::backend::spec::{PortableBackend, TensorSpec, ValueId};
use crate::ops::graph::{context, GraphArena};
use crate::ops::ptir;
use crate::tensor::spec_utils::{self, frontend_dtype, shape_from_spec};
use crate::tensor::DeviceTensor;

/// Returns the graph that a capture over `operands` records into. This is the graph of the first
/// operand that has one, else the current default arena, else a new graph.
pub fn arena<B: PortableBackend + 'static>(operands: &[&DeviceTensor<B>]) -> Arc<GraphArena<B>> {
    operands
        .iter()
        .find_map(|tensor| tensor.graph())
        .or_else(context::current_arena)
        .unwrap_or_else(|| GraphArena::new(operands[0].backend()))
}

/// The PTIR tensor spec of a device tensor.
pub fn tensor_spec<B: PortableBackend + 'static>(tensor: &DeviceTensor<B>) -> TensorSpec {
    TensorSpec::new(
        spec_utils::backend_dtype(tensor.dtype()),
        spec_utils::backend_shape_from_shape(tensor.shape()),
    )
}

/// What a capture body returns: a PTIR tensor or a tuple of them.
pub trait CaptureOutput {
    type Ids: CapturedIds;
    fn into_ids(self) -> Self::Ids;
}

/// The value ids of the capture outputs. They become lazy device tensors after the capture is
/// recorded.
pub trait CapturedIds: Sized {
    type Tensors<B: PortableBackend + 'static>;
    type Values: AsRef<[ValueId]>;
    fn values(&self) -> Self::Values;
    fn into_device_tensors<B: PortableBackend + 'static>(
        self,
        graph: &Arc<GraphArena<B>>,
    ) -> Result<Self::Tensors<B>>;
}

fn device_tensor<B: PortableBackend + 'static>(
    graph: &Arc<GraphArena<B>>,
    value: ValueId,
) -> Result<DeviceTensor<B>> {
    let spec = graph
        .tensor_spec_for(value)
        .ok_or_else(|| anyhow!("value {value:?} has no tensor spec"))?;
    let dtype = frontend_dtype(spec.dtype)?;
    let shape = shape_from_spec(&spec)?;
    DeviceTensor::from_lazy(Arc::clone(graph), shape, dtype, value)
}

impl<B: PortableBackend + 'static> CaptureOutput for ptir::Tensor<'_, '_, B> {
    type Ids = ValueId;
    fn into_ids(self) -> ValueId {
        self.id()
    }
}

impl CapturedIds for ValueId {
    type Tensors<B: PortableBackend + 'static> = DeviceTensor<B>;
    type Values = [ValueId; 1];
    fn values(&self) -> [ValueId; 1] {
        [*self]
    }
    fn into_device_tensors<B: PortableBackend + 'static>(
        self,
        graph: &Arc<GraphArena<B>>,
    ) -> Result<DeviceTensor<B>> {
        device_tensor(graph, self)
    }
}

macro_rules! tuple_outputs {
    ($($name:ident),+) => {
        impl<'ctx, 'gb, B: PortableBackend + 'static> CaptureOutput
            for ($(tuple_outputs!(@tensor $name, 'ctx, 'gb, B),)+)
        {
            type Ids = ($(tuple_outputs!(@id $name),)+);
            fn into_ids(self) -> Self::Ids {
                let ($($name,)+) = self;
                ($($name.id(),)+)
            }
        }

        impl CapturedIds for ($(tuple_outputs!(@id $name),)+) {
            type Tensors<B: PortableBackend + 'static> = ($(tuple_outputs!(@device $name, B),)+);
            type Values = [ValueId; tuple_outputs!(@count $($name)+)];
            fn values(&self) -> Self::Values {
                let ($($name,)+) = *self;
                [$($name),+]
            }
            fn into_device_tensors<B: PortableBackend + 'static>(
                self,
                graph: &Arc<GraphArena<B>>,
            ) -> Result<Self::Tensors<B>> {
                let ($($name,)+) = self;
                Ok(($(device_tensor(graph, $name)?,)+))
            }
        }
    };
    (@tensor $name:ident, $ctx:lifetime, $gb:lifetime, $b:ident) => { ptir::Tensor<$ctx, $gb, $b> };
    (@id $name:ident) => { ValueId };
    (@device $name:ident, $b:ident) => { DeviceTensor<$b> };
    (@count $($name:ident)+) => { 0 $(+ tuple_outputs!(@one $name))+ };
    (@one $name:ident) => { 1 };
}

tuple_outputs!(a, b);
tuple_outputs!(a, b, c);
tuple_outputs!(a, b, c, d);
