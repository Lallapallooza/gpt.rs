//! Core tensor abstractions shared across backends: the lazy [`DeviceTensor`] and its ops, the
//! host tensor, shapes and dtypes.

mod device_tensor;
mod lazy_tensor;
mod ops;
pub mod shape;
mod slicing;

pub use device_tensor::DeviceTensor;
pub use gpt_rs_ir::host::HostTensor;
pub use gpt_rs_ir::types::DType;
pub use lazy_tensor::InputRole;
pub(crate) use lazy_tensor::{LazyHandle, Materialized};
pub use ops::Operand;
pub use shape::Shape;
pub use slicing::{AxisIndex, Position};
