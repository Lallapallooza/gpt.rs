//! Core tensor abstractions shared across backends: the lazy [`DeviceTensor`] and its ops, the
//! host tensor, shapes and dtypes.

mod device_tensor;
mod lazy_tensor;
mod ops;
pub mod shape;
mod slicing;

pub use gpt_rs_ir::host::HostTensor;
pub use gpt_rs_ir::ir::DType;

pub use self::device_tensor::DeviceTensor;
pub use self::lazy_tensor::InputRole;
pub(crate) use self::lazy_tensor::{LazyHandle, Materialized};
pub use self::ops::Operand;
pub use self::shape::Shape;
pub use self::slicing::{AxisIndex, Position};
