//! Core tensor abstractions shared across backends.
//!
//! The tensor module defines portable shapes, dtypes, storage element traits, and the
//! device-aware tensor wrappers that power functional kernels. It re-exports
//! [`DeviceTensorOps`] so helper traits live next to
//! the tensor types consumers manipulate in forward passes.

mod device_tensor;
pub mod dtype;
mod host_tensor;
mod lazy_tensor;
pub mod shape;
pub(crate) mod spec_utils;
pub mod storage;

pub use crate::ops::functional::DeviceTensorOps;
pub use device_tensor::{DeviceTensor, IntoDeviceTensor, IntoDeviceTensorOption};
pub use dtype::DType;
pub use host_tensor::Tensor;
pub use lazy_tensor::InputRole;
pub(crate) use lazy_tensor::LazyHandle;
pub use shape::Shape;
