//! Backend-agnostic functional operators built on top of device tensors.
//!
//! Each functional is a [`#[functional]`](crate::functional). It validates its inputs with the
//! [`validate`] macros. It then captures PTIR with [`capture!`](crate::capture) into the lazy graph
//! of its operands and returns lazy device tensors. The tensors carry the backend, so functionals
//! take no backend argument: `functional::gelu(&x)`. The [`DeviceTensorOps`] extension trait
//! offers the elementwise functionals as methods.
//!
//! To replace an op sequence with a faster kernel, a backend rewrites the pattern view that
//! `#[functional]` generates for the functional. An example is `Conv2dPattern` for [`conv2d`].

#![deny(clippy::disallowed_methods, clippy::disallowed_types)]

pub mod activation;
pub mod attention;
#[doc(hidden)]
pub mod capture;
pub mod conv;
pub mod elementwise;
pub mod embedding;
pub mod linalg;
pub mod normalization;
pub mod pooling;
pub mod rotary;
pub mod shape;
pub mod stochastic;
pub mod tensor_ops;
pub mod validate;

pub use activation::*;
pub use attention::*;
pub use conv::*;
pub use elementwise::*;
pub use embedding::*;
pub use linalg::*;
pub use normalization::*;
pub use pooling::*;
pub use rotary::*;
pub use shape::*;
pub use stochastic::*;
pub use tensor_ops::*;
