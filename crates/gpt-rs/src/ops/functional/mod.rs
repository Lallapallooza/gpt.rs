//! Backend-agnostic functional operators built on top of device tensors.
//!
//! Each functional is a [`#[functional]`](crate::functional). It validates its inputs with the
//! [`validate`] macros, then computes with the ops of [`DeviceTensor`](crate::DeviceTensor) and
//! other functionals, which record into the lazy graph of its inputs. The tensors carry the
//! backend, so functionals take no backend argument: `functional::gelu(&x)`.
//!
//! When a backend compiles an op sequence that a functional records, it can match the ops and run
//! them on a faster kernel, as faer and C do for [`conv2d`].

#![deny(clippy::disallowed_methods, clippy::disallowed_types)]

pub mod activation;
pub mod attention;
pub mod conv;
pub mod elementwise;
pub mod embedding;
pub mod linalg;
pub mod linear_attention;
pub mod normalization;
pub mod pooling;
pub mod rotary;
pub mod sampling;
pub mod validate;

pub use activation::*;
pub use attention::*;
pub use conv::*;
pub use elementwise::*;
pub use embedding::*;
pub use linalg::*;
pub use linear_attention::*;
pub use normalization::*;
pub use pooling::*;
pub use rotary::*;
pub use sampling::*;
