//! Neural network building blocks layered on top of the portable tensor API.
//!
//! Layers are thin wrappers that compose the functional graph primitives defined under
//! `ops::functional` and provide inference-focused ergonomics.

pub mod layers;

pub use layers::*;
