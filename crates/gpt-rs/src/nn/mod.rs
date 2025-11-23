//! Neural network building blocks layered on top of the portable tensor API.
//!
//! This module exposes activation helpers together with higher level GPT-style layers
//! (attention, feed-forward, embedding, layer norm, linear). Layers are thin wrappers that
//! compose the functional graph primitives defined under `ops::functional` and return
//! convenient state structs for caching intermediates across forward/backward passes.

pub mod activations;
pub mod layers;

pub use layers::*;
