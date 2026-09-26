//! Neural network building blocks layered on top of the portable tensor API.
//!
//! Layers are modules ([`crate::module::Module`]) declared with [`module`]. Each layer:
//! - composes the functionals under `ops::functional`,
//! - names its parameters like the Hugging Face / PyTorch module tree,
//! - is built from checkpoint tensors through [`Params`].
//!
//! Layers implement inference forward passes only.

pub mod capture;
pub mod layers;
pub mod params;

pub use gpt_rs_macros::module;
pub use layers::*;
pub use params::Params;
