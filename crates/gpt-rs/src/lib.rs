extern crate self as gpt_rs;

pub use linkme;

pub mod backend;
pub mod checkpoint;
pub mod inference;
pub mod io;
pub mod layout;
pub mod model;
pub mod module;
pub mod nn;
pub mod ops;
pub mod params;
pub mod tensor;
pub use tensor::DeviceTensor;
mod env;
pub mod profiling;
pub mod runtime;
pub mod tokenizer;
pub mod train;
pub mod vision;

pub use backend::spec::PortableBackend;
pub use tensor::{DType, Shape, Tensor};
