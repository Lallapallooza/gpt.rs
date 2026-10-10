extern crate self as gpt_rs;

pub use gpt_rs_ir;
pub use gpt_rs_macros::functional;

pub mod backend;
pub mod checkpoint;
pub mod inference;
pub mod io;
pub mod model;
pub mod module;
pub mod nn;
pub mod ops;
pub mod params;
pub mod tensor;
pub use self::tensor::DeviceTensor;
mod env;
pub mod profiling;
pub mod runtime;
pub mod tokenizer;

pub use self::backend::PortableBackend;
pub use self::tensor::{DType, HostTensor, Shape};
