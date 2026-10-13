#![doc = include_str!("README.md")]

mod interp;
mod ir;
mod passes;
mod utils;
mod verify;

pub use self::interp::Acc;
pub use self::ir::*;
pub use self::passes::SealDispatches;
pub use self::utils::build::{par_iters, push_iter, Body, Inputs};
pub use self::utils::Step;
