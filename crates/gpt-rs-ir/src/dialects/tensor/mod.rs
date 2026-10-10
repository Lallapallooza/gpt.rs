#![doc = include_str!("README.md")]

mod interp;
mod ir;
pub(crate) mod passes;

pub use self::ir::*;
