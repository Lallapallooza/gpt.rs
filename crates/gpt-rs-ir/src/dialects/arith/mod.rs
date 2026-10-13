#![doc = include_str!("README.md")]

mod interp;
mod ir;
mod utils;
mod verify;

pub use self::interp::math;
pub use self::ir::*;
