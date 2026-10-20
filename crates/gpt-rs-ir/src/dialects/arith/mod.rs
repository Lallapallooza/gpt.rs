#![doc = include_str!("README.md")]

mod interp;
mod ir;
mod passes;
mod utils;
mod verify;

pub use self::interp::math;
pub use self::ir::*;
pub use self::passes::{ExpandMath, ExpandRandom};
pub use self::utils::ScalarBuilder;
pub(crate) use self::utils::{ensure_bitcast, ensure_fma, lone_product};
