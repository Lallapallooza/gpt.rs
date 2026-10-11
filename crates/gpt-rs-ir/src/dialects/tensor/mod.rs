#![doc = include_str!("README.md")]

mod cost;
mod fold;
pub(crate) mod interp;
pub(crate) mod ir;
mod passes;
pub(crate) mod utils;
mod verify;

pub use self::ir::*;
pub use self::passes::{DotScaleFold, MergeContractedAxes};
pub use self::utils::class::Class;
pub use self::utils::contraction::{Accumulates, DotAxes, Side};
pub use self::utils::random::state_type;
pub use self::utils::shape::axis_index;
