#![doc = include_str!("README.md")]

mod fold;
mod interp;
mod ir;
mod passes;
mod utils;
mod verify;

pub use self::ir::*;
pub use self::passes::Contract;
pub use self::utils::{Affine, AffineForms, IndexRanges};
