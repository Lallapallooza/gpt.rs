#![doc = include_str!("README.md")]

mod interp;
mod ir;
mod utils;
mod verify;

pub use self::ir::*;
pub use self::utils::{
    in_registers, is_lanewise, lanewise, repeated_position, row_major, vector_runs, Access, Needs,
    Pressure, Register, RegisterCode, Rules,
};
