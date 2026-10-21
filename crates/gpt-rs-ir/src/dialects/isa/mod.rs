#![doc = include_str!("README.md")]

mod interp;
mod ir;
mod utils;
mod verify;

pub use self::ir::*;
pub use self::utils::{
    in_registers, is_lanewise, lanewise, reg_lanes, reg_registers, repeated_position, row_major,
    tile_vectors, vector_runs, Access, Needs, Pressure, Register, RegisterCode, Rules,
};
