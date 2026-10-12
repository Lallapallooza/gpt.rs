//! Helpers of the `tensor` ops: dtype classes, axis and shape checks, constructors and accessors.
//! One file per section of `ir.rs`.

pub(crate) mod class;
mod constant;
pub(crate) mod contraction;
mod layout;
pub(crate) mod random;
pub(crate) mod shape;
pub(crate) mod window;
