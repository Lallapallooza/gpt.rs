//! The costs of the `tensor` ops that differ from the default, which `#[cost]` registers. One file
//! per section of `ir.rs`.

mod contraction;
mod indexing;
