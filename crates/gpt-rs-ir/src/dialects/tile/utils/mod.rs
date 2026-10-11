//! Helpers of the `tile` ops: building blocks, and reading and rewriting their attributes. One file
//! per section of `ir.rs`, and `build.rs` holds the block builders.

mod block;
pub(crate) mod build;
mod combine;
mod dispatch;
