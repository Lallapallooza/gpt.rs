//! The type rules of the `tensor` ops: one `OpDef` impl per op, or one rule per family of ops.

mod constant;
pub(crate) mod contraction;
mod control;
mod custom_call;
pub(crate) mod elementwise;
pub(crate) mod indexing;
mod layout;
pub(crate) mod quant;
pub(crate) mod random;
pub(crate) mod window;
