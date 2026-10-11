//! The semantics of the `tensor` ops: one `Eval` impl per op, used by the interpreter.

mod constant;
pub(crate) mod contraction;
mod control;
mod custom_call;
pub(crate) mod elementwise;
pub(crate) mod indexing;
pub(crate) mod kernels;
mod layout;
pub(crate) mod quant;
pub(crate) mod random;
pub(crate) mod window;
