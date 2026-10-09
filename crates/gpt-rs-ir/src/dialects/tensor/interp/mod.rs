//! The semantics of the `tensor` ops: one `Eval` impl per op, used by the interpreter.

mod constant;
mod contraction;
mod control;
mod custom_call;
mod elementwise;
mod indexing;
mod kernels;
mod layout;
mod quant;
mod random;
mod window;
