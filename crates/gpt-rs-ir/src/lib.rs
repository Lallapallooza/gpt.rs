//! The gpt.rs IR: one container for a closed set of dialects, with a generated text form, a
//! verifier, an interpreter and a rewrite driver.
//!
//! - [`ir`]: the core every dialect is made of: the module, what an op is, types, attributes, the
//!   index language, locations and the verifier.
//! - [`text`]: the text form.
//! - [`host`]: data in host memory.
//! - [`interp`]: the interpreter.
//! - [`rewrite`]: building and rewriting IR.
//! - [`passes`]: the passes over any dialect.
//! - [`analysis`]: what ops cost.
//! - [`traits`]: op traits.
//! - [`dialects`]: the dialects, each with its own passes, and [`Op`](enum@Op), their union.
//! - [`conversion`]: the conversions between dialects.

extern crate self as gpt_rs_ir;

pub mod analysis;
pub mod conversion;
pub mod dialects;
pub mod host;
pub mod interp;
pub mod ir;
pub mod passes;
pub mod rewrite;
pub mod text;
pub mod traits;

#[doc(hidden)]
pub use gpt_rs_ir_macros::family_op;
pub(crate) use gpt_rs_ir_macros::{family, op, Attr, AttrGroup, Dialect, DialectSet, Doc};
#[doc(hidden)]
pub use linkme;
pub use smallvec;

pub use self::dialects::{Dialect, Op};
