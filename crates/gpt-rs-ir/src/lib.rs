//! The gpt.rs IR: one container for a closed set of dialects, with a generated text form, a
//! verifier, an interpreter and a rewrite driver.
//!
//! [`op`] says what an op is. A dialect is an enum of its ops with `#[derive(Dialect)]`, and
//! [`Op`](enum@Op) is the union of the dialects.

extern crate self as gpt_rs_ir;

pub mod attr;
pub mod conversion;
pub mod cost;
pub mod dense;
pub mod dialects;
pub mod doc;
pub mod float8;
pub mod fold;
pub mod host;
pub mod index;
pub mod interp;
pub mod location;
pub mod minifloat;
pub mod module;
pub mod op;
pub mod passes;
pub mod rewrite;
pub mod scalar;
pub mod schedule;
mod scope;
pub mod text;
pub mod traits;
pub mod types;
pub mod verify;

#[doc(hidden)]
pub use gpt_rs_ir_macros::family_op;
pub(crate) use gpt_rs_ir_macros::{family, op, Attr, AttrGroup, Dialect, DialectSet, Doc};
#[doc(hidden)]
pub use linkme;
pub use smallvec;

pub use self::dialects::{Dialect, Op};
