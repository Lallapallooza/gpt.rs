//! The text form of a module.
//!
//! Every op prints in one generic form:
//!
//! ```text
//! %r0, %r1 = dialect.name(%a, %b) {attr = value} ({ ^body(%x: index): ... }) : (types) -> types
//! ```
//!
//! `func.func` alone has its own form: `func.func @name(%0: type {param = 17}) -> types { ... }`.
//! `Module::from_text(&m.to_text())` rebuilds a module that prints the same text.
//!
//! An op may end with its location, which says where it comes from and never what it computes:
//! `loc("model.layers.3.mlp"("crates/gpt-rs/src/nn/layers/mlp.rs":42:9))`, the module path alone
//! (`loc("model.layers.3.mlp")`) or the source alone (`loc("mlp.rs":42:9)`). Dumps print the
//! locations ([`TextOptions`]); `to_text` leaves them out.
//!
//! A program stored for later, such as a dump, starts with the version of the spec it follows,
//! `spec_version = "ptir.v0.5"`
//! ([`Module::to_versioned_text`](crate::module::Module::to_versioned_text)). The parser reads a
//! module with that line or without it, and rejects any other version.

mod cursor;
mod parse;
mod print;

/// The version of the PTIR spec that the text form follows.
pub const SPEC_VERSION: &str = "ptir.v0.5";

pub use self::cursor::Cursor;
pub(crate) use self::cursor::{is_ident_char, is_ident_start};
pub(crate) use self::parse::{named_arg, parse_result_types};
pub(crate) use self::print::results_text;
pub use self::print::TextOptions;
