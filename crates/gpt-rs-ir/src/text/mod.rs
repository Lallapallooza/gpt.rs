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
