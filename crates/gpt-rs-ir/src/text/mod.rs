//! The text form of a module.
//!
//! Every op prints in one generic form:
//!
//! ```text
//! %r0, %r1 = dialect.name(%a, %b) {attr = value} ({ ^body(%x: index): ... }) : (types) -> types
//! ```
//!
//! A symbol op, such as `func.func`, prints as its signature and its body:
//! `func.func @name(%0: type {param = 17}) -> types attributes {key = value} { ... }`.
//! `Module::from_text(&m.to_text())` rebuilds a module that prints the same text.
//!
//! The parser reads the text as the tokens of `gpt_rs_ir_lexer`. Each attribute reads its own
//! text form ([`crate::attr`]), which `#[derive(Attr)]` declares.
//!
//! An op may end with its location, which says where it comes from and never what it computes:
//! `loc("model.layers.3.mlp"("crates/gpt-rs/src/nn/layers/mlp.rs":42:9))`, the module path alone
//! (`loc("model.layers.3.mlp")`) or the source alone (`loc("mlp.rs":42:9)`). Dumps print the
//! locations ([`TextOptions`]); `to_text` leaves them out.
//!
//! A program stored for later, such as a dump, starts with the version of the spec it follows,
//! `spec_version = "ptir.v0.6"`
//! ([`Module::to_versioned_text`](crate::module::Module::to_versioned_text)). The parser reads a
//! module with that line or without it, and rejects any other version.

mod cursor;
mod parse;
mod print;

/// The version of the PTIR spec that the text form follows.
pub const SPEC_VERSION: &str = "ptir.v0.6";

pub use gpt_rs_ir_lexer::{is_ident_start, is_name_char, Token, TokenKind};

pub use self::cursor::Cursor;
pub use self::print::TextOptions;

/// Whether `src` starts with the version of the spec it follows, as a program stored for later
/// does.
pub fn is_versioned(src: &str) -> bool {
    Cursor::new(src).peek_is("spec_version")
}

/// ` {a = 1, b = 2}`, the attributes of an argument or a result of a symbol op after its type, or
/// nothing for none.
pub fn item_attrs_text(entries: &[String]) -> String {
    if entries.is_empty() {
        return String::new();
    }
    format!(" {{{}}}", entries.join(", "))
}

/// The attribute of each item, or none when no item has one.
pub fn any_items<T>(items: Vec<Option<T>>) -> Vec<Option<T>> {
    if items.iter().any(Option::is_some) {
        items
    } else {
        Vec::new()
    }
}

/// The attribute `key` of each item, which every item has, or none when no item has one. `what`
/// names the items in the error.
pub fn every_item<T>(items: Vec<Option<T>>, key: &str, what: &str) -> Result<Vec<T>, String> {
    if items.iter().all(Option::is_none) {
        return Ok(Vec::new());
    }
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        let Some(item) = item else {
            return Err(format!("some {what} have `{key}` and some do not"));
        };
        out.push(item);
    }
    Ok(out)
}
