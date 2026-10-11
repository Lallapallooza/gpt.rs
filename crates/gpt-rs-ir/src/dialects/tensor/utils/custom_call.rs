//! The text form of the attributes of `tensor.custom_call`.

use anyhow::{bail, Result};

use crate::attr::{print_string, Attr, F64};
use crate::dialects::tensor::ir::*;
use crate::text::Cursor;

fn print_array<T>(tag: &str, items: &[T], print: impl Fn(&T, &mut String), out: &mut String) {
    out.push_str("array<");
    out.push_str(tag);
    out.push(':');
    for (i, item) in items.iter().enumerate() {
        out.push_str(if i == 0 { " " } else { ", " });
        print(item, out);
    }
    out.push('>');
}

/// Scalars print plainly (`3`, `1.5`, `true`, `"s"`). Arrays print with their element type
/// (`array<i64: 1, 2>`, `array<f64: 0.5>`, `array<bool: true>`, `array<str: "a">`).
impl Attr for CustomAttr {
    fn print(&self, out: &mut String) {
        match self {
            CustomAttr::I64(v) => v.print(out),
            CustomAttr::F64(v) => v.print(out),
            CustomAttr::Bool(v) => v.print(out),
            CustomAttr::String(v) => print_string(v, out),
            CustomAttr::I64Array(v) => print_array("i64", v, |x, o| x.print(o), out),
            CustomAttr::F64Array(v) => print_array("f64", v, |x, o| x.print(o), out),
            CustomAttr::BoolArray(v) => print_array("bool", v, |x, o| x.print(o), out),
            CustomAttr::StringArray(v) => print_array("str", v, |x, o| print_string(x, o), out),
        }
    }

    fn parse(c: &mut Cursor<'_>) -> Result<Self> {
        if c.eat("array") {
            c.expect("<")?;
            let tag = c.ident()?;
            c.expect(":")?;
            return match tag {
                "i64" => c.list(">", i64::parse).map(CustomAttr::I64Array),
                "f64" => c.list(">", F64::parse).map(CustomAttr::F64Array),
                "bool" => c.list(">", bool::parse).map(CustomAttr::BoolArray),
                "str" => c.list(">", |c| c.string()).map(CustomAttr::StringArray),
                other => bail!(c.error(format!("unknown array element type `{other}`"))),
            };
        }
        if c.peek() == Some('"') {
            return c.string().map(CustomAttr::String);
        }
        if c.peek_str("true") || c.peek_str("false") {
            return bool::parse(c).map(CustomAttr::Bool);
        }
        if c.peek_is_float() {
            return F64::parse(c).map(CustomAttr::F64);
        }
        i64::parse(c).map(CustomAttr::I64)
    }
}
