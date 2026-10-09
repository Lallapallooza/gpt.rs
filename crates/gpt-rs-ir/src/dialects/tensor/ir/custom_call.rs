//! `tensor.custom_call`: a call into a backend kernel by name.

use std::collections::BTreeMap;

use anyhow::{bail, ensure, Result};

use crate::attr::{print_string, Attr, F64};
use crate::op::{NoRegions, OpDef, OwnedTypes, TypeCx, Types};
use crate::text::Cursor;
use crate::types::TensorType;

/// A value in a custom call's attribute map.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CustomAttr {
    I64(i64),
    F64(F64),
    Bool(bool),
    String(String),
    I64Array(Vec<i64>),
    F64Array(Vec<F64>),
    BoolArray(Vec<bool>),
    StringArray(Vec<String>),
}

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

/// `tensor.custom_call`: runs the backend kernel `target` on operands that the target defines, with
/// one result per entry of `results`. The result types are attributes, since only the rewrite that
/// creates the call knows them.
///
/// - Effects: a custom call has effects that the IR does not describe. Passes MUST NOT merge,
///   precompute or erase a custom call, even when nothing reads its results. Backends MAY cache
///   internal compiled kernels, packed weights, or plans.
/// - Portability: if a backend does not recognize `target`, it MUST reject it as unimplemented
///   (no silent fallback to a portable decomposition). The interpreter has no semantics for it.
/// - Typing: `results` is authoritative. Backends MAY validate that operand dtypes and shapes and
///   `attrs` agree with it.
/// - Stable naming: `target` strings MUST be namespaced and versioned, such as
///   `<namespace>.conv2d.nhwc.f32.v1`.
/// - Attributes: `attrs` is a small serializable map of primitives (integers, floats, bools and
///   strings) and arrays of primitives. Backends SHOULD reject unknown or malformed attributes as
///   a spec violation.
#[derive(crate::Op, Debug, Clone, PartialEq, Eq, Hash)]
#[op(
    "tensor.custom_call",
    operands(inputs: Variadic<Tensor>),
    results(outputs: Variadic<Tensor>),
    effects(opaque)
)]
pub struct CustomCall {
    pub target: String,
    #[attr(default)]
    pub attrs: BTreeMap<String, CustomAttr>,
    pub results: Vec<TensorType>,
}

impl OpDef for CustomCall {
    fn result_types(
        &self,
        _ins: CustomCallOperands<Types<'_>>,
        _cx: &TypeCx<'_, NoRegions>,
    ) -> Result<CustomCallResults<OwnedTypes>> {
        ensure!(!self.target.is_empty(), "the target is empty");
        Ok(CustomCallResults {
            outputs: self.results.clone(),
        })
    }
}
