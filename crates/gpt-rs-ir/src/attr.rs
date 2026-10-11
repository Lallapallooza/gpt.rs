//! Attribute values: the fields of an op struct.
//!
//! Every field type implements [`Attr`], which gives its text form. `#[derive(Attr)]` implements
//! it for fieldless enums, printed as their snake_case variant names.

use std::collections::BTreeMap;
use std::fmt;
use std::hash::{Hash, Hasher};

use anyhow::{bail, ensure, Result};
use smallvec::{Array, SmallVec};
use tinyvec::ArrayVec;

use crate::text::{is_ident_char, is_ident_start, Cursor};
use crate::types::{ranked, DType, Dim, Perm, Ranked, Sym, TensorType, Type};

pub trait Attr: Sized + Clone + PartialEq + Eq + Hash + fmt::Debug {
    fn print(&self, out: &mut String);
    fn parse(c: &mut Cursor<'_>) -> Result<Self>;
}

/// The attributes that the ops of a family share, held in one struct: each op of the family is a
/// newtype over it. `#[derive(AttrGroup)]` implements it from the struct's fields, which are the
/// ops' attributes, written as an op's own fields are.
pub trait AttrGroup: Sized {
    const INFOS: &'static [crate::op::AttrInfo];
    /// Pushes `(name, text)` for every attribute, except an `#[attr(default)]` attribute at its
    /// default.
    fn print_attrs(&self, out: &mut Vec<(&'static str, String)>);
    /// Parses the optional attribute dictionary of the op `op`.
    fn parse_attrs(op: &str, c: &mut Cursor<'_>) -> Result<Self>;
    /// Parses the optional parameters, `<key = value, ...>`, after `name`, the name of a variant of
    /// an attribute enum that holds the group.
    fn parse_params(name: &str, c: &mut Cursor<'_>) -> Result<Self>;
}

/// An enum attribute type, which `#[derive(Attr)]` registers so that the generated docs can list
/// its values.
pub struct AttrEnum {
    /// The name of the enum type.
    pub name: &'static str,
    /// The text name and the doc comment of each value, in order.
    pub values: fn() -> Vec<(&'static str, &'static str)>,
}

/// Every enum that `#[derive(Attr)]` defines.
#[linkme::distributed_slice]
pub static ATTR_ENUMS: [AttrEnum];

/// The text name and the doc comment of each value of `T`, in order.
pub fn enum_values<T: Mnemonic>() -> Vec<(&'static str, &'static str)> {
    T::ALL
        .iter()
        .map(|&value| (value.mnemonic(), value.doc()))
        .collect()
}

/// A fieldless enum with one text name per variant.
pub trait Mnemonic: Sized + Copy + 'static {
    const ALL: &'static [Self];
    fn mnemonic(self) -> &'static str;

    /// The doc comment of the variant, or empty.
    fn doc(self) -> &'static str {
        ""
    }

    fn from_mnemonic(name: &str) -> Option<Self> {
        Self::ALL.iter().copied().find(|v| v.mnemonic() == name)
    }
}

macro_rules! int_attr {
    ($($t:ty),*) => {$(
        impl Attr for $t {
            fn print(&self, out: &mut String) {
                out.push_str(&self.to_string());
            }
            fn parse(c: &mut Cursor<'_>) -> Result<Self> {
                let value = c.int()?;
                <$t>::try_from(value)
                    .map_err(|_| c.error(format!("{value} does not fit {}", stringify!($t))))
            }
        }
    )*};
}

int_attr!(usize, isize, i8, u8, i16, u16, i32, u32, i64, u64);

impl Attr for bool {
    fn print(&self, out: &mut String) {
        out.push_str(if *self { "true" } else { "false" });
    }
    fn parse(c: &mut Cursor<'_>) -> Result<Self> {
        if c.eat("true") {
            Ok(true)
        } else if c.eat("false") {
            Ok(false)
        } else {
            Err(c.error("expected `true` or `false`"))
        }
    }
}

pub(crate) fn print_string(s: &str, out: &mut String) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            ch => out.push(ch),
        }
    }
    out.push('"');
}

impl Attr for String {
    fn print(&self, out: &mut String) {
        print_string(self, out);
    }
    fn parse(c: &mut Cursor<'_>) -> Result<Self> {
        c.string()
    }
}

impl Attr for DType {
    fn print(&self, out: &mut String) {
        out.push_str(self.mnemonic());
    }
    fn parse(c: &mut Cursor<'_>) -> Result<Self> {
        let name = c.ident()?;
        DType::from_mnemonic(name).ok_or_else(|| c.error(format!("unknown dtype `{name}`")))
    }
}

/// Prints `items` as `[a, b, ...]`.
fn print_list<T: Attr>(items: &[T], out: &mut String) {
    out.push('[');
    for (i, item) in items.iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        item.print(out);
    }
    out.push(']');
}

impl<T: Attr> Attr for Vec<T> {
    fn print(&self, out: &mut String) {
        print_list(self, out);
    }
    fn parse(c: &mut Cursor<'_>) -> Result<Self> {
        c.expect("[")?;
        c.list("]", T::parse)
    }
}

impl<A: Array> Attr for SmallVec<A>
where
    A::Item: Attr,
{
    fn print(&self, out: &mut String) {
        print_list(self, out);
    }
    fn parse(c: &mut Cursor<'_>) -> Result<Self> {
        Vec::parse(c).map(SmallVec::from_vec)
    }
}

impl<A: tinyvec::Array + Clone> Attr for ArrayVec<A>
where
    A::Item: Attr,
{
    fn print(&self, out: &mut String) {
        print_list(self, out);
    }
    fn parse(c: &mut Cursor<'_>) -> Result<Self> {
        let items = Vec::parse(c)?;
        ensure!(
            items.len() <= A::CAPACITY,
            c.error(format!(
                "a list of {} entries, above its limit of {}",
                items.len(),
                A::CAPACITY
            ))
        );
        Ok(items.into_iter().collect())
    }
}

impl Attr for Perm {
    fn print(&self, out: &mut String) {
        print_list(self, out);
    }
    fn parse(c: &mut Cursor<'_>) -> Result<Self> {
        Ranked::<usize>::parse(c).map(Perm::from_iter)
    }
}

impl<T: Attr> Attr for Option<T> {
    fn print(&self, out: &mut String) {
        match self {
            Some(value) => value.print(out),
            None => out.push_str("none"),
        }
    }
    fn parse(c: &mut Cursor<'_>) -> Result<Self> {
        if c.eat("none") {
            Ok(None)
        } else {
            T::parse(c).map(Some)
        }
    }
}

impl<A: Attr, B: Attr> Attr for (A, B) {
    fn print(&self, out: &mut String) {
        out.push('(');
        self.0.print(out);
        out.push_str(", ");
        self.1.print(out);
        out.push(')');
    }
    fn parse(c: &mut Cursor<'_>) -> Result<Self> {
        c.expect("(")?;
        let first = A::parse(c)?;
        c.expect(",")?;
        let second = B::parse(c)?;
        c.expect(")")?;
        Ok((first, second))
    }
}

/// Printed as `{key = value, ...}`. A key that is not an identifier without dots is printed in
/// quotes.
impl<T: Attr> Attr for BTreeMap<String, T> {
    fn print(&self, out: &mut String) {
        out.push('{');
        for (i, (key, value)) in self.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            if key.starts_with(is_ident_start) && key.chars().all(|c| c != '.' && is_ident_char(c))
            {
                out.push_str(key);
            } else {
                print_string(key, out);
            }
            out.push_str(" = ");
            value.print(out);
        }
        out.push('}');
    }
    fn parse(c: &mut Cursor<'_>) -> Result<Self> {
        c.expect("{")?;
        let mut map = BTreeMap::new();
        c.list("}", |c| {
            let key = if c.peek() == Some('"') {
                c.string()?
            } else {
                c.ident()?.to_string()
            };
            c.expect("=")?;
            let value = T::parse(c)?;
            if map.insert(key.clone(), value).is_some() {
                bail!(c.error(format!("duplicate key `{key}`")));
            }
            Ok(())
        })?;
        Ok(map)
    }
}

/// The name of a top-level symbol, such as a `tile.dispatch`, written `@name`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SymbolRef(pub String);

impl Attr for SymbolRef {
    fn print(&self, out: &mut String) {
        out.push('@');
        out.push_str(&self.0);
    }
    fn parse(c: &mut Cursor<'_>) -> Result<Self> {
        Ok(SymbolRef(c.sigil_name('@')?.to_string()))
    }
}

/// An `f64` compared and hashed by its bits, so it can be an attribute.
#[derive(Debug, Clone, Copy)]
pub struct F64(pub f64);

impl PartialEq for F64 {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}

impl Eq for F64 {}

impl Hash for F64 {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

/// Prints the shortest decimal that parses back to the same value. A finite value always gets a
/// `.` or an exponent, so it reads as a float.
pub(crate) fn print_float(value: impl fmt::Debug, out: &mut String) {
    let text = format!("{value:?}");
    out.push_str(&text);
    if !text.contains(['.', 'e', 'N', 'i']) {
        out.push_str(".0");
    }
}

impl Attr for F64 {
    fn print(&self, out: &mut String) {
        print_float(self.0, out);
    }
    fn parse(c: &mut Cursor<'_>) -> Result<Self> {
        c.float().map(F64)
    }
}

impl Attr for Dim {
    fn print(&self, out: &mut String) {
        out.push_str(&self.to_string());
    }
    fn parse(c: &mut Cursor<'_>) -> Result<Self> {
        if c.eat("{") {
            let name = c.ident()?;
            c.expect("}")?;
            Ok(Dim::Dynamic(Sym::new(name)))
        } else {
            usize::parse(c).map(Dim::Static)
        }
    }
}

fn parse_tensor_body(c: &Cursor<'_>, body: &str) -> Result<TensorType> {
    let mut parts = Vec::new();
    let mut depth = 0usize;
    let mut start = 0usize;
    for (i, ch) in body.char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => depth = depth.saturating_sub(1),
            'x' if depth == 0 => {
                parts.push(&body[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    parts.push(&body[start..]);
    let (dtype_name, dims) = parts
        .split_last()
        .expect("one part more than there are `x`s");
    let dtype_name = dtype_name.trim();
    let dtype = DType::from_mnemonic(dtype_name)
        .ok_or_else(|| c.error(format!("unknown dtype `{dtype_name}` in tensor type")))?;
    let dims = dims
        .iter()
        .map(|part| {
            let mut dim = Cursor::new(part);
            Dim::parse(&mut dim)
                .ok()
                .filter(|_| dim.at_end())
                .ok_or_else(|| c.error(format!("bad dimension `{}` in tensor type", part.trim())))
        })
        .collect::<Result<Vec<_>>>()?;
    let dims = ranked(dims).map_err(|err| c.error(format!("tensor type: {err}")))?;
    Ok(TensorType { dtype, dims })
}

impl Attr for TensorType {
    fn print(&self, out: &mut String) {
        out.push_str(&self.to_string());
    }
    fn parse(c: &mut Cursor<'_>) -> Result<Self> {
        c.expect("tensor")?;
        c.expect("<")?;
        let body = c.until_close_angle()?;
        let tensor = parse_tensor_body(c, body)?;
        c.expect(">")?;
        Ok(tensor)
    }
}

impl Attr for Type {
    fn print(&self, out: &mut String) {
        out.push_str(&self.to_string());
    }
    fn parse(c: &mut Cursor<'_>) -> Result<Self> {
        if c.peek_str("tensor<") {
            return TensorType::parse(c).map(Type::Tensor);
        }
        if c.eat("index") {
            return Ok(Type::Index);
        }
        let name = c.ident()?;
        DType::from_mnemonic(name)
            .map(Type::Scalar)
            .ok_or_else(|| c.error(format!("unknown type `{name}`")))
    }
}
