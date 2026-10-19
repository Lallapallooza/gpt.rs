//! Attribute values and their text forms, which [`Attr`] describes. An attribute value is a field
//! of an op struct.

mod dense;

use std::collections::BTreeMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use anyhow::{bail, ensure, Result};
use smallvec::{Array, SmallVec};
use tinyvec::ArrayVec;

pub use self::dense::Dense;
use crate::ir::{ranked, DType, Dim, Ranked, Sym};
use crate::text::{is_ident_start, is_name_char, Cursor, TokenKind};

/// Every field type implements [`Attr`]: it prints its text form and reads it back with a
/// [`Cursor`], one token at a time. The impls in `ir/attr/mod.rs` are the leaves, each one token
/// or a list of attributes: integers, floats, strings, names, `[a, b]`, `(a, b)`, `{key = value}`
/// and `none`. Every other attribute declares its text form with `#[derive(Attr)]`, which generates
/// the printer and the parser from one declaration:
///
/// - A fieldless enum is written by the snake_case name of each variant: `red_ordered`.
/// - A variant with named fields, or one that holds an [`AttrGroup`], is written by its name and
///   its fields as parameters, each left out at its default: `top_k<k = 3>`.
/// - `#[attr(format = "...")]` on a struct or on a variant gives its text form as a sequence of
///   elements, MLIR's declarative format:
///   - `` `by` `` is a literal: one token, a keyword, a name with its sigil (`!loop.buf`,
///     `#sched`) or punctuation (`->`, `[`).
///   - `$extent` is the text form of the field `extent`, and `$0` that of the first field of a
///     tuple struct or variant.
///   - `` (`pad` $pad^)? `` is an optional group: it is printed when its anchor, the field marked `^`,
///     is not at its default (an `Option` field is `Some`), and read when the token its first
///     element starts with comes next. A field of a group that is not its anchor has a default.
///   - `shape($dims, $dtype)` is the extents and the dtype of a tensor: `4x{T}xf32`.
///   - `tuple($writes)` is a list written as op results are: one item alone, any other number in
///     parentheses.
///   - `struct($nest, ...)` is `key = value` for each field, in any order, a field at its default
///     left out. A closing `>`, `}`, `)` or `]` follows it.
///   - ``` `` ``` removes the space between the elements around it.
///
///   The printer puts one space between elements, except after `(`, `[`, `<` and `{`, and before
///   `)`, `]`, `>`, `}`, `,`, `(`, `[`, `<`, and on either side of `..`.
///
///   The next token decides each choice, and `#[derive(Attr)]` rejects a format where it cannot:
///   two variants that start with one literal, or an optional group that starts with the literal
///   that follows it. The parser tries the variants in order and reads the first whose first
///   token comes next, so a variant that starts with a field goes after every variant whose
///   first token the field could also start with.
/// - `#[attr(via = Text)]` on a type whose text form is that of the type `Text`, a struct that
///   declares it: `From<&T> for Text` writes it, and `TryFrom<Text> for T` checks and builds the
///   value, failing at the start of its text.
///
/// Fields take `#[attr(default)]`, the type's default, or `#[attr(default = expr)]`. A type with
/// a format or a via also gets `Display`, its text form.
pub trait Attr: Sized + Clone + PartialEq + Eq + Hash + fmt::Debug {
    /// The name of the type whose values the text writes: the type of the items of a list, a map
    /// or an option, else the type itself. The generated docs show the values or the format of
    /// the type of this name.
    const ITEM: &'static str;

    /// The type as the reference pages name it: `[Fold]` for a list of folds, the type of its
    /// value for an option, else its own name.
    fn type_name() -> String {
        Self::ITEM.to_string()
    }

    fn print(&self, out: &mut String);
    fn parse(cursor: &mut Cursor) -> Result<Self>;
    /// Whether the next token can start the text form. An optional group of a format, or a
    /// variant of an enum, that starts with this attribute is read only when it does.
    fn peek(cursor: &Cursor) -> bool;
}

/// The attributes that the ops of a family share, held in one struct: each op of the family is a
/// newtype over it. `#[derive(AttrGroup)]` implements it from the struct's fields, which are the
/// ops' attributes, written as an op's own fields are.
pub trait AttrGroup: Sized {
    const INFOS: &'static [crate::ir::AttrInfo];
    /// Pushes `(name, text)` for every attribute, except an `#[attr(default)]` attribute at its
    /// default.
    fn print_attrs(&self, out: &mut Vec<(&'static str, String)>);
    /// Parses the optional attribute dictionary of the op `op`. A key the group does not have is
    /// handed to `other`, with its position.
    fn parse_attrs(
        op: &str,
        cursor: &mut Cursor,
        other: &mut dyn FnMut(&str, &mut Cursor, usize) -> Result<()>,
    ) -> Result<Self>;
    /// Parses the optional parameters, `<key = value, ...>`, after `name`, the name of a variant of
    /// an attribute enum that holds the group.
    fn parse_params(name: &str, cursor: &mut Cursor) -> Result<Self>;
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

/// An attribute type with a format or a via, which `#[derive(Attr)]` registers so that the
/// generated docs can show its text form.
pub struct AttrFormat {
    /// The name of the type, or `Type::Variant` for a variant of an enum.
    pub name: &'static str,
    /// The format, as declared, or empty for a type with a via.
    pub format: &'static str,
    /// The types of the fields it writes, as declared, or its via.
    pub fields: &'static [&'static str],
}

/// Every format that `#[derive(Attr)]` declares.
#[linkme::distributed_slice]
pub static ATTR_FORMATS: [AttrFormat];

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
        Self::ALL
            .iter()
            .copied()
            .find(|value| value.mnemonic() == name)
    }
}

/// Reads a value of the fieldless enum `T`, written by its name.
pub fn parse_mnemonic<T: Mnemonic>(cursor: &mut Cursor, what: &str) -> Result<T> {
    let start = cursor.position();
    let name = cursor.ident()?;
    T::from_mnemonic(name).ok_or_else(|| cursor.error_at(start, format!("unknown {what} `{name}`")))
}

/// Whether the next token is the name of a value of the fieldless enum `T`.
pub fn peek_mnemonic<T: Mnemonic>(cursor: &Cursor) -> bool {
    let next = cursor.peek();
    next.kind == TokenKind::Ident && T::from_mnemonic(next.text).is_some()
}

macro_rules! int_attr {
    ($($t:ty),*) => {$(
        impl Attr for $t {
            const ITEM: &'static str = stringify!($t);

            fn print(&self, out: &mut String) {
                out.push_str(&self.to_string());
            }
            fn parse(cursor: &mut Cursor) -> Result<Self> {
                let start = cursor.position();
                let text = cursor.int_text()?;
                text.parse().map_err(|_| {
                    cursor.error_at(start, format!("{text} does not fit {}", stringify!($t)))
                })
            }
            fn peek(cursor: &Cursor) -> bool {
                cursor.peek_unsigned().kind == TokenKind::Int
            }
        }
    )*};
}

int_attr!(usize, isize, i8, u8, i16, u16, i32, u32, i64, u64, u128);

impl Attr for bool {
    const ITEM: &'static str = "bool";

    fn print(&self, out: &mut String) {
        out.push_str(if *self { "true" } else { "false" });
    }
    fn parse(cursor: &mut Cursor) -> Result<Self> {
        if cursor.eat("true") {
            Ok(true)
        } else if cursor.eat("false") {
            Ok(false)
        } else {
            Err(cursor.unexpected("`true` or `false`"))
        }
    }
    fn peek(cursor: &Cursor) -> bool {
        cursor.peek_is("true") || cursor.peek_is("false")
    }
}

pub(crate) fn print_string(text: &str, out: &mut String) {
    out.push('"');
    for ch in text.chars() {
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
    const ITEM: &'static str = "String";

    fn print(&self, out: &mut String) {
        print_string(self, out);
    }
    fn parse(cursor: &mut Cursor) -> Result<Self> {
        cursor.string()
    }
    fn peek(cursor: &Cursor) -> bool {
        cursor.peek_kind(TokenKind::String)
    }
}

impl Attr for Arc<str> {
    const ITEM: &'static str = "String";

    fn print(&self, out: &mut String) {
        print_string(self, out);
    }
    fn parse(cursor: &mut Cursor) -> Result<Self> {
        cursor.string().map(Arc::from)
    }
    fn peek(cursor: &Cursor) -> bool {
        cursor.peek_kind(TokenKind::String)
    }
}

impl Attr for DType {
    const ITEM: &'static str = "DType";

    fn print(&self, out: &mut String) {
        out.push_str(self.mnemonic());
    }
    fn parse(cursor: &mut Cursor) -> Result<Self> {
        if !cursor.peek_kind(TokenKind::Ident) {
            return Err(cursor.unexpected("a dtype"));
        }
        parse_mnemonic(cursor, "dtype")
    }
    fn peek(cursor: &Cursor) -> bool {
        peek_mnemonic::<DType>(cursor)
    }
}

/// The name of a dimension symbol, written `{T}` where it is an extent.
impl Attr for Sym {
    const ITEM: &'static str = "Sym";

    fn print(&self, out: &mut String) {
        out.push_str(self.as_str());
    }
    fn parse(cursor: &mut Cursor) -> Result<Self> {
        cursor.ident().map(Sym::new)
    }
    fn peek(cursor: &Cursor) -> bool {
        cursor.peek_kind(TokenKind::Ident)
    }
}

/// A level of a target that a schedule names: a memory level, `l1`, where a copy stages and a
/// prefetch asks for lines, or an execution level, `core`, whose units run the iterations of a
/// parallel loop. Each target defines its levels.
#[derive(crate::Attr, Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[attr(format = "$0")]
pub struct Level(Sym);

impl Level {
    pub fn new(name: impl AsRef<str>) -> Self {
        Self(Sym::new(name))
    }

    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

/// Prints `items` as `[a, b, ...]`.
fn print_list<T: Attr>(items: &[T], out: &mut String) {
    out.push('[');
    for (position, item) in items.iter().enumerate() {
        if position > 0 {
            out.push_str(", ");
        }
        item.print(out);
    }
    out.push(']');
}

impl<T: Attr> Attr for Vec<T> {
    const ITEM: &'static str = T::ITEM;

    fn type_name() -> String {
        format!("[{}]", T::type_name())
    }

    fn print(&self, out: &mut String) {
        print_list(self, out);
    }
    fn parse(cursor: &mut Cursor) -> Result<Self> {
        cursor.expect("[")?;
        cursor.list("]", T::parse)
    }
    fn peek(cursor: &Cursor) -> bool {
        cursor.peek_is("[")
    }
}

impl<A: Array> Attr for SmallVec<A>
where
    A::Item: Attr,
{
    const ITEM: &'static str = A::Item::ITEM;

    fn type_name() -> String {
        format!("[{}]", A::Item::type_name())
    }

    fn print(&self, out: &mut String) {
        print_list(self, out);
    }
    fn parse(cursor: &mut Cursor) -> Result<Self> {
        Vec::parse(cursor).map(SmallVec::from_vec)
    }
    fn peek(cursor: &Cursor) -> bool {
        cursor.peek_is("[")
    }
}

impl<A: tinyvec::Array + Clone> Attr for ArrayVec<A>
where
    A::Item: Attr,
{
    const ITEM: &'static str = <A::Item as Attr>::ITEM;

    fn type_name() -> String {
        format!("[{}]", <A::Item as Attr>::type_name())
    }

    fn print(&self, out: &mut String) {
        print_list(self, out);
    }
    fn parse(cursor: &mut Cursor) -> Result<Self> {
        let start = cursor.position();
        let items = Vec::parse(cursor)?;
        ensure!(
            items.len() <= A::CAPACITY,
            cursor.error_at(
                start,
                format!(
                    "a list of {} entries, above its limit of {}",
                    items.len(),
                    A::CAPACITY
                )
            )
        );
        Ok(items.into_iter().collect())
    }
    fn peek(cursor: &Cursor) -> bool {
        cursor.peek_is("[")
    }
}

impl<T: Attr> Attr for Option<T> {
    const ITEM: &'static str = T::ITEM;

    fn type_name() -> String {
        T::type_name()
    }

    fn print(&self, out: &mut String) {
        match self {
            Some(value) => value.print(out),
            None => out.push_str("none"),
        }
    }
    fn parse(cursor: &mut Cursor) -> Result<Self> {
        if cursor.eat("none") {
            Ok(None)
        } else {
            T::parse(cursor).map(Some)
        }
    }
    fn peek(cursor: &Cursor) -> bool {
        cursor.peek_is("none") || T::peek(cursor)
    }
}

impl<A: Attr, B: Attr> Attr for (A, B) {
    const ITEM: &'static str = "tuple";

    fn type_name() -> String {
        format!("({}, {})", A::type_name(), B::type_name())
    }

    fn print(&self, out: &mut String) {
        out.push('(');
        self.0.print(out);
        out.push_str(", ");
        self.1.print(out);
        out.push(')');
    }
    fn parse(cursor: &mut Cursor) -> Result<Self> {
        cursor.expect("(")?;
        let first = A::parse(cursor)?;
        cursor.expect(",")?;
        let second = B::parse(cursor)?;
        cursor.expect(")")?;
        Ok((first, second))
    }
    fn peek(cursor: &Cursor) -> bool {
        cursor.peek_is("(")
    }
}

/// Reads `key = value, ...`: a key is a name or a string, and `entry` reads the value of each,
/// given the key and its position. It reads entries while a key comes next, and none when none
/// does.
pub fn parse_entries<'a>(
    cursor: &mut Cursor<'a>,
    mut entry: impl FnMut(&mut Cursor<'a>, &str, usize) -> Result<()>,
) -> Result<()> {
    while cursor.peek_kind(TokenKind::Ident) || cursor.peek_kind(TokenKind::String) {
        let at = cursor.position();
        let key = if cursor.peek_kind(TokenKind::String) {
            cursor.string()?
        } else {
            cursor.ident()?.to_string()
        };
        cursor.expect("=")?;
        entry(cursor, &key, at)?;
        if !cursor.eat(",") {
            break;
        }
    }
    Ok(())
}

/// Reads `{key = value, ...}` when `{` comes next, handing each key to `entry`.
pub fn parse_dict<'a>(
    cursor: &mut Cursor<'a>,
    entry: impl FnMut(&mut Cursor<'a>, &str, usize) -> Result<()>,
) -> Result<()> {
    if cursor.eat("{") {
        parse_entries(cursor, entry)?;
        cursor.expect("}")?;
    }
    Ok(())
}

/// Printed as `{key = value, ...}`. A key that is not an identifier without dots is printed in
/// quotes.
impl<T: Attr> Attr for BTreeMap<String, T> {
    const ITEM: &'static str = T::ITEM;

    fn type_name() -> String {
        format!("{{name = {}}}", T::type_name())
    }

    fn print(&self, out: &mut String) {
        out.push('{');
        for (position, (key, value)) in self.iter().enumerate() {
            if position > 0 {
                out.push_str(", ");
            }
            if key.starts_with(is_ident_start)
                && key
                    .chars()
                    .all(|character| character != '.' && is_name_char(character))
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
    fn parse(cursor: &mut Cursor) -> Result<Self> {
        let mut map = BTreeMap::new();
        cursor.expect("{")?;
        parse_entries(cursor, |entry_cursor, key, at| {
            let value = T::parse(entry_cursor)?;
            if map.insert(key.to_string(), value).is_some() {
                bail!(entry_cursor.error_at(at, format!("duplicate key `{key}`")));
            }
            Ok(())
        })?;
        cursor.expect("}")?;
        Ok(map)
    }
    fn peek(cursor: &Cursor) -> bool {
        cursor.peek_is("{")
    }
}

/// A list attribute, which `tuple(..)` writes: one item alone, any other number in parentheses.
pub trait AttrList: FromIterator<Self::Item> {
    type Item: Attr;
    fn items(&self) -> &[Self::Item];
}

impl<T: Attr> AttrList for Vec<T> {
    type Item = T;
    fn items(&self) -> &[T] {
        self
    }
}

impl<A: Array> AttrList for SmallVec<A>
where
    A::Item: Attr,
{
    type Item = A::Item;
    fn items(&self) -> &[A::Item] {
        self
    }
}

/// `item`, or `(a, b, ...)` for any number of items but one.
pub fn print_tuple<'t, T: Attr + 't>(list: impl IntoIterator<Item = &'t T>, out: &mut String) {
    let items: Vec<&T> = list.into_iter().collect();
    if let [item] = items.as_slice() {
        item.print(out);
        return;
    }
    out.push('(');
    for (position, item) in items.iter().enumerate() {
        if position > 0 {
            out.push_str(", ");
        }
        item.print(out);
    }
    out.push(')');
}

/// Reads what [`print_tuple`] writes.
pub fn parse_tuple<L: AttrList>(cursor: &mut Cursor) -> Result<L> {
    if cursor.eat("(") {
        return Ok(cursor.list(")", L::Item::parse)?.into_iter().collect());
    }
    Ok(std::iter::once(L::Item::parse(cursor)?).collect())
}

/// Whether what [`print_tuple`] writes comes next.
pub fn peek_tuple<L: AttrList>(cursor: &Cursor) -> bool {
    cursor.peek_is("(") || L::Item::peek(cursor)
}

/// `4x{T}xf32`: the extents of a tensor, each followed by `x`, then its dtype.
pub fn print_shape(dims: &[Dim], dtype: &DType, out: &mut String) {
    for dim in dims {
        dim.print(out);
        out.push('x');
    }
    dtype.print(out);
}

/// Reads what [`print_shape`] writes.
pub fn parse_shape(cursor: &mut Cursor) -> Result<(Ranked<Dim>, DType)> {
    let start = cursor.position();
    let mut extents = Vec::new();
    while Dim::peek(cursor) {
        extents.push(Dim::parse(cursor)?);
        cursor.shape_x()?;
    }
    if !cursor.peek_kind(TokenKind::Ident) {
        return Err(cursor.unexpected("an extent or a dtype"));
    }
    let dtype = DType::parse(cursor)?;
    let dims = ranked(extents).map_err(|err| cursor.error_at(start, err))?;
    Ok((dims, dtype))
}

/// Whether what [`print_shape`] writes comes next.
pub fn peek_shape(cursor: &Cursor) -> bool {
    Dim::peek(cursor) || DType::peek(cursor)
}

/// The name of a top-level symbol, such as a `tile.dispatch`, written `@name`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SymbolRef(pub String);

impl Attr for SymbolRef {
    const ITEM: &'static str = "SymbolRef";

    fn print(&self, out: &mut String) {
        out.push('@');
        out.push_str(&self.0);
    }
    fn parse(cursor: &mut Cursor) -> Result<Self> {
        Ok(SymbolRef(
            cursor
                .sigil_name(TokenKind::Symbol, "a symbol")?
                .to_string(),
        ))
    }
    fn peek(cursor: &Cursor) -> bool {
        cursor.peek_kind(TokenKind::Symbol)
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

/// Printed as a float, `0.5`, `-inf`; read from an integer too. It starts an optional group or a
/// variant only as a float.
impl Attr for F64 {
    const ITEM: &'static str = "F64";

    fn print(&self, out: &mut String) {
        print_float(self.0, out);
    }
    fn parse(cursor: &mut Cursor) -> Result<Self> {
        cursor.float().map(F64)
    }
    fn peek(cursor: &Cursor) -> bool {
        cursor.peek_float()
    }
}

/// A number as written, its sign included: `-1.5`, `3`, `-inf`, `NaN`. The text form of a value
/// whose dtype says how to read it.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Number(pub String);

impl Number {
    /// Whether it is written as a float: with a fraction or an exponent, or `inf` or `NaN`.
    pub fn is_float(&self) -> bool {
        self.0.contains(['.', 'e', 'E', 'i', 'N'])
    }
}

impl Attr for Number {
    const ITEM: &'static str = "Number";

    fn print(&self, out: &mut String) {
        out.push_str(&self.0);
    }
    fn parse(cursor: &mut Cursor) -> Result<Self> {
        cursor.number().map(Number)
    }
    fn peek(cursor: &Cursor) -> bool {
        cursor.peek_number()
    }
}
