//! The format compiler of `#[derive(Attr)]`: it reads the format of a struct or a variant, checks
//! it, and generates the code that prints, reads and peeks the fields it writes.
//! `gpt_rs_ir::ir::Attr` documents the format language.

use gpt_rs_ir_lexer::{token, TokenKind};
use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::{format_ident, quote};
use syn::{Ident, LitStr, Type as SynType};

/// A field that a format writes.
pub(crate) struct FormatField {
    /// Its name in the format: `extent`, or `0` for the first field of a tuple.
    pub name: String,
    /// The local that holds it in the generated code.
    pub binding: Ident,
    pub ty: SynType,
    /// The value it takes when the text leaves it out, from `#[attr(default)]` or
    /// `#[attr(default = expr)]`. An `Option` field is `None` without one.
    pub default: Option<TokenStream2>,
}

impl FormatField {
    pub fn new(name: String, ty: SynType, default: Option<TokenStream2>) -> Self {
        Self {
            binding: format_ident!("f_{}", name),
            name,
            ty,
            default,
        }
    }

    /// Whether the text may leave the field out.
    pub fn has_default(&self) -> bool {
        self.default_value().is_some()
    }

    /// The type inside `Option<..>`, when the field is an `Option`.
    fn option_inner(&self) -> Option<&SynType> {
        option_inner(&self.ty)
    }

    /// The value the field takes when its text is left out, if it has one.
    fn default_value(&self) -> Option<TokenStream2> {
        if let Some(default) = &self.default {
            return Some(default.clone());
        }
        self.option_inner()
            .map(|_| quote!(::std::option::Option::None))
    }
}

/// A field that a `key = value` entry writes.
pub(crate) struct Entry {
    pub key: String,
    pub ty: SynType,
    /// Its value when the text leaves the entry out, or `None` when it must be there.
    pub default: Option<TokenStream2>,
}

impl Entry {
    /// The local that holds the value of the entry, once read, until every entry is read.
    pub fn slot(&self) -> Ident {
        format_ident!("{}_slot", self.key)
    }
}

/// The code that reads `key = value, ...` into an `Option` slot per entry ([`Entry::slot`]): the
/// declarations of the slots, and the closure that `parse_entries` and `parse_dict` take. A key
/// given twice is an error, and so is a key no entry has, unless `unknown` reads it, with `key`,
/// `c` and `at` in scope. `owner`, an expression, and `kind` name them in the errors:
/// ``tile.block has no attribute `x` ``, ``duplicate attribute `x` ``.
pub(crate) fn read_entries(
    entries: &[Entry],
    owner: &TokenStream2,
    kind: &str,
    unknown: Option<TokenStream2>,
) -> (TokenStream2, TokenStream2) {
    let slots = entries.iter().map(|entry| {
        let (slot, ty) = (entry.slot(), &entry.ty);
        quote!(let mut #slot: ::std::option::Option<#ty> = ::std::option::Option::None;)
    });
    let arms = entries.iter().map(|entry| {
        let (slot, ty, key) = (entry.slot(), &entry.ty, &entry.key);
        quote! {
            #key => {
                if #slot.is_some() {
                    ::anyhow::bail!(c.error_at(at, format!(concat!("duplicate ", #kind, " `{}`"), key)));
                }
                #slot = ::std::option::Option::Some(<#ty as ::gpt_rs_ir::ir::Attr>::parse(c)?);
            }
        }
    });
    let unknown = unknown.unwrap_or_else(|| {
        quote! {
            ::anyhow::bail!(c.error_at(at, format!(concat!("{} has no ", #kind, " `{}`"), #owner, key)))
        }
    });
    let reader = quote! {
        |c: &mut ::gpt_rs_ir::text::Cursor<'_>, key: &str, at: usize| -> ::anyhow::Result<()> {
            match key {
                #(#arms)*
                _ => #unknown,
            }
            ::std::result::Result::Ok(())
        }
    };
    (quote!(#(#slots)*), reader)
}

/// The value of `entry` once every entry is read: what its slot holds, else its default, else
/// the error that it is missing.
pub(crate) fn entry_value(entry: &Entry, owner: &TokenStream2, kind: &str) -> TokenStream2 {
    let (slot, key) = (entry.slot(), &entry.key);
    match &entry.default {
        Some(default) => quote!(#slot.unwrap_or_else(|| #default)),
        None => quote! {
            #slot.ok_or_else(|| c.error(format!(concat!("{} needs ", #kind, " `{}`"), #owner, #key)))?
        },
    }
}

/// `T` when `ty` is `Option<T>`.
pub(crate) fn option_inner(ty: &SynType) -> Option<&SynType> {
    let SynType::Path(path) = ty else {
        return None;
    };
    let segment = path.path.segments.last()?;
    if segment.ident != "Option" {
        return None;
    }
    let syn::PathArguments::AngleBracketed(args) = &segment.arguments else {
        return None;
    };
    match args.args.first()? {
        syn::GenericArgument::Type(inner) => Some(inner),
        _ => None,
    }
}

/// One element of a format.
pub(crate) enum Element {
    /// A literal token.
    Literal(String),
    /// No space between the elements around it.
    Glue,
    /// A field, the anchor of its optional group when `anchor`.
    Field { field: usize, anchor: bool },
    /// `shape($dims, $dtype)`.
    Shape { dims: usize, dtype: usize },
    /// `tuple($items)`.
    Tuple { field: usize, anchor: bool },
    /// `struct($a, ...)`, the anchor of its optional group when `anchor`.
    Struct { fields: Vec<usize>, anchor: bool },
    /// `(...)?`.
    Optional(Vec<Element>),
}

/// The code a format compiles to.
pub(crate) struct Compiled {
    /// Statements that print the fields, bound by reference to their bindings, into `out`.
    pub print: TokenStream2,
    /// Statements that read the fields from `c` into their bindings.
    pub parse: TokenStream2,
    /// Whether the next token of `c` starts the text.
    pub peek: TokenStream2,
    /// The literal the text starts with, when it starts with one.
    pub leading: Option<String>,
}

/// Compiles `format`, which writes `fields` of `owner`, the words that name the type or variant in
/// errors. A format that may write nothing is an error unless `may_be_empty`.
pub(crate) fn compile(
    format: &LitStr,
    fields: &[FormatField],
    owner: &str,
    may_be_empty: bool,
) -> syn::Result<Compiled> {
    let elements = parse_format(format, fields)?;
    compile_elements(&elements, fields, owner, format.span(), may_be_empty)
}

/// Compiles parsed `elements`, which write `fields` of `owner`.
pub(crate) fn compile_elements(
    elements: &[Element],
    fields: &[FormatField],
    owner: &str,
    span: Span,
    may_be_empty: bool,
) -> syn::Result<Compiled> {
    check(elements, fields, span)?;
    let (alternatives, nullable) = first(elements, fields);
    if nullable && !may_be_empty {
        return Err(syn::Error::new(
            span,
            "a variant's format starts with a token, which picks the variant",
        ));
    }
    let peek = if nullable {
        quote!(true)
    } else {
        quote!(#(#alternatives)||*)
    };
    let in_groups = grouped_fields(elements, fields);
    let mut declare = Vec::new();
    for (index, field) in fields.iter().enumerate() {
        if !in_groups.contains(&index) {
            continue;
        }
        let (binding, ty) = (&field.binding, &field.ty);
        let default = field
            .default_value()
            .expect("a field of a group has a default, as `check` makes sure");
        declare.push(quote!(let mut #binding: #ty = #default;));
    }
    let mut print = Vec::new();
    let mut parse = Vec::new();
    for element in elements {
        print.push(print_element(element, fields));
        parse.push(parse_element(element, fields, owner, &in_groups));
    }
    Ok(Compiled {
        leading: leading_literal(elements).map(str::to_string),
        print: quote! {
            let mut space = false;
            #(#print)*
        },
        parse: quote! {
            #(#declare)*
            #(#parse)*
        },
        peek,
    })
}

/// The literal `elements` start with, when they start with one rather than a field or a group.
fn leading_literal(elements: &[Element]) -> Option<&str> {
    match elements
        .iter()
        .find(|element| !matches!(element, Element::Glue))?
    {
        Element::Literal(text) => Some(text),
        _ => None,
    }
}

/// The fields that an optional group or a `struct(..)` may leave out, which the parser declares
/// with their defaults before it reads them.
fn grouped_fields(elements: &[Element], fields: &[FormatField]) -> Vec<usize> {
    let mut out = Vec::new();
    for element in elements {
        match element {
            Element::Optional(inner) => {
                collect_fields(inner, &mut out);
            }
            Element::Struct {
                fields: entries, ..
            } => out.extend(
                entries
                    .iter()
                    .copied()
                    .filter(|&index| fields[index].default_value().is_some()),
            ),
            _ => {}
        }
    }
    out
}

/// Every field `elements` write.
fn collect_fields(elements: &[Element], out: &mut Vec<usize>) {
    for element in elements {
        match element {
            Element::Literal(_) | Element::Glue => {}
            Element::Field { field, .. } | Element::Tuple { field, .. } => out.push(*field),
            Element::Shape { dims, dtype } => out.extend([*dims, *dtype]),
            Element::Struct { fields, .. } => out.extend(fields.iter().copied()),
            Element::Optional(inner) => collect_fields(inner, out),
        }
    }
}

// ------------------------------------------------------------------------------------------------
// Reading a format

struct FormatReader<'a> {
    chars: Vec<char>,
    pos: usize,
    span: Span,
    fields: &'a [FormatField],
}

fn parse_format(format: &LitStr, fields: &[FormatField]) -> syn::Result<Vec<Element>> {
    let mut reader = FormatReader {
        chars: format.value().chars().collect(),
        pos: 0,
        span: format.span(),
        fields,
    };
    let elements = reader.elements()?;
    if reader.pos < reader.chars.len() {
        return Err(reader.error("`)` closes no group"));
    }
    Ok(elements)
}

impl FormatReader<'_> {
    fn error(&self, msg: impl std::fmt::Display) -> syn::Error {
        syn::Error::new(self.span, format!("format: {msg}"))
    }

    fn skip_space(&mut self) {
        while self
            .chars
            .get(self.pos)
            .is_some_and(|character| character.is_whitespace())
        {
            self.pos += 1;
        }
    }

    fn eat(&mut self, expected: char) -> bool {
        if self.chars.get(self.pos) == Some(&expected) {
            self.pos += 1;
            return true;
        }
        false
    }

    fn word(&mut self) -> String {
        let start = self.pos;
        while self
            .chars
            .get(self.pos)
            .is_some_and(|character| character.is_ascii_alphanumeric() || *character == '_')
        {
            self.pos += 1;
        }
        self.chars[start..self.pos].iter().collect()
    }

    /// Elements up to the end of the format or the `)` of the group they are in.
    fn elements(&mut self) -> syn::Result<Vec<Element>> {
        let mut elements = Vec::new();
        loop {
            self.skip_space();
            let Some(&next) = self.chars.get(self.pos) else {
                return Ok(elements);
            };
            match next {
                ')' => return Ok(elements),
                '`' => elements.push(self.literal()?),
                '$' => {
                    let (field, anchor) = self.field()?;
                    elements.push(Element::Field { field, anchor });
                }
                '(' => {
                    self.pos += 1;
                    let inner = self.elements()?;
                    if !(self.eat(')') && self.eat('?')) {
                        return Err(self.error("a group is `(...)?`"));
                    }
                    elements.push(Element::Optional(inner));
                }
                letter if letter.is_ascii_alphabetic() => elements.push(self.directive()?),
                other => return Err(self.error(format!("unexpected `{other}`"))),
            }
        }
    }

    fn literal(&mut self) -> syn::Result<Element> {
        self.pos += 1;
        let start = self.pos;
        while self
            .chars
            .get(self.pos)
            .is_some_and(|character| *character != '`')
        {
            self.pos += 1;
        }
        if !self.eat('`') {
            return Err(self.error("a literal ends with a backtick"));
        }
        let text: String = self.chars[start..self.pos - 1].iter().collect();
        if text.is_empty() {
            return Ok(Element::Glue);
        }
        let lexed = token(&text, 0);
        let one_token = lexed.start == 0 && lexed.text == text;
        let literal_kind = matches!(
            lexed.kind,
            TokenKind::Ident | TokenKind::Punct | TokenKind::Hash | TokenKind::Bang
        );
        if !one_token || !literal_kind {
            return Err(self.error(format!(
                "`{text}` is no literal: a literal is one keyword, name with its sigil or \
                 punctuation token"
            )));
        }
        Ok(Element::Literal(text))
    }

    /// `$name` or `$name^`.
    fn field(&mut self) -> syn::Result<(usize, bool)> {
        if !self.eat('$') {
            return Err(self.error("expected `$` and a field"));
        }
        let name = self.word();
        let field = self
            .fields
            .iter()
            .position(|field| field.name == name)
            .ok_or_else(|| self.error(format!("there is no field `{name}`")))?;
        Ok((field, self.eat('^')))
    }

    /// `shape($dims, $dtype)`, `tuple($items)` or `struct($a, ...)`.
    fn directive(&mut self) -> syn::Result<Element> {
        let name = self.word();
        if !self.eat('(') {
            return Err(self.error(format!("expected `(` after `{name}`")));
        }
        let mut args = Vec::new();
        let mut anchor = false;
        loop {
            self.skip_space();
            let (field, marked) = self.field()?;
            if marked {
                return Err(self.error("`^` marks a directive, after its `)`"));
            }
            args.push(field);
            self.skip_space();
            if self.eat(')') {
                break;
            }
            if !self.eat(',') {
                return Err(self.error("expected `,` or `)`"));
            }
        }
        if self.eat('^') {
            anchor = true;
        }
        match (name.as_str(), args.as_slice()) {
            ("shape", &[dims, dtype]) if !anchor => Ok(Element::Shape { dims, dtype }),
            ("tuple", &[field]) => Ok(Element::Tuple { field, anchor }),
            ("struct", _) => Ok(Element::Struct {
                fields: args,
                anchor,
            }),
            _ => Err(self.error(format!(
                "unknown directive `{name}` with {} fields: the directives are shape($dims, \
                 $dtype), tuple($items) and struct($a, ...)",
                args.len()
            ))),
        }
    }
}

// ------------------------------------------------------------------------------------------------
// Checking a format

/// Checks that each field is written once, and that each optional group has one anchor, starts
/// with a token and writes only fields with defaults.
fn check(elements: &[Element], fields: &[FormatField], span: Span) -> syn::Result<()> {
    let error = |msg: String| syn::Error::new(span, format!("format: {msg}"));
    let mut written = Vec::new();
    collect_fields(elements, &mut written);
    for (index, field) in fields.iter().enumerate() {
        let count = written.iter().filter(|&&used| used == index).count();
        if count != 1 {
            return Err(error(format!(
                "it writes the field `{}` {count} times, not once",
                field.name
            )));
        }
    }
    check_structs(elements, &error)?;
    for (index, element) in elements.iter().enumerate() {
        match element {
            Element::Field { anchor: true, .. }
            | Element::Tuple { anchor: true, .. }
            | Element::Struct { anchor: true, .. } => {
                return Err(error("`^` marks the anchor of an optional group".into()));
            }
            Element::Optional(inner) => {
                check_group(inner, fields, &error)?;
                // The token after the group must tell that the group is not there.
                let after = leading_literal(&elements[index + 1..]);
                if let (Some(first), Some(after)) = (leading_literal(inner), after) {
                    if first == after {
                        return Err(error(format!(
                            "an optional group starts with `{first}`, which also follows it"
                        )));
                    }
                }
            }
            _ => {}
        }
    }
    Ok(())
}

/// Checks that a closing literal follows each `struct(..)`: its entries go on while a name or a
/// `,` comes next.
fn check_structs(elements: &[Element], error: &dyn Fn(String) -> syn::Error) -> syn::Result<()> {
    for (index, element) in elements.iter().enumerate() {
        if let Element::Optional(inner) = element {
            check_structs(inner, error)?;
        }
        if !matches!(element, Element::Struct { .. }) {
            continue;
        }
        let closed = matches!(
            elements.get(index + 1),
            Some(Element::Literal(text)) if matches!(text.as_str(), ">" | "}" | ")" | "]")
        );
        if !closed {
            return Err(error(
                "`struct(..)` is followed by a closing `>`, `}`, `)` or `]`".into(),
            ));
        }
    }
    Ok(())
}

fn check_group(
    elements: &[Element],
    fields: &[FormatField],
    error: &dyn Fn(String) -> syn::Error,
) -> syn::Result<()> {
    let anchors = elements
        .iter()
        .filter(|element| {
            matches!(
                element,
                Element::Field { anchor: true, .. }
                    | Element::Tuple { anchor: true, .. }
                    | Element::Struct { anchor: true, .. }
            )
        })
        .count();
    if anchors != 1 {
        return Err(error(format!(
            "an optional group has one anchor, marked `^`, not {anchors}"
        )));
    }
    let leading = elements
        .iter()
        .find(|element| !matches!(element, Element::Glue));
    if !matches!(
        leading,
        Some(Element::Literal(_) | Element::Field { .. } | Element::Tuple { .. })
    ) {
        return Err(error(
            "an optional group starts with a literal or a field, whose token says it is there"
                .into(),
        ));
    }
    let mut written = Vec::new();
    collect_fields(elements, &mut written);
    for index in written {
        if fields[index].default_value().is_none() {
            return Err(error(format!(
                "an optional group may leave out `{}`, which has no default",
                fields[index].name
            )));
        }
    }
    if elements
        .iter()
        .any(|element| matches!(element, Element::Optional(_) | Element::Shape { .. }))
    {
        return Err(error(
            "an optional group holds no group and no shape".into(),
        ));
    }
    Ok(())
}

// ------------------------------------------------------------------------------------------------
// Generating code

/// Whether the printer puts no space before the literal `text`.
fn no_space_before(text: &str) -> bool {
    matches!(text, ")" | "]" | ">" | "}" | "," | "(" | "[" | "<" | "..")
}

/// Whether the printer puts no space after the literal `text`.
fn no_space_after(text: &str) -> bool {
    matches!(text, "(" | "[" | "<" | "{" | "..")
}

/// The tests that the next token starts `elements`, any of which holds, and whether they may
/// write nothing.
fn first(elements: &[Element], fields: &[FormatField]) -> (Vec<TokenStream2>, bool) {
    let mut alternatives = Vec::new();
    for element in elements {
        let test = match element {
            Element::Glue => continue,
            Element::Optional(inner) => {
                alternatives.extend(first(inner, fields).0);
                continue;
            }
            Element::Struct { .. } => {
                alternatives.push(quote!(c.peek_kind(::gpt_rs_ir::text::TokenKind::Ident)));
                continue;
            }
            Element::Literal(text) => quote!(c.peek_is(#text)),
            Element::Field { field, anchor } => {
                let written = &fields[*field];
                let ty = match (anchor, written.option_inner()) {
                    (true, Some(inner)) => inner,
                    _ => &written.ty,
                };
                quote!(<#ty as ::gpt_rs_ir::ir::Attr>::peek(c))
            }
            Element::Tuple { field, .. } => {
                let ty = &fields[*field].ty;
                quote!(::gpt_rs_ir::ir::peek_tuple::<#ty>(c))
            }
            Element::Shape { .. } => quote!(::gpt_rs_ir::ir::peek_shape(c)),
        };
        alternatives.push(test);
        return (alternatives, false);
    }
    (alternatives, true)
}

/// The test that the anchor of the group `elements` holds a value its text writes.
fn anchor_present(elements: &[Element], fields: &[FormatField]) -> TokenStream2 {
    let not_default = |index: usize| {
        let field = &fields[index];
        let binding = &field.binding;
        if field.default.is_none() && field.option_inner().is_some() {
            return quote!(#binding.is_some());
        }
        let default = field.default_value().expect("an anchor has a default");
        quote!(*#binding != #default)
    };
    for element in elements {
        match element {
            Element::Field {
                field,
                anchor: true,
            }
            | Element::Tuple {
                field,
                anchor: true,
            } => return not_default(*field),
            Element::Struct {
                fields: entries,
                anchor: true,
            } => {
                let tests = entries.iter().map(|&index| not_default(index));
                return quote!(false #(|| #tests)*);
            }
            _ => {}
        }
    }
    unreachable!("`check` gives each group one anchor")
}

fn print_element(element: &Element, fields: &[FormatField]) -> TokenStream2 {
    let space_before = quote! {
        if space {
            out.push(' ');
        }
    };
    match element {
        Element::Literal(text) => {
            let before = if no_space_before(text) {
                quote!()
            } else {
                space_before
            };
            let after = !no_space_after(text);
            quote! {
                #before
                out.push_str(#text);
                space = #after;
            }
        }
        Element::Glue => quote!(space = false;),
        Element::Field { field, anchor } => {
            let written = &fields[*field];
            let (binding, ty) = (&written.binding, &written.ty);
            let print = match (anchor, written.option_inner()) {
                (true, Some(inner)) => quote! {
                    if let ::std::option::Option::Some(value) = #binding {
                        <#inner as ::gpt_rs_ir::ir::Attr>::print(value, out);
                    }
                },
                _ => quote!(<#ty as ::gpt_rs_ir::ir::Attr>::print(#binding, out);),
            };
            quote! {
                #space_before
                #print
                space = true;
            }
        }
        Element::Tuple { field, .. } => {
            let binding = &fields[*field].binding;
            quote! {
                #space_before
                ::gpt_rs_ir::ir::print_tuple(::gpt_rs_ir::ir::AttrList::items(#binding), out);
                space = true;
            }
        }
        Element::Shape { dims, dtype } => {
            let (dims, dtype) = (&fields[*dims].binding, &fields[*dtype].binding);
            quote! {
                #space_before
                ::gpt_rs_ir::ir::print_shape(&#dims[..], #dtype, out);
                space = true;
            }
        }
        Element::Struct {
            fields: entries, ..
        } => {
            let entries = entries.iter().map(|&index| {
                let field = &fields[index];
                let (binding, ty, key) = (&field.binding, &field.ty, &field.name);
                let present = match field.default_value() {
                    Some(default) => quote!(*#binding != #default),
                    None => quote!(true),
                };
                quote! {
                    if #present {
                        if !first {
                            out.push_str(", ");
                        } else if space {
                            out.push(' ');
                        }
                        out.push_str(concat!(#key, " = "));
                        <#ty as ::gpt_rs_ir::ir::Attr>::print(#binding, out);
                        first = false;
                    }
                }
            });
            quote! {
                let mut first = true;
                #(#entries)*
                if !first {
                    space = true;
                }
            }
        }
        Element::Optional(inner) => {
            let present = anchor_present(inner, fields);
            let prints = inner.iter().map(|element| print_element(element, fields));
            quote! {
                if #present {
                    #(#prints)*
                }
            }
        }
    }
}

fn parse_element(
    element: &Element,
    fields: &[FormatField],
    owner: &str,
    declared: &[usize],
) -> TokenStream2 {
    // A field the parser declared before gets its value; any other is bound where it is read.
    let bind = |index: usize, value: TokenStream2| {
        let field = &fields[index];
        let (binding, ty) = (&field.binding, &field.ty);
        if declared.contains(&index) {
            quote!(#binding = #value;)
        } else {
            quote!(let #binding: #ty = #value;)
        }
    };
    match element {
        Element::Literal(text) => quote!(c.expect(#text)?;),
        Element::Glue => quote!(),
        Element::Field { field, anchor } => {
            let ty = &fields[*field].ty;
            let value = match (anchor, fields[*field].option_inner()) {
                (true, Some(inner)) => quote! {
                    ::std::option::Option::Some(<#inner as ::gpt_rs_ir::ir::Attr>::parse(c)?)
                },
                _ => quote!(<#ty as ::gpt_rs_ir::ir::Attr>::parse(c)?),
            };
            bind(*field, value)
        }
        Element::Tuple { field, .. } => {
            let ty = &fields[*field].ty;
            bind(*field, quote!(::gpt_rs_ir::ir::parse_tuple::<#ty>(c)?))
        }
        Element::Shape { dims, dtype } => {
            let dims_value = bind(*dims, quote!(::std::convert::Into::into(shape.0)));
            let dtype_value = bind(*dtype, quote!(::std::convert::Into::into(shape.1)));
            quote! {
                let shape = ::gpt_rs_ir::ir::parse_shape(c)?;
                #dims_value
                #dtype_value
            }
        }
        Element::Struct {
            fields: written, ..
        } => {
            let mut entries = Vec::with_capacity(written.len());
            for &index in written {
                let field = &fields[index];
                entries.push(Entry {
                    key: field.name.clone(),
                    ty: field.ty.clone(),
                    default: field.default_value(),
                });
            }
            let owner = quote!(#owner);
            let (slots, reader) = read_entries(&entries, &owner, "parameter", None);
            let values = written
                .iter()
                .zip(&entries)
                .map(|(&index, entry)| bind(index, entry_value(entry, &owner, "parameter")));
            quote! {
                #slots
                ::gpt_rs_ir::ir::parse_entries(c, #reader)?;
                #(#values)*
            }
        }
        Element::Optional(inner) => {
            let (alternatives, _) = first(inner, fields);
            let parses = inner
                .iter()
                .map(|element| parse_element(element, fields, owner, declared));
            quote! {
                if #(#alternatives)||* {
                    #(#parses)*
                }
            }
        }
    }
}
