//! Derives and attributes for `gpt-rs-ir`.
//!
//! - `#[op("dialect.name", ...)]` on an op struct. It derives the op's standard traits, generates
//!   its named operand, result and region group structs and implements `OpKind` and `OpDyn`. An
//!   op with `family(F, Item(value), ..)` takes its groups, effects and traits from the family
//!   `F`, its type rule and semantics from the family's, and sets the family's associated items.
//!   Every argument is meta syntax (`name(..)`, paths, literals), so rustfmt formats it.
//! - `#[family(...)]` on the trait of an op family declares the groups, effects and traits its ops
//!   share, its type rule and semantics, and generates the group structs. A hidden macro of the
//!   trait's name hands them to the `#[op]` of each op of the family (`family_op!`).
//! - `#[derive(AttrGroup)]` on the struct of attributes that the ops of a family share.
//! - `#[derive(Dialect)]` with `#[dialect(name = "...", op = Variant)]` on the enum listing a
//!   dialect's ops, each variant holding its op in place or boxed (`Block(Box<Block>)`). It
//!   generates the dispatch over its ops and their place in the `Op` union, and it sets the enum's
//!   `DIALECT` constant to `name`.
//! - `#[derive(DialectSet)]` with `#[dialect_set(kind(Kind))]` on the `Op` union of the dialects.
//!   It generates the dispatch over the dialects and `Kind`, the enum that names them.
//! - `#[derive(Attr)]` on a fieldless enum: one snake_case text name per variant.
//! - `#[pass(name = "...", stats(...))]` on a pass struct. It gives the pass its name and
//!   statistics, makes its fields options, and registers it with its doc comment.
//! - `#[pattern]` on a pattern struct records its name and doc comment. `#[pattern(canonical)]`
//!   also registers it with `canonicalize`.
//! - `#[fold]` on `impl Fold for <op>` registers the op's folder with its doc comment.
//! - `#[cost]` on `impl OpCost for <op>` registers the op's cost.
//! - `#[derive(Doc)]` on an op trait struct records its name and doc comment.
//! - `matched!(let <ops> = value else { .. })` binds the ops that define a value and its operands.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote, ToTokens};
use syn::ext::IdentExt;
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::{
    parse_macro_input, Data, DeriveInput, Expr, Fields, Ident, LitStr, Token, Type as SynType,
};

fn snake_case(name: &str) -> String {
    let mut out = String::new();
    for (i, ch) in name.chars().enumerate() {
        if ch.is_ascii_uppercase() {
            if i > 0 {
                out.push('_');
            }
            out.push(ch.to_ascii_lowercase());
        } else {
            out.push(ch);
        }
    }
    out
}

fn one_attr<'a>(
    input: &'a DeriveInput,
    name: &str,
    missing: &str,
) -> syn::Result<&'a syn::Attribute> {
    let mut attrs = input.attrs.iter().filter(|a| a.path().is_ident(name));
    let Some(attr) = attrs.next() else {
        return Err(syn::Error::new_spanned(input, missing));
    };
    if let Some(extra) = attrs.next() {
        return Err(syn::Error::new_spanned(
            extra,
            format!("duplicate #[{name}] attribute"),
        ));
    }
    Ok(attr)
}

// ------------------------------------------------------------------------------------------------
// #[derive(Attr)]

#[proc_macro_derive(Attr, attributes(attr))]
pub fn derive_attr(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match expand_attr(&input) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

fn expand_attr(input: &DeriveInput) -> syn::Result<TokenStream2> {
    let Data::Enum(data) = &input.data else {
        return Err(syn::Error::new_spanned(
            input,
            "#[derive(Attr)] supports enums",
        ));
    };
    if data
        .variants
        .iter()
        .any(|v| !matches!(v.fields, Fields::Unit))
    {
        return expand_attr_with_fields(input, data);
    }
    let name = &input.ident;
    let mut variants = Vec::new();
    let mut names = Vec::new();
    let mut docs = Vec::new();
    for variant in &data.variants {
        variants.push(variant.ident.clone());
        names.push(snake_case(&variant.ident.to_string()));
        docs.push(doc_text(&variant.attrs));
    }
    let type_name = name.to_string();
    let entry = format_ident!("__GPT_RS_ATTR_ENUM_{}", type_name.to_uppercase());
    Ok(quote! {
        impl ::gpt_rs_ir::attr::Mnemonic for #name {
            const ALL: &'static [Self] = &[#(Self::#variants),*];
            fn mnemonic(self) -> &'static str {
                match self { #(Self::#variants => #names),* }
            }
            fn doc(self) -> &'static str {
                match self { #(Self::#variants => #docs),* }
            }
        }

        impl ::gpt_rs_ir::attr::Attr for #name {
            fn print(&self, out: &mut ::std::string::String) {
                out.push_str(::gpt_rs_ir::attr::Mnemonic::mnemonic(*self));
            }
            fn parse(c: &mut ::gpt_rs_ir::text::Cursor<'_>) -> ::anyhow::Result<Self> {
                let name = c.ident()?;
                <Self as ::gpt_rs_ir::attr::Mnemonic>::from_mnemonic(name)
                    .ok_or_else(|| c.error(format!("unknown {} `{}`", #type_name, name)))
            }
        }

        #[::gpt_rs_ir::linkme::distributed_slice(::gpt_rs_ir::attr::ATTR_ENUMS)]
        #[linkme(crate = ::gpt_rs_ir::linkme)]
        static #entry: ::gpt_rs_ir::attr::AttrEnum = ::gpt_rs_ir::attr::AttrEnum {
            name: #type_name,
            values: ::gpt_rs_ir::attr::enum_values::<#name>,
        };
    })
}

/// `#[derive(Attr)]` on an enum whose variants have named fields, or hold one attribute group: a
/// variant is written by its snake-case name, with its fields or the group's attributes after it,
/// `top_k<k = 3>`. A field with `#[attr(default)]` may be left out, and the printer leaves it out
/// when it holds its default. The enum gets `name()`, the name of its variant, and `Display`, its
/// text form.
fn expand_attr_with_fields(input: &DeriveInput, data: &syn::DataEnum) -> syn::Result<TokenStream2> {
    let name = &input.ident;
    let type_name = name.to_string();
    let mut name_arms = Vec::new();
    let mut print_arms = Vec::new();
    let mut parse_arms = Vec::new();
    for variant in &data.variants {
        let ident = &variant.ident;
        let text = snake_case(&ident.to_string());
        let fields = match &variant.fields {
            Fields::Unit => Vec::new(),
            Fields::Named(named) => named
                .named
                .iter()
                .map(field_spec)
                .collect::<syn::Result<Vec<_>>>()?,
            // A variant that holds an attribute group writes the group's attributes as its own.
            Fields::Unnamed(unnamed) if unnamed.unnamed.len() == 1 => {
                let group = &unnamed.unnamed[0].ty;
                name_arms.push(quote! { Self::#ident(_) => #text });
                print_arms.push(quote! {
                    Self::#ident(group) => {
                        let mut attrs = ::std::vec::Vec::new();
                        ::gpt_rs_ir::attr::AttrGroup::print_attrs(group, &mut attrs);
                        for (key, value) in attrs {
                            params.push(format!("{} = {}", key, value));
                        }
                    }
                });
                parse_arms.push(quote! {
                    #text => Ok(Self::#ident(<#group as ::gpt_rs_ir::attr::AttrGroup>::parse_params(#text, c)?)),
                });
                continue;
            }
            Fields::Unnamed(_) => {
                return Err(syn::Error::new_spanned(
                    variant,
                    "#[derive(Attr)] supports unit variants, variants with named fields, and \
                     variants that hold one attribute group",
                ))
            }
        };
        let idents: Vec<&Ident> = fields.iter().map(|f| &f.ident).collect();
        let keys: Vec<&String> = fields.iter().map(|f| &f.key).collect();
        let types: Vec<&SynType> = fields.iter().map(|f| &f.ty).collect();
        name_arms.push(quote! { Self::#ident { .. } => #text });
        let prints = fields.iter().map(|f| {
            let (ident, key) = (&f.ident, &f.key);
            let print = quote! {
                let mut value = ::std::string::String::new();
                ::gpt_rs_ir::attr::Attr::print(#ident, &mut value);
                params.push(format!("{} = {}", #key, value));
            };
            if f.default {
                quote! { if *#ident != ::std::default::Default::default() { #print } }
            } else {
                print
            }
        });
        print_arms.push(quote! {
            Self::#ident { #(#idents),* } => { #(#prints)* }
        });
        let finish = fields.iter().map(|f| {
            let (ident, key) = (&f.ident, &f.key);
            if f.default {
                quote! { #ident: #ident.unwrap_or_default() }
            } else {
                quote! {
                    #ident: #ident.ok_or_else(|| c.error(format!("{} needs {}", #text, #key)))?
                }
            }
        });
        parse_arms.push(quote! {
            #text => {
                #(let mut #idents: ::std::option::Option<#types> = None;)*
                if c.eat("<") {
                    c.list(">", |c| -> ::anyhow::Result<()> {
                        let key = c.ident()?;
                        c.expect("=")?;
                        match key {
                            #(#keys => {
                                #idents = Some(<#types as ::gpt_rs_ir::attr::Attr>::parse(c)?);
                                Ok(())
                            })*
                            other => Err(c.error(format!("{} has no parameter `{}`", #text, other))),
                        }
                    })?;
                }
                Ok(Self::#ident { #(#finish),* })
            }
        });
    }
    Ok(quote! {
        impl #name {
            /// The name of the variant in the text form, without its fields.
            pub fn name(&self) -> &'static str {
                match self { #(#name_arms),* }
            }
        }

        impl ::std::fmt::Display for #name {
            fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
                let mut text = ::std::string::String::new();
                ::gpt_rs_ir::attr::Attr::print(self, &mut text);
                f.write_str(&text)
            }
        }

        impl ::gpt_rs_ir::attr::Attr for #name {
            fn print(&self, out: &mut ::std::string::String) {
                out.push_str(self.name());
                let mut params: ::std::vec::Vec<::std::string::String> = ::std::vec::Vec::new();
                match self { #(#print_arms)* }
                if !params.is_empty() {
                    out.push('<');
                    out.push_str(&params.join(", "));
                    out.push('>');
                }
            }

            fn parse(c: &mut ::gpt_rs_ir::text::Cursor<'_>) -> ::anyhow::Result<Self> {
                let name = c.ident()?;
                match name {
                    #(#parse_arms)*
                    other => Err(c.error(format!("unknown {} `{}`", #type_name, other))),
                }
            }
        }
    })
}

/// A `'static` slice of the trait values `exprs`, each a constant expression.
fn trait_slice(exprs: &[Expr]) -> TokenStream2 {
    quote! {{
        const TRAITS: &[&dyn ::gpt_rs_ir::traits::OpTrait] = &[#(&#exprs),*];
        TRAITS
    }}
}

// ------------------------------------------------------------------------------------------------
// #[op]

#[derive(Clone, Copy, PartialEq)]
enum KindSpec {
    Any,
    Tensor,
    Index,
    Scalar,
}

#[derive(Clone, Copy, PartialEq)]
enum AritySpec {
    One,
    Optional,
    Variadic,
}

#[derive(Clone)]
struct GroupSpec {
    name: Ident,
    kind: KindSpec,
    arity: AritySpec,
}

#[derive(Clone)]
struct RegionSpec {
    name: Ident,
    terminator: SynType,
}

#[derive(Clone, Default)]
struct EffectSpec {
    opaque: bool,
    rng: bool,
    alloc: bool,
    reads: Vec<Ident>,
    writes: Vec<Ident>,
}

impl EffectSpec {
    fn is_none(&self) -> bool {
        !self.opaque && !self.rng && !self.alloc && self.reads.is_empty() && self.writes.is_empty()
    }
}

struct OpArgs {
    name: LitStr,
    family: Option<FamilyRef>,
    operands: Vec<GroupSpec>,
    results: Vec<GroupSpec>,
    regions: Vec<RegionSpec>,
    effects: EffectSpec,
    traits: Vec<Expr>,
    /// The op is a `ConstantOp`.
    constant: bool,
    /// The op is a `SymbolOp`.
    symbol: bool,
}

fn parse_kind(input: ParseStream) -> syn::Result<(KindSpec, AritySpec)> {
    let ident: Ident = input.parse()?;
    let arity = match ident.to_string().as_str() {
        "Variadic" => Some(AritySpec::Variadic),
        "Optional" => Some(AritySpec::Optional),
        _ => None,
    };
    if let Some(arity) = arity {
        let held;
        syn::parenthesized!(held in input);
        let (kind, nested) = parse_kind(&held)?;
        if nested != AritySpec::One {
            return Err(syn::Error::new(
                ident.span(),
                format!("`{ident}` cannot hold another Variadic or Optional"),
            ));
        }
        return Ok((kind, arity));
    }
    let kind = match ident.to_string().as_str() {
        "Any" => KindSpec::Any,
        "Tensor" => KindSpec::Tensor,
        "Index" => KindSpec::Index,
        "Scalar" => KindSpec::Scalar,
        other => {
            return Err(syn::Error::new(
                ident.span(),
                format!(
                    "unknown kind `{other}`, expected Any, Tensor, Index, Scalar, \
                     Optional(..) or Variadic(..)"
                ),
            ))
        }
    };
    Ok((kind, AritySpec::One))
}

/// `family(Unary, Scalar(arith::Neg))` of an op: its family, and the family's associated items as
/// the op sets them.
struct FamilyRef {
    ident: Ident,
    items: Vec<(Ident, TokenStream2)>,
}

impl Parse for FamilyRef {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let ident = input.parse()?;
        let mut items = Vec::new();
        while input.parse::<Option<Token![,]>>()?.is_some() && !input.is_empty() {
            let key: Ident = input.parse()?;
            // A type or an expression, as the family declares the item.
            let value;
            syn::parenthesized!(value in input);
            items.push((key, value.parse()?));
        }
        Ok(FamilyRef { ident, items })
    }
}

fn parse_list<T>(
    input: ParseStream,
    item: fn(ParseStream) -> syn::Result<T>,
) -> syn::Result<Vec<T>> {
    let content;
    syn::parenthesized!(content in input);
    Ok(content
        .parse_terminated(item, Token![,])?
        .into_iter()
        .collect())
}

/// A group: `lhs(Tensor)`, `ys(Variadic(Tensor))`.
fn parse_group(input: ParseStream) -> syn::Result<GroupSpec> {
    let name: Ident = input.parse()?;
    let spec;
    syn::parenthesized!(spec in input);
    let (kind, arity) = parse_kind(&spec)?;
    Ok(GroupSpec { name, kind, arity })
}

/// A region: `body(Block(Yield))`, with its terminator.
fn parse_region(input: ParseStream) -> syn::Result<RegionSpec> {
    let name: Ident = input.parse()?;
    let spec;
    syn::parenthesized!(spec in input);
    let block: Ident = spec.parse()?;
    if block != "Block" {
        return Err(syn::Error::new(block.span(), "expected Block(Terminator)"));
    }
    let terminator;
    syn::parenthesized!(terminator in spec);
    Ok(RegionSpec {
        name,
        terminator: terminator.parse()?,
    })
}

fn parse_effect(input: ParseStream) -> syn::Result<(Ident, Option<Ident>)> {
    let name: Ident = input.parse()?;
    if name != "read" && name != "write" {
        return Ok((name, None));
    }
    let group;
    syn::parenthesized!(group in input);
    Ok((name, Some(group.parse()?)))
}

fn parse_effects(input: ParseStream) -> syn::Result<EffectSpec> {
    let mut effects = EffectSpec::default();
    for (name, group) in parse_list(input, parse_effect)? {
        match (name.to_string().as_str(), group) {
            ("opaque", _) => effects.opaque = true,
            ("rng", _) => effects.rng = true,
            ("alloc", _) => effects.alloc = true,
            ("read", Some(group)) => effects.reads.push(group),
            ("write", Some(group)) => effects.writes.push(group),
            (other, _) => {
                return Err(syn::Error::new(
                    name.span(),
                    format!(
                    "unknown effect `{other}`, expected opaque, rng, alloc, read(..) or write(..)"
                ),
                ))
            }
        }
    }
    Ok(effects)
}

impl Parse for OpArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut args = OpArgs {
            name: input.parse()?,
            family: None,
            operands: Vec::new(),
            results: Vec::new(),
            regions: Vec::new(),
            effects: EffectSpec::default(),
            traits: Vec::new(),
            constant: false,
            symbol: false,
        };
        let mut seen = Vec::new();
        while !input.is_empty() {
            input.parse::<Token![,]>()?;
            if input.is_empty() {
                break;
            }
            let key: Ident = input.parse()?;
            if seen.contains(&key) {
                return Err(syn::Error::new(
                    key.span(),
                    format!("duplicate #[op] clause `{key}`"),
                ));
            }
            match key.to_string().as_str() {
                "family" => {
                    let family;
                    syn::parenthesized!(family in input);
                    args.family = Some(family.parse()?);
                }
                "operands" => args.operands = parse_list(input, parse_group)?,
                "results" => args.results = parse_list(input, parse_group)?,
                "regions" => args.regions = parse_list(input, parse_region)?,
                "effects" => args.effects = parse_effects(input)?,
                "traits" => args.traits = parse_list(input, Expr::parse)?,
                "constant" => args.constant = true,
                "symbol" => args.symbol = true,
                other => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!(
                            "unknown #[op] clause `{other}`, expected family, operands, \
                             results, regions, effects, traits, constant or symbol"
                        ),
                    ))
                }
            }
            seen.push(key);
        }
        Ok(args)
    }
}

struct FieldSpec {
    ident: Ident,
    key: String,
    ty: SynType,
    default: bool,
}

fn field_spec(field: &syn::Field) -> syn::Result<FieldSpec> {
    let mut default = false;
    for attr in field.attrs.iter().filter(|a| a.path().is_ident("attr")) {
        attr.parse_nested_meta(|meta| {
            if !meta.path.is_ident("default") {
                return Err(meta.error("expected #[attr(default)]"));
            }
            default = true;
            Ok(())
        })?;
    }
    let Some(ident) = field.ident.clone() else {
        return Err(syn::Error::new_spanned(
            field,
            "op structs have named fields or none",
        ));
    };
    Ok(FieldSpec {
        key: ident.unraw().to_string(),
        ident,
        ty: field.ty.clone(),
        default,
    })
}

/// `#[op("dialect.name", operands(..), results(..), regions(..), effects(..), traits(..))]` on the
/// struct of an op, or `#[op("dialect.name", family(Family, Item(value), ..), traits(..))]` on an op
/// of a family: declares the op, its derives and everything the dialect reads of it. A group is
/// `x(Tensor)`, `ys(Variadic(Tensor))` or `z(Optional(Tensor))`, and a region `body(Block(Yield))`.
#[proc_macro_attribute]
pub fn op(args: TokenStream, input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    expand_op(args.into(), input)
        .unwrap_or_else(|err| err.to_compile_error())
        .into()
}

fn kind_tokens(kind: KindSpec) -> TokenStream2 {
    match kind {
        KindSpec::Any => quote!(::gpt_rs_ir::op::Kind::Any),
        KindSpec::Tensor => quote!(::gpt_rs_ir::op::Kind::Tensor),
        KindSpec::Index => quote!(::gpt_rs_ir::op::Kind::Index),
        KindSpec::Scalar => quote!(::gpt_rs_ir::op::Kind::Scalar),
    }
}

/// The element type of a group under view `V`.
fn element_type(kind: KindSpec) -> TokenStream2 {
    match kind {
        KindSpec::Any => quote!(<V as ::gpt_rs_ir::op::View>::Any),
        KindSpec::Tensor => quote!(<V as ::gpt_rs_ir::op::View>::Tensor),
        KindSpec::Index => quote!(<V as ::gpt_rs_ir::op::View>::Index),
        KindSpec::Scalar => quote!(<V as ::gpt_rs_ir::op::View>::Scalar),
    }
}

/// Projects `V::Any` to the group's element type.
fn project(kind: KindSpec) -> TokenStream2 {
    match kind {
        KindSpec::Any => quote!(::std::option::Option::Some),
        KindSpec::Tensor => quote!(<V as ::gpt_rs_ir::op::View>::tensor),
        KindSpec::Index => quote!(<V as ::gpt_rs_ir::op::View>::index),
        KindSpec::Scalar => quote!(<V as ::gpt_rs_ir::op::View>::scalar),
    }
}

/// Embeds the group's element type back into `V::Any`.
fn embed(kind: KindSpec) -> TokenStream2 {
    match kind {
        KindSpec::Any => quote!((|x| x)),
        KindSpec::Tensor => quote!(<V as ::gpt_rs_ir::op::Embed>::from_tensor),
        KindSpec::Index => quote!(<V as ::gpt_rs_ir::op::Embed>::from_index),
        KindSpec::Scalar => quote!(<V as ::gpt_rs_ir::op::Embed>::from_scalar),
    }
}

struct GroupCode {
    struct_def: TokenStream2,
    consts: TokenStream2,
    split: TokenStream2,
    flatten: TokenStream2,
}

fn group_code(
    vis: &syn::Visibility,
    struct_name: &Ident,
    groups: &[GroupSpec],
    consts_name: &Ident,
    what: &str,
) -> GroupCode {
    let fields: Vec<TokenStream2> = groups
        .iter()
        .map(|g| {
            let name = &g.name;
            let elem = element_type(g.kind);
            match g.arity {
                AritySpec::One => quote!(pub #name: #elem),
                AritySpec::Optional => quote!(pub #name: ::std::option::Option<#elem>),
                AritySpec::Variadic => quote!(pub #name: ::std::vec::Vec<#elem>),
            }
        })
        .collect();
    let phantom = if groups.is_empty() {
        quote!(_view: ::std::marker::PhantomData<V>,)
    } else {
        quote!()
    };
    let struct_def = quote! {
        #vis struct #struct_name<V: ::gpt_rs_ir::op::View> {
            #(#fields,)*
            #phantom
        }
    };

    let group_entries = groups.iter().map(|g| {
        let name = g.name.unraw().to_string();
        let kind = kind_tokens(g.kind);
        let arity = match g.arity {
            AritySpec::One => quote!(::gpt_rs_ir::op::Arity::One),
            AritySpec::Optional => quote!(::gpt_rs_ir::op::Arity::Optional),
            AritySpec::Variadic => quote!(::gpt_rs_ir::op::Arity::Variadic),
        };
        quote!(::gpt_rs_ir::op::Group { name: #name, kind: #kind, arity: #arity })
    });
    let consts =
        quote!(const #consts_name: &'static [::gpt_rs_ir::op::Group] = &[#(#group_entries),*];);

    let takes = groups.iter().enumerate().map(|(i, g)| {
        let name = &g.name;
        let proj = project(g.kind);
        match g.arity {
            AritySpec::One => quote!(#name: ::gpt_rs_ir::op::take_one(&mut items, #proj, &Self::#consts_name[#i], #what)?),
            AritySpec::Optional => quote!(#name: ::gpt_rs_ir::op::take_optional(&mut items, sizes[#i], #proj, &Self::#consts_name[#i], #what)?),
            AritySpec::Variadic => quote!(#name: ::gpt_rs_ir::op::take_many(&mut items, sizes[#i], #proj, &Self::#consts_name[#i], #what)?),
        }
    });
    let phantom_init = if groups.is_empty() {
        quote!(_view: ::std::marker::PhantomData,)
    } else {
        quote!()
    };
    // Groups of one value each need no sizes: only their count is checked, as `group_sizes`
    // checks it.
    let sizes = if groups.iter().all(|g| g.arity == AritySpec::One) {
        let count = groups.len();
        quote! {
            match segments {
                ::std::option::Option::Some(_) => {
                    ::gpt_rs_ir::op::group_sizes(Self::#consts_name, flat.len(), segments, #what)?;
                }
                ::std::option::Option::None => ::anyhow::ensure!(
                    flat.len() == #count,
                    "expected {} {}s, got {}",
                    #count,
                    #what,
                    flat.len()
                ),
            }
        }
    } else {
        quote! {
            let sizes = ::gpt_rs_ir::op::group_sizes(Self::#consts_name, flat.len(), segments, #what)?;
        }
    };
    let split = quote! {
        #sizes
        let mut items = flat;
        ::std::result::Result::Ok(#struct_name { #(#takes,)* #phantom_init })
    };

    let pushes = groups.iter().map(|g| {
        let name = &g.name;
        let emb = embed(g.kind);
        match g.arity {
            AritySpec::One => quote! {
                sizes.push(1);
                flat.push(#emb(groups.#name));
            },
            AritySpec::Optional => quote! {
                sizes.push(usize::from(groups.#name.is_some()));
                flat.extend(groups.#name.map(#emb));
            },
            AritySpec::Variadic => quote! {
                sizes.push(groups.#name.len());
                flat.extend(groups.#name.into_iter().map(#emb));
            },
        }
    });
    let flatten = quote! {
        let mut flat = ::gpt_rs_ir::op::Flat::new();
        let mut sizes: ::gpt_rs_ir::op::Flat<usize> = ::gpt_rs_ir::op::Flat::new();
        #(#pushes)*
        (flat, ::gpt_rs_ir::op::segments_if_needed(Self::#consts_name, &sizes))
    };

    let struct_def = if groups.is_empty() {
        quote! {
            #struct_def
            impl<V: ::gpt_rs_ir::op::View> ::std::default::Default for #struct_name<V> {
                fn default() -> Self { #struct_name { _view: ::std::marker::PhantomData } }
            }
        }
    } else {
        struct_def
    };

    GroupCode {
        struct_def,
        consts,
        split,
        flatten,
    }
}

/// The conversions of operands into the `Values` view that `create` takes: a value, or a tuple of
/// values for several groups, where a value is anything that converts into a `Value`.
fn operands_from(struct_name: &Ident, groups: &[GroupSpec]) -> TokenStream2 {
    let values = quote!(::gpt_rs_ir::op::Values);
    let value = quote!(::gpt_rs_ir::module::Value);
    if groups.is_empty() {
        return quote! {
            impl ::std::convert::From<()> for #struct_name<#values> {
                fn from((): ()) -> Self { ::std::default::Default::default() }
            }
        };
    }
    if let [group] = groups {
        let name = &group.name;
        return match group.arity {
            AritySpec::One => quote! {
                impl ::std::convert::From<#value> for #struct_name<#values> {
                    fn from(value: #value) -> Self { #struct_name { #name: value } }
                }
                impl ::std::convert::From<::gpt_rs_ir::module::ValueRef<'_>> for #struct_name<#values> {
                    fn from(value: ::gpt_rs_ir::module::ValueRef<'_>) -> Self {
                        #struct_name { #name: value.into() }
                    }
                }
            },
            AritySpec::Optional => quote! {
                impl<A: ::std::convert::Into<#value>> ::std::convert::From<::std::option::Option<A>> for #struct_name<#values> {
                    fn from(value: ::std::option::Option<A>) -> Self {
                        #struct_name { #name: value.map(::std::convert::Into::into) }
                    }
                }
            },
            AritySpec::Variadic => quote! {
                impl<A: ::std::convert::Into<#value>> ::std::convert::From<::std::vec::Vec<A>> for #struct_name<#values> {
                    fn from(values: ::std::vec::Vec<A>) -> Self {
                        #struct_name { #name: values.into_iter().map(::std::convert::Into::into).collect() }
                    }
                }
            },
        };
    }
    if groups.iter().any(|g| g.arity != AritySpec::One) {
        return quote!();
    }
    let names = groups.iter().map(|g| &g.name);
    let params: Vec<Ident> = (0..groups.len()).map(|i| format_ident!("A{}", i)).collect();
    let idx = (0..groups.len()).map(syn::Index::from);
    quote! {
        impl<#(#params: ::std::convert::Into<#value>),*> ::std::convert::From<(#(#params),*)> for #struct_name<#values> {
            fn from(values: (#(#params),*)) -> Self {
                #struct_name { #(#names: values.#idx.into()),* }
            }
        }
    }
}

/// What `create` gives for an op with these result groups, and how it is made from the groups.
fn created(results_ident: &Ident, groups: &[GroupSpec]) -> (TokenStream2, TokenStream2) {
    let value = quote!(::gpt_rs_ir::module::Value);
    match groups {
        [] => (quote!(()), quote!(let _ = results;)),
        [group] if group.arity == AritySpec::One => {
            let name = &group.name;
            (value, quote!(results.#name))
        }
        [group] if group.arity == AritySpec::Variadic => {
            let name = &group.name;
            (quote!(::std::vec::Vec<#value>), quote!(results.#name))
        }
        _ => (
            quote!(#results_ident<::gpt_rs_ir::op::Values>),
            quote!(results),
        ),
    }
}

/// The accessors of a [`Matched`] op: one per operand, result and region group.
fn matched_accessors(
    groups: &TokenStream2,
    operands: &[GroupSpec],
    results: &[GroupSpec],
    regions: &[RegionSpec],
) -> TokenStream2 {
    let value_ref = quote!(::gpt_rs_ir::module::ValueRef<'m>);
    let group = |field: TokenStream2, g: &GroupSpec| {
        let name = &g.name;
        match g.arity {
            AritySpec::One => quote! {
                pub fn #name(&self) -> #value_ref { self.#field.#name }
            },
            AritySpec::Optional => quote! {
                pub fn #name(&self) -> ::std::option::Option<#value_ref> { self.#field.#name }
            },
            AritySpec::Variadic => quote! {
                pub fn #name(&self) -> &[#value_ref] { &self.#field.#name }
            },
        }
    };
    let operand_fns = operands.iter().map(|g| group(quote!(operands), g));
    let result_fns = results.iter().map(|g| group(quote!(results), g));
    let region_fns = regions.iter().map(|r| {
        let name = &r.name;
        quote! {
            pub fn #name(&self) -> ::gpt_rs_ir::module::RegionRef<'m> { self.regions.#name }
        }
    });
    quote! {
        impl<'m, T: ::gpt_rs_ir::op::OpKind> ::gpt_rs_ir::op::Matched<'m, T, #groups> {
            #(#operand_fns)*
            #(#result_fns)*
            #(#region_fns)*
        }
    }
}

/// The groups of an op or a family, implemented on `target`: the structs `<base>Operands`,
/// `<base>Results` and `<base>Regions`, their `Groups` impl and the accessors of `Matched` on them.
fn groups_code(
    vis: &syn::Visibility,
    target: &TokenStream2,
    base: &Ident,
    operands: &[GroupSpec],
    results: &[GroupSpec],
    regions: &[RegionSpec],
) -> TokenStream2 {
    let operands_ident = format_ident!("{}Operands", base);
    let results_ident = format_ident!("{}Results", base);
    let regions_ident = format_ident!("{}Regions", base);
    let GroupCode {
        struct_def: operands_def,
        consts: operands_const,
        split: split_operands,
        flatten: flatten_operands,
    } = group_code(
        vis,
        &operands_ident,
        operands,
        &format_ident!("OPERANDS"),
        "operand",
    );
    let GroupCode {
        struct_def: results_def,
        consts: results_const,
        split: split_results,
        flatten: flatten_results,
    } = group_code(
        vis,
        &results_ident,
        results,
        &format_ident!("RESULTS"),
        "result",
    );
    let operands_from = operands_from(&operands_ident, operands);
    let (created_ty, created_body) = created(&results_ident, results);
    let accessors = matched_accessors(target, operands, results, regions);
    let region_names: Vec<String> = regions.iter().map(|r| r.name.unraw().to_string()).collect();
    let region_fields: Vec<&Ident> = regions.iter().map(|r| &r.name).collect();
    let region_count = regions.len();
    let (regions_def, regions_ty, split_regions, flatten_regions) = if regions.is_empty() {
        (
            quote!(),
            quote!(::gpt_rs_ir::op::NoRegions),
            quote! {
                ::anyhow::ensure!(flat.is_empty(), "expected no regions, got {}", flat.len());
                ::std::result::Result::Ok(::gpt_rs_ir::op::NoRegions)
            },
            quote! { ::std::vec::Vec::new() },
        )
    } else {
        (
            quote! {
                #vis struct #regions_ident<R> { #(pub #region_fields: R,)* }
            },
            quote!(#regions_ident<R>),
            quote! {
                ::anyhow::ensure!(flat.len() == #region_count, "expected {} regions, got {}", #region_count, flat.len());
                let mut items = flat.into_iter();
                ::std::result::Result::Ok(#regions_ident { #(#region_fields: items.next().expect("checked count"),)* })
            },
            quote! { ::std::vec![#(regions.#region_fields),*] },
        )
    };
    let terminator_arms = regions.iter().enumerate().map(|(i, r)| {
        let term = &r.terminator;
        quote!(#i => <#term as ::gpt_rs_ir::op::InOp>::cast(op).is_some(),)
    });
    quote! {
        #operands_def
        #results_def
        #regions_def
        #operands_from
        #accessors

        impl ::gpt_rs_ir::op::Groups for #target {
            type Operands<V: ::gpt_rs_ir::op::View> = #operands_ident<V>;
            type Results<V: ::gpt_rs_ir::op::View> = #results_ident<V>;
            type Regions<R> = #regions_ty;
            type Created = #created_ty;

            #operands_const
            #results_const
            const REGIONS: &'static [&'static str] = &[#(#region_names),*];

            fn split_operands<V: ::gpt_rs_ir::op::View>(
                flat: impl ::std::iter::ExactSizeIterator<Item = V::Any>,
                segments: ::std::option::Option<&[usize]>,
            ) -> ::anyhow::Result<Self::Operands<V>> {
                #split_operands
            }

            fn flatten_operands<V: ::gpt_rs_ir::op::Embed>(
                groups: Self::Operands<V>,
            ) -> (::gpt_rs_ir::op::Flat<V::Any>, ::std::option::Option<::std::vec::Vec<usize>>) {
                #flatten_operands
            }

            fn split_results<V: ::gpt_rs_ir::op::View>(
                flat: impl ::std::iter::ExactSizeIterator<Item = V::Any>,
                segments: ::std::option::Option<&[usize]>,
            ) -> ::anyhow::Result<Self::Results<V>> {
                #split_results
            }

            fn flatten_results<V: ::gpt_rs_ir::op::Embed>(
                groups: Self::Results<V>,
            ) -> (::gpt_rs_ir::op::Flat<V::Any>, ::std::option::Option<::std::vec::Vec<usize>>) {
                #flatten_results
            }

            fn split_regions<R>(flat: ::std::vec::Vec<R>) -> ::anyhow::Result<Self::Regions<R>> {
                #split_regions
            }

            fn flatten_regions<R>(regions: Self::Regions<R>) -> ::std::vec::Vec<R> {
                #flatten_regions
            }

            fn created(results: Self::Results<::gpt_rs_ir::op::Values>) -> Self::Created {
                #created_body
            }

            fn is_region_terminator(index: usize, op: &::gpt_rs_ir::Op) -> bool {
                match index {
                    #(#terminator_arms)*
                    _ => false,
                }
            }
        }
    }
}

fn expand_op(args: TokenStream2, mut input: DeriveInput) -> syn::Result<TokenStream2> {
    // The expansion reads the op with its arguments as an `#[op]` attribute, and the attributes of
    // its fields, which the declared struct leaves out.
    let mut op = input.clone();
    op.attrs.push(syn::parse_quote!(#[op(#args)]));
    let Data::Struct(data) = &mut input.data else {
        return Err(syn::Error::new_spanned(&op, "#[op] supports structs"));
    };
    let derives = if matches!(data.fields, Fields::Unit) {
        quote!(
            ::std::fmt::Debug,
            ::std::clone::Clone,
            ::std::marker::Copy,
            ::std::default::Default,
            ::std::cmp::PartialEq,
            ::std::cmp::Eq,
            ::std::hash::Hash
        )
    } else {
        quote!(
            ::std::fmt::Debug,
            ::std::clone::Clone,
            ::std::cmp::PartialEq,
            ::std::cmp::Eq,
            ::std::hash::Hash
        )
    };
    for field in data.fields.iter_mut() {
        field.attrs.retain(|attr| !attr.path().is_ident("attr"));
    }
    let parsed = op_args(&op)?;
    let impls = match &parsed.family {
        // The family's macro hands its groups to `family_op!`, which expands the op.
        Some(family) => {
            let family = &family.ident;
            quote!(#family! { #op })
        }
        None => expand_op_in(&op, parsed, None)?,
    };
    Ok(quote! {
        #[derive(#derives)]
        #input
        #impls
    })
}

fn op_args(input: &DeriveInput) -> syn::Result<OpArgs> {
    one_attr(input, "op", "an op needs #[op(\"dialect.name\", ...)]")?.parse_args()
}

/// What an op family gives each of its ops: its groups, effects and traits, from `#[family]` on
/// the family trait `ident`, whose doc comment is `doc`.
struct FamilySpec {
    ident: Ident,
    doc: LitStr,
    operands: Vec<GroupSpec>,
    results: Vec<GroupSpec>,
    regions: Vec<RegionSpec>,
    effects: EffectSpec,
    traits: Vec<Expr>,
    /// The family's type rule and semantics: generic functions over the ops of the family.
    rule: syn::Path,
    eval: syn::Path,
    /// The associated items of the trait, which each op sets in its `family(..)`.
    items: Vec<FamilyItem>,
}

/// An associated item of a family trait: `type Scalar` or `const CLASSES: &'static [Class]`.
enum FamilyItem {
    Type(Ident),
    Const(Ident, Box<SynType>),
}

impl FamilyItem {
    fn ident(&self) -> &Ident {
        match self {
            FamilyItem::Type(ident) | FamilyItem::Const(ident, _) => ident,
        }
    }
}

impl Parse for FamilyItem {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        if input.parse::<Option<Token![type]>>()?.is_some() {
            return Ok(FamilyItem::Type(input.parse()?));
        }
        input.parse::<Token![const]>()?;
        let ident = input.parse()?;
        input.parse::<Token![:]>()?;
        Ok(FamilyItem::Const(ident, Box::new(input.parse()?)))
    }
}

impl ToTokens for FamilyItem {
    fn to_tokens(&self, tokens: &mut TokenStream2) {
        tokens.extend(match self {
            FamilyItem::Type(ident) => quote!(type #ident),
            FamilyItem::Const(ident, ty) => quote!(const #ident: #ty),
        });
    }
}

/// The impl of the family trait for `op`, with the associated items it sets.
fn family_trait_impl(
    op: &Ident,
    family: &FamilySpec,
    set: &FamilyRef,
) -> syn::Result<TokenStream2> {
    let trait_ident = &family.ident;
    let mut items = Vec::new();
    for (key, value) in &set.items {
        let item = family
            .items
            .iter()
            .find(|item| item.ident() == key)
            .ok_or_else(|| {
                syn::Error::new(
                    key.span(),
                    format!("the family `{trait_ident}` has no associated item `{key}`"),
                )
            })?;
        items.push(match item {
            FamilyItem::Type(ident) => {
                let ty: SynType = syn::parse2(value.clone())?;
                quote!(type #ident = #ty;)
            }
            FamilyItem::Const(ident, ty) => {
                let expr: Expr = syn::parse2(value.clone())?;
                quote!(const #ident: #ty = #expr;)
            }
        });
    }
    Ok(quote!(impl #trait_ident for #op { #(#items)* }))
}

/// Where the attributes of an op are: its named fields, or the fields of the attribute group that
/// it wraps, which the ops of a family share.
enum AttrSource {
    Fields(Vec<FieldSpec>),
    Group(SynType),
}

fn attr_source(data: &syn::DataStruct) -> syn::Result<AttrSource> {
    match &data.fields {
        Fields::Unit => Ok(AttrSource::Fields(Vec::new())),
        Fields::Named(named) => Ok(AttrSource::Fields(
            named
                .named
                .iter()
                .map(field_spec)
                .collect::<syn::Result<_>>()?,
        )),
        Fields::Unnamed(unnamed) if unnamed.unnamed.len() == 1 => {
            Ok(AttrSource::Group(unnamed.unnamed[0].ty.clone()))
        }
        Fields::Unnamed(unnamed) => Err(syn::Error::new_spanned(
            unnamed,
            "an op struct has named fields, or wraps one attribute group",
        )),
    }
}

/// The text form of attributes held in named fields: the statements that print them into `out`,
/// the expression that parses them, a `Result<Self>` that reads the op name `name` and the
/// cursor `c`, and one `AttrInfo` per field.
struct AttrCode {
    print: TokenStream2,
    parse: TokenStream2,
    infos: Vec<TokenStream2>,
}

fn attr_code(fields: &[FieldSpec], (open, close): (&str, &str)) -> AttrCode {
    // `parse` reads each field into a local `<field>_slot`. The suffix keeps a field named `c`,
    // `name` or `key` from shadowing a local of `parse`.
    let slot_of = |f: &FieldSpec| format_ident!("{}_slot", f.ident);
    let prints = fields.iter().map(|f| {
        let (fid, key, ty) = (&f.ident, &f.key, &f.ty);
        let skip = if f.default {
            quote!(self.#fid == <#ty as ::std::default::Default>::default())
        } else {
            quote!(false)
        };
        quote! {
            if !(#skip) {
                let mut text = ::std::string::String::new();
                ::gpt_rs_ir::attr::Attr::print(&self.#fid, &mut text);
                out.push((#key, text));
            }
        }
    });
    let slot_decls = fields.iter().map(|f| {
        let (slot, ty) = (slot_of(f), &f.ty);
        quote!(let mut #slot: ::std::option::Option<#ty> = ::std::option::Option::None;)
    });
    let key_arms = fields.iter().map(|f| {
        let (slot, key, ty) = (slot_of(f), &f.key, &f.ty);
        quote! {
            #key => {
                if #slot.is_some() {
                    ::anyhow::bail!(c.error(format!("duplicate attribute `{}`", #key)));
                }
                #slot = ::std::option::Option::Some(<#ty as ::gpt_rs_ir::attr::Attr>::parse(c)?);
            }
        }
    });
    let field_inits = fields.iter().map(|f| {
        let (fid, key, slot) = (&f.ident, &f.key, slot_of(f));
        if f.default {
            quote!(#fid: #slot.unwrap_or_default())
        } else {
            quote!(#fid: #slot.ok_or_else(|| c.error(format!("{} needs attribute `{}`", name, #key)))?)
        }
    });
    let infos = fields
        .iter()
        .map(|f| {
            let (key, ty, default) = (&f.key, &f.ty, f.default);
            let ty_text = quote!(#ty).to_string().replace(' ', "");
            quote!(::gpt_rs_ir::op::AttrInfo { name: #key, ty: #ty_text, default: #default })
        })
        .collect();
    AttrCode {
        print: quote!(#(#prints)*),
        parse: quote! {
            (|| -> ::anyhow::Result<Self> {
                #(#slot_decls)*
                if c.eat(#open) {
                    c.list(#close, |c| {
                        let key = c.ident()?;
                        c.expect("=")?;
                        match key {
                            #(#key_arms)*
                            other => c.segment_attr(name, other)?,
                        }
                        ::std::result::Result::Ok(())
                    })?;
                }
                ::std::result::Result::Ok(Self { #(#field_inits,)* })
            })()
        },
        infos,
    }
}

/// Expands `#[op]` on `input`, an op of `family` when it is in one.
fn expand_op_in(
    input: &DeriveInput,
    args: OpArgs,
    family: Option<FamilySpec>,
) -> syn::Result<TokenStream2> {
    let Data::Struct(data) = &input.data else {
        return Err(syn::Error::new_spanned(input, "#[op] supports structs"));
    };
    let ident = &input.ident;
    let vis = &input.vis;
    let op_name = &args.name;
    if op_name.value().contains('{') {
        return Err(syn::Error::new(
            op_name.span(),
            "an op has one name; the ops of a family are structs of their own",
        ));
    }
    let attrs = attr_source(data)?;
    let family_set = &args.family;
    let (constant, symbol) = (args.constant, args.symbol);
    let (operands, results, regions, effects, traits) = match family.as_ref() {
        Some(f) => {
            let own_groups =
                !args.operands.is_empty() || !args.results.is_empty() || !args.regions.is_empty();
            if own_groups || !args.effects.is_none() {
                return Err(syn::Error::new(
                    op_name.span(),
                    "an op of a family takes its groups and effects from the family",
                ));
            }
            let mut traits = f.traits.clone();
            traits.extend(args.traits.iter().cloned());
            (
                f.operands.clone(),
                f.results.clone(),
                f.regions.clone(),
                f.effects.clone(),
                traits,
            )
        }
        None => (
            args.operands,
            args.results,
            args.regions,
            args.effects,
            args.traits,
        ),
    };

    let has_regions = !regions.is_empty();
    // An op of a family shares the family's groups; an op of its own has its own.
    let (groups, groups_code) = match &family {
        Some(f) => {
            let marker = format_ident!("{}Family", f.ident);
            (quote!(#marker), quote!())
        }
        None => {
            let target = quote!(#ident);
            let code = groups_code(vis, &target, ident, &operands, &results, &regions);
            (target, code)
        }
    };

    let EffectSpec {
        opaque,
        rng,
        alloc,
        reads,
        writes,
    } = &effects;
    let group_bit = |group: &Ident| -> syn::Result<TokenStream2> {
        let index = operands
            .iter()
            .position(|g| g.name == *group)
            .ok_or_else(|| syn::Error::new(group.span(), "effect names no operand group"))?;
        let bit = 1u64 << index;
        Ok(quote!(#bit))
    };
    let read_bits = reads
        .iter()
        .map(group_bit)
        .collect::<syn::Result<Vec<_>>>()?;
    let write_bits = writes
        .iter()
        .map(group_bit)
        .collect::<syn::Result<Vec<_>>>()?;
    let op_traits = trait_slice(&traits);
    // The folder that `#[fold]` registers for this op, found once.
    let fold_body = quote! {
        static FOLDER: ::std::sync::OnceLock<::std::option::Option<::gpt_rs_ir::op::FoldFn>> =
            ::std::sync::OnceLock::new();
        let fold = (*FOLDER.get_or_init(|| {
            ::gpt_rs_ir::op::folder_of(::std::any::TypeId::of::<Self>())
        }))?;
        fold(self, ins, segments, results)
    };
    let declare_regions = if has_regions {
        quote!(impl ::gpt_rs_ir::op::DeclaresRegions for #ident {})
    } else {
        quote!()
    };

    let (print_attrs, parse_attrs, attr_infos, attrs_deref) = match &attrs {
        AttrSource::Fields(fields) => {
            let AttrCode {
                print,
                parse,
                infos,
            } = attr_code(fields, ("{", "}"));
            (print, parse, quote!(&[#(#infos),*]), quote!())
        }
        AttrSource::Group(group) => (
            quote!(<#group as ::gpt_rs_ir::attr::AttrGroup>::print_attrs(&self.0, out)),
            quote!(<#group as ::gpt_rs_ir::attr::AttrGroup>::parse_attrs(name, c).map(Self)),
            quote!(<#group as ::gpt_rs_ir::attr::AttrGroup>::INFOS),
            quote! {
                impl ::std::ops::Deref for #ident {
                    type Target = #group;
                    fn deref(&self) -> &#group {
                        &self.0
                    }
                }

                impl ::std::ops::DerefMut for #ident {
                    fn deref_mut(&mut self) -> &mut #group {
                        &mut self.0
                    }
                }

                impl ::std::convert::From<#group> for #ident {
                    fn from(attrs: #group) -> Self {
                        Self(attrs)
                    }
                }
            },
        ),
    };

    let op_doc = doc_text(&input.attrs);
    let family_info = match &family {
        Some(f) => {
            let (name, doc) = (f.ident.to_string(), &f.doc);
            quote!(::std::option::Option::Some(::gpt_rs_ir::op::FamilyInfo { name: #name, doc: #doc }))
        }
        None => quote!(::std::option::Option::None),
    };
    // The type rule and the semantics of an op of a family are the family's.
    let family_impls = match &family {
        Some(f) => {
            let (rule, eval) = (&f.rule, &f.eval);
            let set = family_set
                .as_ref()
                .expect("an op of a family names it in `family(..)`");
            let trait_impl = family_trait_impl(ident, f, set)?;
            quote! {
                #trait_impl

                impl ::gpt_rs_ir::op::OpDef for #ident {
                    fn result_types(
                        &self,
                        ins: ::gpt_rs_ir::op::Operands<Self, ::gpt_rs_ir::op::Types<'_>>,
                        cx: &::gpt_rs_ir::op::TypeCx<'_, ::gpt_rs_ir::op::Regions<Self, ::gpt_rs_ir::module::Region>>,
                    ) -> ::anyhow::Result<::gpt_rs_ir::op::Results<Self, ::gpt_rs_ir::op::OwnedTypes>> {
                        #rule(self, ins, cx)
                    }
                }

                impl ::gpt_rs_ir::op::Eval for #ident {
                    fn eval(
                        &self,
                        ins: ::gpt_rs_ir::op::Operands<Self, ::gpt_rs_ir::op::Host<'_>>,
                        cx: &mut ::gpt_rs_ir::op::EvalCx<'_, ::gpt_rs_ir::op::Regions<Self, ::gpt_rs_ir::module::Region>>,
                    ) -> ::anyhow::Result<::gpt_rs_ir::op::Results<Self, ::gpt_rs_ir::op::OwnedHost>> {
                        #eval(self, ins, cx)
                    }
                }
            }
        }
        None => quote!(),
    };

    let constant_hook = if constant {
        quote! {
            fn dyn_constant(&self) -> ::std::option::Option<&::gpt_rs_ir::dense::Dense> {
                ::std::option::Option::Some(<Self as ::gpt_rs_ir::op::ConstantOp>::value(self))
            }
        }
    } else {
        quote!()
    };
    let (symbol_hook, parse_symbol) = if symbol {
        (
            quote! {
                fn dyn_symbol(&self) -> ::std::option::Option<&dyn ::gpt_rs_ir::op::SymbolOp> {
                    ::std::option::Option::Some(self)
                }
            },
            quote! {
                fn parse_symbol(
                    name: &str,
                    c: &mut ::gpt_rs_ir::text::Cursor<'_>,
                ) -> ::std::option::Option<::anyhow::Result<::gpt_rs_ir::op::ParsedSymbol<Self>>> {
                    if name != #op_name {
                        return ::std::option::Option::None;
                    }
                    ::std::option::Option::Some(<Self as ::gpt_rs_ir::op::SymbolOp>::parse_signature(c))
                }
            },
        )
    } else {
        (quote!(), quote!())
    };

    let region_sigs = if has_regions {
        quote! {
            let typed = <<Self as ::gpt_rs_ir::op::OpKind>::Groups as ::gpt_rs_ir::op::Groups>::split_operands::<::gpt_rs_ir::op::Types<'_>>(ins.iter().copied(), segments)?;
            let sigs = <Self as ::gpt_rs_ir::op::RegionOpDef>::region_signatures(self, typed)?;
            ::std::result::Result::Ok(<<Self as ::gpt_rs_ir::op::OpKind>::Groups as ::gpt_rs_ir::op::Groups>::flatten_regions(sigs))
        }
    } else {
        quote!(::std::result::Result::Ok(::std::vec::Vec::new()))
    };

    Ok(quote! {
        #groups_code
        #declare_regions
        #attrs_deref
        #family_impls

        impl ::gpt_rs_ir::op::OpKind for #ident {
            type Groups = #groups;

            const INFO: ::gpt_rs_ir::op::OpInfo = ::gpt_rs_ir::op::OpInfo {
                name: #op_name,
                rust: ::std::stringify!(#ident),
                file: ::std::file!(),
                doc: #op_doc,
                operands: <#groups as ::gpt_rs_ir::op::Groups>::OPERANDS,
                results: <#groups as ::gpt_rs_ir::op::Groups>::RESULTS,
                regions: <#groups as ::gpt_rs_ir::op::Groups>::REGIONS,
                attrs: #attr_infos,
                traits: #op_traits,
                effects: ::gpt_rs_ir::op::Effects {
                    opaque: #opaque,
                    rng: #rng,
                    alloc: #alloc,
                    reads: 0 #(| #read_bits)*,
                    writes: 0 #(| #write_bits)*,
                },
                family: #family_info,
            };

            fn name(&self) -> ::std::borrow::Cow<'static, str> {
                ::std::borrow::Cow::Borrowed(#op_name)
            }

            fn effects(&self) -> ::gpt_rs_ir::op::Effects {
                <Self as ::gpt_rs_ir::op::OpKind>::INFO.effects
            }

            fn traits(&self) -> ::gpt_rs_ir::traits::Traits {
                ::gpt_rs_ir::traits::Traits::new(#op_traits)
            }

            fn print_attrs(&self, out: &mut ::std::vec::Vec<(&'static str, ::std::string::String)>) {
                #print_attrs
            }

            fn parse(
                name: &str,
                c: &mut ::gpt_rs_ir::text::Cursor<'_>,
            ) -> ::std::option::Option<::anyhow::Result<Self>> {
                if name != #op_name {
                    return ::std::option::Option::None;
                }
                ::std::option::Option::Some(#parse_attrs)
            }

            #parse_symbol
        }

        impl ::gpt_rs_ir::op::OpDyn for #ident {
            fn dyn_result_types(
                &self,
                ins: &[&::gpt_rs_ir::types::Type],
                segments: ::std::option::Option<&[usize]>,
                regions: &[::gpt_rs_ir::module::Region],
                module: &::gpt_rs_ir::module::Module,
            ) -> ::anyhow::Result<(::gpt_rs_ir::op::Flat<::gpt_rs_ir::types::Type>, ::std::option::Option<::std::vec::Vec<usize>>)> {
                let typed = <<Self as ::gpt_rs_ir::op::OpKind>::Groups as ::gpt_rs_ir::op::Groups>::split_operands::<::gpt_rs_ir::op::Types<'_>>(ins.iter().copied(), segments)?;
                let regions = <<Self as ::gpt_rs_ir::op::OpKind>::Groups as ::gpt_rs_ir::op::Groups>::split_regions(regions.to_vec())?;
                let out = <Self as ::gpt_rs_ir::op::OpDef>::result_types(
                    self,
                    typed,
                    &::gpt_rs_ir::op::TypeCx { module, regions },
                )?;
                ::std::result::Result::Ok(<<Self as ::gpt_rs_ir::op::OpKind>::Groups as ::gpt_rs_ir::op::Groups>::flatten_results::<::gpt_rs_ir::op::OwnedTypes>(out))
            }

            fn dyn_any(&self) -> &dyn ::std::any::Any {
                self
            }

            fn dyn_region_signatures(
                &self,
                ins: &[&::gpt_rs_ir::types::Type],
                segments: ::std::option::Option<&[usize]>,
            ) -> ::anyhow::Result<::std::vec::Vec<::gpt_rs_ir::op::RegionSig>> {
                #region_sigs
            }

            fn dyn_fold(
                &self,
                ins: &[::gpt_rs_ir::op::FoldIn<'_>],
                segments: ::std::option::Option<&[usize]>,
                results: &[::gpt_rs_ir::types::Type],
            ) -> ::std::option::Option<::std::vec::Vec<::gpt_rs_ir::op::Folded>> {
                #fold_body
            }

            fn dyn_eval(
                &self,
                ins: &[&::gpt_rs_ir::host::HostValue],
                segments: ::std::option::Option<&[usize]>,
                regions: &[::gpt_rs_ir::module::Region],
                results: &[::gpt_rs_ir::types::Type],
                runner: &mut dyn ::gpt_rs_ir::op::RegionRunner,
            ) -> ::anyhow::Result<::std::vec::Vec<::gpt_rs_ir::host::HostValue>> {
                let typed = <<Self as ::gpt_rs_ir::op::OpKind>::Groups as ::gpt_rs_ir::op::Groups>::split_operands::<::gpt_rs_ir::op::Host<'_>>(ins.iter().copied(), segments)?;
                let regions = <<Self as ::gpt_rs_ir::op::OpKind>::Groups as ::gpt_rs_ir::op::Groups>::split_regions(regions.to_vec())?;
                let mut cx = ::gpt_rs_ir::op::EvalCx { regions, results, runner };
                let out = <Self as ::gpt_rs_ir::op::Eval>::eval(self, typed, &mut cx)?;
                ::std::result::Result::Ok(<<Self as ::gpt_rs_ir::op::OpKind>::Groups as ::gpt_rs_ir::op::Groups>::flatten_results::<::gpt_rs_ir::op::OwnedHost>(out).0.into_vec())
            }

            #constant_hook
            #symbol_hook
        }
    })
}

/// A group as `#[op(..)]` declares it: `lhs: Tensor`, `ys: Variadic<Tensor>`.
fn group_tokens(group: &GroupSpec) -> TokenStream2 {
    let name = &group.name;
    let kind = match group.kind {
        KindSpec::Any => quote!(Any),
        KindSpec::Tensor => quote!(Tensor),
        KindSpec::Index => quote!(Index),
        KindSpec::Scalar => quote!(Scalar),
    };
    match group.arity {
        AritySpec::One => quote!(#name(#kind)),
        AritySpec::Optional => quote!(#name(Optional(#kind))),
        AritySpec::Variadic => quote!(#name(Variadic(#kind))),
    }
}

/// Effects as `#[op(effects(..))]` declares them.
fn effect_tokens(effects: &EffectSpec) -> TokenStream2 {
    let mut items = Vec::new();
    for (set, name) in [
        (effects.opaque, "opaque"),
        (effects.rng, "rng"),
        (effects.alloc, "alloc"),
    ] {
        if set {
            let name = format_ident!("{}", name);
            items.push(quote!(#name));
        }
    }
    items.extend(effects.reads.iter().map(|group| quote!(read(#group))));
    items.extend(effects.writes.iter().map(|group| quote!(write(#group))));
    quote!(effects(#(#items),*))
}

// ------------------------------------------------------------------------------------------------
// #[family]

/// The arguments of `#[family(operands(..), results(..), regions(..), effects(..), traits(..))]`.
#[derive(Default)]
struct FamilyArgs {
    operands: Vec<GroupSpec>,
    results: Vec<GroupSpec>,
    regions: Vec<RegionSpec>,
    effects: EffectSpec,
    traits: Vec<Expr>,
    rule: Option<syn::Path>,
    eval: Option<syn::Path>,
}

impl FamilyArgs {
    /// Parses the clauses, each `key(..)`, separated by commas.
    fn parse_clauses(input: ParseStream, args: &mut FamilyArgs) -> syn::Result<()> {
        while !input.is_empty() {
            let key: Ident = input.parse()?;
            match key.to_string().as_str() {
                "operands" => args.operands = parse_list(input, parse_group)?,
                "results" => args.results = parse_list(input, parse_group)?,
                "regions" => args.regions = parse_list(input, parse_region)?,
                "effects" => args.effects = parse_effects(input)?,
                "traits" => args.traits = parse_list(input, Expr::parse)?,
                "rule" => {
                    let path;
                    syn::parenthesized!(path in input);
                    args.rule = Some(path.parse()?);
                }
                "eval" => {
                    let path;
                    syn::parenthesized!(path in input);
                    args.eval = Some(path.parse()?);
                }
                other => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!(
                            "unknown #[family] clause `{other}`, expected operands, results, \
                             regions, effects, traits, rule or eval"
                        ),
                    ))
                }
            }
            if input.is_empty() {
                break;
            }
            input.parse::<Token![,]>()?;
        }
        Ok(())
    }
}

impl Parse for FamilyArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut args = FamilyArgs::default();
        FamilyArgs::parse_clauses(input, &mut args)?;
        Ok(args)
    }
}

/// `#[family(..)]` on the trait of an op family: the trait gives each op of the family what the
/// family's type rule and semantics read of it. `rule(<path>)` names the type rule and
/// `eval(<path>)` the semantics, functions generic over the ops of the family, kept with the other
/// type rules and semantics of the dialect. The ops share the groups, effects and traits the
/// attribute declares, and the group structs it generates, named after the trait.
#[proc_macro_attribute]
pub fn family(args: TokenStream, input: TokenStream) -> TokenStream {
    let args = parse_macro_input!(args as FamilyArgs);
    let input = parse_macro_input!(input as syn::ItemTrait);
    expand_family(args, input)
        .unwrap_or_else(|err| err.to_compile_error())
        .into()
}

fn expand_family(args: FamilyArgs, mut input: syn::ItemTrait) -> syn::Result<TokenStream2> {
    let ident = input.ident.clone();
    let vis = input.vis.clone();
    let marker = format_ident!("{}Family", ident);
    let groups = groups_code(
        &vis,
        &quote!(#marker),
        &ident,
        &args.operands,
        &args.results,
        &args.regions,
    );
    // The ops of the family are ops of a dialect, with the family's groups.
    input
        .supertraits
        .push(syn::parse_quote!(::gpt_rs_ir::op::InOp));
    input
        .supertraits
        .push(syn::parse_quote!(::gpt_rs_ir::op::OpKind<Groups = #marker>));
    let doc = doc_text(&input.attrs);
    let marker_doc = format!("The groups of the ops of the family [`{ident}`].");
    let operands = args.operands.iter().map(group_tokens);
    let results = args.results.iter().map(group_tokens);
    let regions = args.regions.iter().map(|r| {
        let (name, term) = (&r.name, &r.terminator);
        quote!(#name(Block(#term)))
    });
    let effects = effect_tokens(&args.effects);
    let traits = &args.traits;
    let mut items = Vec::new();
    for item in &input.items {
        match item {
            syn::TraitItem::Type(item) => items.push(FamilyItem::Type(item.ident.clone())),
            syn::TraitItem::Const(item) => items.push(FamilyItem::Const(
                item.ident.clone(),
                Box::new(item.ty.clone()),
            )),
            _ => {}
        }
    }
    let (Some(rule), Some(eval)) = (&args.rule, &args.eval) else {
        return Err(syn::Error::new_spanned(
            &ident,
            "a family needs `rule(<path>)`, its type rule, and `eval(<path>)`, its semantics",
        ));
    };
    // The `#[op]` of each op of the family calls this macro, which has the trait's name and travels
    // with every import of it.
    let family_macro = format_ident!("__gpt_rs_ir_family_{}", ident);
    Ok(quote! {
        #input

        #[doc = #marker_doc]
        #vis struct #marker;

        #groups

        #[doc(hidden)]
        #[allow(unused_macros)]
        macro_rules! #family_macro {
            ($($op:tt)*) => {
                ::gpt_rs_ir::family_op! {
                    family = #ident,
                    doc = #doc,
                    items(#(#items),*),
                    operands(#(#operands),*),
                    results(#(#results),*),
                    regions(#(#regions),*),
                    #effects,
                    traits(#(#traits),*),
                    rule(#rule),
                    eval(#eval);
                    $($op)*
                }
            };
        }
        #[doc(hidden)]
        #[allow(unused_imports)]
        pub(crate) use #family_macro as #ident;
    })
}

/// What a family's macro passes to `family_op!`: the family, then the op.
struct FamilyOpInput {
    family: FamilySpec,
    op: DeriveInput,
}

impl Parse for FamilyOpInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        fn assigned<T: Parse>(input: ParseStream, expected: &str) -> syn::Result<T> {
            let key: Ident = input.parse()?;
            if key != expected {
                return Err(syn::Error::new(
                    key.span(),
                    format!("expected `{expected}`"),
                ));
            }
            input.parse::<Token![=]>()?;
            let value = input.parse()?;
            input.parse::<Token![,]>()?;
            Ok(value)
        }
        let ident = assigned(input, "family")?;
        let doc = assigned(input, "doc")?;
        let items_key: Ident = input.parse()?;
        if items_key != "items" {
            return Err(syn::Error::new(items_key.span(), "expected `items`"));
        }
        let items = parse_list(input, FamilyItem::parse)?;
        input.parse::<Token![,]>()?;
        let clauses;
        let mut args = FamilyArgs::default();
        {
            let mut tokens = TokenStream2::new();
            while !input.peek(Token![;]) {
                let tree: proc_macro2::TokenTree = input.parse()?;
                tokens.extend([tree]);
            }
            clauses = tokens;
        }
        syn::parse::Parser::parse2(
            |stream: ParseStream| FamilyArgs::parse_clauses(stream, &mut args),
            clauses,
        )?;
        input.parse::<Token![;]>()?;
        Ok(Self {
            family: FamilySpec {
                ident,
                doc,
                operands: args.operands,
                results: args.results,
                regions: args.regions,
                effects: args.effects,
                traits: args.traits,
                rule: args
                    .rule
                    .ok_or_else(|| input.error("the family names no rule"))?,
                eval: args
                    .eval
                    .ok_or_else(|| input.error("the family names no eval"))?,
                items,
            },
            op: input.parse()?,
        })
    }
}

/// Expands `#[op]` on an op of a family, with the family's groups, effects and traits.
#[doc(hidden)]
#[proc_macro]
pub fn family_op(input: TokenStream) -> TokenStream {
    let FamilyOpInput { family, op } = parse_macro_input!(input as FamilyOpInput);
    op_args(&op)
        .and_then(|args| expand_op_in(&op, args, Some(family)))
        .unwrap_or_else(|err| err.to_compile_error())
        .into()
}

// ------------------------------------------------------------------------------------------------
// #[derive(AttrGroup)]

#[proc_macro_derive(AttrGroup, attributes(attr))]
pub fn derive_attr_group(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    expand_attr_group(&input)
        .unwrap_or_else(|err| err.to_compile_error())
        .into()
}

fn expand_attr_group(input: &DeriveInput) -> syn::Result<TokenStream2> {
    let Data::Struct(syn::DataStruct {
        fields: Fields::Named(named),
        ..
    }) = &input.data
    else {
        return Err(syn::Error::new_spanned(
            input,
            "#[derive(AttrGroup)] supports structs with named fields",
        ));
    };
    let fields = named
        .named
        .iter()
        .map(field_spec)
        .collect::<syn::Result<Vec<_>>>()?;
    let ident = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();
    let AttrCode {
        print,
        parse,
        infos,
    } = attr_code(&fields, ("{", "}"));
    let params = attr_code(&fields, ("<", ">")).parse;
    Ok(quote! {
        impl #impl_generics ::gpt_rs_ir::attr::AttrGroup for #ident #ty_generics #where_clause {
            const INFOS: &'static [::gpt_rs_ir::op::AttrInfo] = &[#(#infos),*];

            fn print_attrs(&self, out: &mut ::std::vec::Vec<(&'static str, ::std::string::String)>) {
                #print
            }

            fn parse_attrs(
                name: &str,
                c: &mut ::gpt_rs_ir::text::Cursor<'_>,
            ) -> ::anyhow::Result<Self> {
                #parse
            }

            fn parse_params(
                name: &str,
                c: &mut ::gpt_rs_ir::text::Cursor<'_>,
            ) -> ::anyhow::Result<Self> {
                #params
            }
        }
    })
}

// ------------------------------------------------------------------------------------------------
// #[derive(Dialect)]

struct DialectArgs {
    name: LitStr,
    op_variant: Ident,
}

fn dialect_arg<T: Parse>(input: ParseStream, expected: &str) -> syn::Result<T> {
    let key: Ident = input.parse()?;
    if key != expected {
        return Err(syn::Error::new(
            key.span(),
            format!("expected `{expected} = ..`"),
        ));
    }
    input.parse::<Token![=]>()?;
    input.parse()
}

impl Parse for DialectArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let name = dialect_arg(input, "name")?;
        input.parse::<Token![,]>()?;
        let op_variant = dialect_arg(input, "op")?;
        input.parse::<Option<Token![,]>>()?;
        Ok(DialectArgs { name, op_variant })
    }
}

/// `T` when `ty` is `Box<T>`.
fn boxed_type(ty: &SynType) -> Option<&SynType> {
    let SynType::Path(path) = ty else {
        return None;
    };
    let segment = path.path.segments.last()?;
    if segment.ident != "Box" {
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

#[proc_macro_derive(Dialect, attributes(dialect))]
pub fn derive_dialect(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match expand_dialect(&input) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

fn expand_dialect(input: &DeriveInput) -> syn::Result<TokenStream2> {
    let args: DialectArgs = one_attr(
        input,
        "dialect",
        "#[derive(Dialect)] needs #[dialect(name = \"...\", op = Variant)]",
    )?
    .parse_args()?;
    let Data::Enum(data) = &input.data else {
        return Err(syn::Error::new_spanned(
            input,
            "#[derive(Dialect)] supports enums",
        ));
    };
    let ident = &input.ident;
    let op_variant = &args.op_variant;
    let dialect_name = &args.name;

    let mut variants = Vec::new();
    let mut tys = Vec::new();
    // How a variant's field gives the op (`op` is a reference to the field), and how the field
    // holds an op `op`: in place, or boxed.
    let mut access = Vec::new();
    let mut wrap = Vec::new();
    for variant in &data.variants {
        let field = match &variant.fields {
            Fields::Unnamed(fields) if fields.unnamed.len() == 1 => &fields.unnamed[0].ty,
            _ => {
                return Err(syn::Error::new_spanned(
                    variant,
                    "dialect variants hold one op: `Name(Name)` or `Name(Box<Name>)`",
                ))
            }
        };
        variants.push(&variant.ident);
        match boxed_type(field) {
            Some(op) => {
                tys.push(op);
                access.push(quote!(&**op));
                wrap.push(quote!(::std::boxed::Box::new(op)));
            }
            None => {
                tys.push(field);
                access.push(quote!(op));
                wrap.push(quote!(op));
            }
        }
    }

    let op_kind = quote!(::gpt_rs_ir::op::OpKind);
    let op_dyn = quote!(::gpt_rs_ir::op::OpDyn);

    Ok(quote! {
        impl #ident {
            pub const DIALECT: &'static str = #dialect_name;

            /// What the reference of the dialect shows about each of its ops, in order.
            pub fn infos() -> ::std::vec::Vec<::gpt_rs_ir::op::OpInfo> {
                ::std::vec![#(<#tys as #op_kind>::INFO),*]
            }

            pub fn name(&self) -> ::std::borrow::Cow<'static, str> {
                match self { #(Self::#variants(op) => #op_kind::name(#access),)* }
            }

            pub fn effects(&self) -> ::gpt_rs_ir::op::Effects {
                match self { #(Self::#variants(op) => #op_kind::effects(#access),)* }
            }

            pub fn traits(&self) -> ::gpt_rs_ir::traits::Traits {
                match self { #(Self::#variants(op) => #op_kind::traits(#access),)* }
            }

            pub fn operand_groups(&self) -> &'static [::gpt_rs_ir::op::Group] {
                match self { #(Self::#variants(_) => <<#tys as #op_kind>::Groups as ::gpt_rs_ir::op::Groups>::OPERANDS,)* }
            }

            pub fn result_groups(&self) -> &'static [::gpt_rs_ir::op::Group] {
                match self { #(Self::#variants(_) => <<#tys as #op_kind>::Groups as ::gpt_rs_ir::op::Groups>::RESULTS,)* }
            }

            pub fn region_names(&self) -> &'static [&'static str] {
                match self { #(Self::#variants(_) => <<#tys as #op_kind>::Groups as ::gpt_rs_ir::op::Groups>::REGIONS,)* }
            }

            pub fn is_region_terminator(&self, index: usize, op: &::gpt_rs_ir::Op) -> bool {
                match self { #(Self::#variants(_) => <<#tys as #op_kind>::Groups as ::gpt_rs_ir::op::Groups>::is_region_terminator(index, op),)* }
            }

            pub fn print_attrs(&self, out: &mut ::std::vec::Vec<(&'static str, ::std::string::String)>) {
                match self { #(Self::#variants(op) => #op_kind::print_attrs(#access, out),)* }
            }

            pub fn parse(
                name: &str,
                c: &mut ::gpt_rs_ir::text::Cursor<'_>,
            ) -> ::std::option::Option<::anyhow::Result<Self>> {
                #(
                    if let ::std::option::Option::Some(result) = <#tys as #op_kind>::parse(name, c) {
                        return ::std::option::Option::Some(result.map(|op| Self::#variants(#wrap)));
                    }
                )*
                ::std::option::Option::None
            }

            /// Parses the signature of the symbol op named `name`, or returns `None` when the dialect
            /// has no symbol op of that name.
            pub fn parse_symbol(
                name: &str,
                c: &mut ::gpt_rs_ir::text::Cursor<'_>,
            ) -> ::std::option::Option<::anyhow::Result<::gpt_rs_ir::op::ParsedSymbol<Self>>> {
                #(
                    if let ::std::option::Option::Some(result) = <#tys as #op_kind>::parse_symbol(name, c) {
                        return ::std::option::Option::Some(result.map(|(op, args)| (Self::#variants(#wrap), args)));
                    }
                )*
                ::std::option::Option::None
            }

            /// Adds every op of the dialect to `target` as the conversion `C` decides: legal, or
            /// illegal with the pattern that rewrites it, added to `patterns`. Every op needs a
            /// decision, so an op without one does not compile.
            pub fn convert<C>(
                target: &mut ::gpt_rs_ir::conversion::ConversionTarget,
                patterns: &mut ::gpt_rs_ir::rewrite::PatternSet,
            )
            where
                #(#tys: ::gpt_rs_ir::conversion::ConvertOp<C>,)*
            {
                #(
                    target.decide::<#tys>(
                        <#tys as ::gpt_rs_ir::conversion::ConvertOp<C>>::form(),
                        patterns,
                    );
                )*
            }
        }

        impl #op_dyn for #ident {
            fn dyn_result_types(
                &self,
                ins: &[&::gpt_rs_ir::types::Type],
                segments: ::std::option::Option<&[usize]>,
                regions: &[::gpt_rs_ir::module::Region],
                module: &::gpt_rs_ir::module::Module,
            ) -> ::anyhow::Result<(::gpt_rs_ir::op::Flat<::gpt_rs_ir::types::Type>, ::std::option::Option<::std::vec::Vec<usize>>)> {
                match self { #(Self::#variants(op) => #op_dyn::dyn_result_types(#access, ins, segments, regions, module),)* }
            }

            fn dyn_any(&self) -> &dyn ::std::any::Any {
                match self { #(Self::#variants(op) => #op_dyn::dyn_any(#access),)* }
            }

            fn dyn_region_signatures(
                &self,
                ins: &[&::gpt_rs_ir::types::Type],
                segments: ::std::option::Option<&[usize]>,
            ) -> ::anyhow::Result<::std::vec::Vec<::gpt_rs_ir::op::RegionSig>> {
                match self { #(Self::#variants(op) => #op_dyn::dyn_region_signatures(#access, ins, segments),)* }
            }

            fn dyn_fold(
                &self,
                ins: &[::gpt_rs_ir::op::FoldIn<'_>],
                segments: ::std::option::Option<&[usize]>,
                results: &[::gpt_rs_ir::types::Type],
            ) -> ::std::option::Option<::std::vec::Vec<::gpt_rs_ir::op::Folded>> {
                match self { #(Self::#variants(op) => #op_dyn::dyn_fold(#access, ins, segments, results),)* }
            }

            fn dyn_eval(
                &self,
                ins: &[&::gpt_rs_ir::host::HostValue],
                segments: ::std::option::Option<&[usize]>,
                regions: &[::gpt_rs_ir::module::Region],
                results: &[::gpt_rs_ir::types::Type],
                runner: &mut dyn ::gpt_rs_ir::op::RegionRunner,
            ) -> ::anyhow::Result<::std::vec::Vec<::gpt_rs_ir::host::HostValue>> {
                match self { #(Self::#variants(op) => #op_dyn::dyn_eval(#access, ins, segments, regions, results, runner),)* }
            }

            fn dyn_constant(&self) -> ::std::option::Option<&::gpt_rs_ir::dense::Dense> {
                match self { #(Self::#variants(op) => #op_dyn::dyn_constant(#access),)* }
            }

            fn dyn_symbol(&self) -> ::std::option::Option<&dyn ::gpt_rs_ir::op::SymbolOp> {
                match self { #(Self::#variants(op) => #op_dyn::dyn_symbol(#access),)* }
            }
        }

        #(
            impl ::std::convert::From<#tys> for ::gpt_rs_ir::Op {
                fn from(op: #tys) -> Self { ::gpt_rs_ir::Op::#op_variant(#ident::#variants(#wrap)) }
            }

            impl ::gpt_rs_ir::op::InOp for #tys {
                fn cast(op: &::gpt_rs_ir::Op) -> ::std::option::Option<&Self> {
                    match op {
                        ::gpt_rs_ir::Op::#op_variant(#ident::#variants(op)) => ::std::option::Option::Some(#access),
                        _ => ::std::option::Option::None,
                    }
                }
            }
        )*
    })
}

// ------------------------------------------------------------------------------------------------
// #[derive(DialectSet)]

/// `#[derive(DialectSet)]` with `#[dialect_set(kind(Kind))]` on the union of the dialects, an enum
/// whose variants each hold a dialect's enum: it generates `Kind`, the enum of the dialects, and
/// the dispatch over every dialect.
#[proc_macro_derive(DialectSet, attributes(dialect_set))]
pub fn derive_dialect_set(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    expand_dialect_set(&input)
        .unwrap_or_else(|err| err.to_compile_error())
        .into()
}

fn expand_dialect_set(input: &DeriveInput) -> syn::Result<TokenStream2> {
    let mut kind = None;
    one_attr(
        input,
        "dialect_set",
        "#[derive(DialectSet)] needs #[dialect_set(kind(Kind))]",
    )?
    .parse_nested_meta(|meta| {
        if !meta.path.is_ident("kind") {
            return Err(meta.error("expected kind(Kind)"));
        }
        let content;
        syn::parenthesized!(content in meta.input);
        kind = Some(content.parse::<Ident>()?);
        Ok(())
    })?;
    let kind = kind.ok_or_else(|| syn::Error::new_spanned(input, "expected kind(Kind)"))?;
    let Data::Enum(data) = &input.data else {
        return Err(syn::Error::new_spanned(
            input,
            "#[derive(DialectSet)] supports enums",
        ));
    };
    let ident = &input.ident;
    let vis = &input.vis;
    let mut variants = Vec::new();
    let mut tys = Vec::new();
    for variant in &data.variants {
        let Fields::Unnamed(fields) = &variant.fields else {
            return Err(syn::Error::new_spanned(
                variant,
                "a dialect set's variants hold one dialect: `Name(NameOp)`",
            ));
        };
        variants.push(&variant.ident);
        tys.push(&fields.unnamed[0].ty);
    }
    let op_dyn = quote!(::gpt_rs_ir::op::OpDyn);
    Ok(quote! {
        /// The dialect an op belongs to.
        #[derive(::std::fmt::Debug, ::std::clone::Clone, ::std::marker::Copy, ::std::cmp::PartialEq, ::std::cmp::Eq, ::std::hash::Hash)]
        #vis enum #kind {
            #(#variants,)*
        }

        impl #ident {
            /// The dialect of the op.
            pub fn dialect(&self) -> #kind {
                match self { #(Self::#variants(_) => #kind::#variants,)* }
            }

            pub fn name(&self) -> ::std::borrow::Cow<'static, str> {
                match self { #(Self::#variants(op) => op.name(),)* }
            }

            pub fn effects(&self) -> ::gpt_rs_ir::op::Effects {
                match self { #(Self::#variants(op) => op.effects(),)* }
            }

            pub fn traits(&self) -> ::gpt_rs_ir::traits::Traits {
                match self { #(Self::#variants(op) => op.traits(),)* }
            }

            /// The op's operand groups, in operand order.
            pub fn operand_groups(&self) -> &'static [::gpt_rs_ir::op::Group] {
                match self { #(Self::#variants(op) => op.operand_groups(),)* }
            }

            /// The op's result groups, in result order.
            pub fn result_groups(&self) -> &'static [::gpt_rs_ir::op::Group] {
                match self { #(Self::#variants(op) => op.result_groups(),)* }
            }

            pub fn region_names(&self) -> &'static [&'static str] {
                match self { #(Self::#variants(op) => op.region_names(),)* }
            }

            pub fn is_region_terminator(&self, index: usize, op: &#ident) -> bool {
                match self { #(Self::#variants(inner) => inner.is_region_terminator(index, op),)* }
            }

            pub fn print_attrs(&self, out: &mut ::std::vec::Vec<(&'static str, ::std::string::String)>) {
                match self { #(Self::#variants(op) => op.print_attrs(out),)* }
            }

            /// Parses the op named `name` and its attribute dictionary, or returns `None` when no
            /// dialect has an op of that name.
            pub fn parse(
                name: &str,
                c: &mut ::gpt_rs_ir::text::Cursor<'_>,
            ) -> ::std::option::Option<::anyhow::Result<Self>> {
                #(
                    if let ::std::option::Option::Some(result) = <#tys>::parse(name, c) {
                        return ::std::option::Option::Some(result.map(Self::#variants));
                    }
                )*
                ::std::option::Option::None
            }

            /// Parses the signature of the symbol op named `name`, or returns `None` when no
            /// dialect has a symbol op of that name.
            pub fn parse_symbol(
                name: &str,
                c: &mut ::gpt_rs_ir::text::Cursor<'_>,
            ) -> ::std::option::Option<::anyhow::Result<::gpt_rs_ir::op::ParsedSymbol<Self>>> {
                #(
                    if let ::std::option::Option::Some(result) = <#tys>::parse_symbol(name, c) {
                        return ::std::option::Option::Some(result.map(|(op, args)| (Self::#variants(op), args)));
                    }
                )*
                ::std::option::Option::None
            }
        }

        impl #op_dyn for #ident {
            fn dyn_any(&self) -> &dyn ::std::any::Any {
                match self { #(Self::#variants(op) => #op_dyn::dyn_any(op),)* }
            }

            fn dyn_result_types(
                &self,
                ins: &[&::gpt_rs_ir::types::Type],
                segments: ::std::option::Option<&[usize]>,
                regions: &[::gpt_rs_ir::module::Region],
                module: &::gpt_rs_ir::module::Module,
            ) -> ::anyhow::Result<(::gpt_rs_ir::op::Flat<::gpt_rs_ir::types::Type>, ::std::option::Option<::std::vec::Vec<usize>>)> {
                match self { #(Self::#variants(op) => #op_dyn::dyn_result_types(op, ins, segments, regions, module),)* }
            }

            fn dyn_region_signatures(
                &self,
                ins: &[&::gpt_rs_ir::types::Type],
                segments: ::std::option::Option<&[usize]>,
            ) -> ::anyhow::Result<::std::vec::Vec<::gpt_rs_ir::op::RegionSig>> {
                match self { #(Self::#variants(op) => #op_dyn::dyn_region_signatures(op, ins, segments),)* }
            }

            fn dyn_fold(
                &self,
                ins: &[::gpt_rs_ir::op::FoldIn<'_>],
                segments: ::std::option::Option<&[usize]>,
                results: &[::gpt_rs_ir::types::Type],
            ) -> ::std::option::Option<::std::vec::Vec<::gpt_rs_ir::op::Folded>> {
                match self { #(Self::#variants(op) => #op_dyn::dyn_fold(op, ins, segments, results),)* }
            }

            fn dyn_eval(
                &self,
                ins: &[&::gpt_rs_ir::host::HostValue],
                segments: ::std::option::Option<&[usize]>,
                regions: &[::gpt_rs_ir::module::Region],
                results: &[::gpt_rs_ir::types::Type],
                runner: &mut dyn ::gpt_rs_ir::op::RegionRunner,
            ) -> ::anyhow::Result<::std::vec::Vec<::gpt_rs_ir::host::HostValue>> {
                match self { #(Self::#variants(op) => #op_dyn::dyn_eval(op, ins, segments, regions, results, runner),)* }
            }

            fn dyn_constant(&self) -> ::std::option::Option<&::gpt_rs_ir::dense::Dense> {
                match self { #(Self::#variants(op) => #op_dyn::dyn_constant(op),)* }
            }

            fn dyn_symbol(&self) -> ::std::option::Option<&dyn ::gpt_rs_ir::op::SymbolOp> {
                match self { #(Self::#variants(op) => #op_dyn::dyn_symbol(op),)* }
            }
        }
    })
}

// ------------------------------------------------------------------------------------------------
// #[pass]

/// The arguments of `#[pass(name = "...", stats(counter = "doc", ...))]`.
struct PassArgs {
    name: LitStr,
    stats: Vec<(Ident, LitStr)>,
}

impl Parse for PassArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut name = None;
        let mut stats = Vec::new();
        while !input.is_empty() {
            let key: Ident = input.parse()?;
            match key.to_string().as_str() {
                "name" => {
                    input.parse::<Token![=]>()?;
                    name = Some(input.parse::<LitStr>()?);
                }
                "stats" => {
                    let content;
                    syn::parenthesized!(content in input);
                    while !content.is_empty() {
                        let stat: Ident = content.parse()?;
                        content.parse::<Token![=]>()?;
                        stats.push((stat, content.parse::<LitStr>()?));
                        if !content.is_empty() {
                            content.parse::<Token![,]>()?;
                        }
                    }
                }
                other => {
                    return Err(syn::Error::new_spanned(
                        &key,
                        format!("unknown #[pass] argument `{other}`; expected `name` or `stats`"),
                    ))
                }
            }
            if !input.is_empty() {
                input.parse::<Token![,]>()?;
            }
        }
        let name = name.ok_or_else(|| input.error("#[pass] needs `name = \"...\"`"))?;
        Ok(Self { name, stats })
    }
}

/// The text of the doc comments in `attrs`, one line each, without the space after `///`.
fn doc_text(attrs: &[syn::Attribute]) -> String {
    let lines: Vec<String> = attrs
        .iter()
        .filter(|a| a.path().is_ident("doc"))
        .filter_map(|a| match &a.meta {
            syn::Meta::NameValue(nv) => match &nv.value {
                Expr::Lit(syn::ExprLit {
                    lit: syn::Lit::Str(s),
                    ..
                }) => Some(s.value()),
                _ => None,
            },
            _ => None,
        })
        .map(|line| line.strip_prefix(' ').unwrap_or(&line).to_string())
        .collect();
    lines.join("\n").trim().to_string()
}

#[proc_macro_attribute]
pub fn pass(args: TokenStream, input: TokenStream) -> TokenStream {
    let args = parse_macro_input!(args as PassArgs);
    let input = parse_macro_input!(input as DeriveInput);
    expand_pass(args, input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

fn expand_pass(args: PassArgs, input: DeriveInput) -> syn::Result<TokenStream2> {
    let ident = &input.ident;
    let name = &args.name;
    let Data::Struct(data) = &input.data else {
        return Err(syn::Error::new_spanned(&input, "#[pass] goes on a struct"));
    };
    let doc = doc_text(&input.attrs);
    if doc.is_empty() {
        return Err(syn::Error::new_spanned(
            &input.ident,
            "a pass needs a doc comment: its first paragraph is the summary in the pass catalog",
        ));
    }
    let krate = quote!(::gpt_rs_ir::passes);

    // Options: the named fields, each with a doc comment, defaults from `Default`.
    let options: Vec<(&Ident, &SynType, String)> = match &data.fields {
        Fields::Unit => Vec::new(),
        Fields::Named(fields) => fields
            .named
            .iter()
            .map(|f| {
                let field = f.ident.as_ref().expect("a named field");
                let doc = doc_text(&f.attrs);
                if doc.is_empty() {
                    return Err(syn::Error::new_spanned(
                        field,
                        "a pass option needs a doc comment",
                    ));
                }
                Ok((field, &f.ty, doc))
            })
            .collect::<syn::Result<_>>()?,
        Fields::Unnamed(_) => {
            return Err(syn::Error::new_spanned(
                &input,
                "a pass is a unit struct or has named fields, its options",
            ))
        }
    };
    let value = if options.is_empty() {
        quote!(#ident)
    } else {
        quote!(<#ident as ::std::default::Default>::default())
    };

    let upper = snake_case(&ident.to_string()).to_uppercase();
    let make_fn = format_ident!("__GPT_RS_PASS_MAKE_{}", upper);
    let entry = format_ident!("__GPT_RS_PASS_{}", upper);

    let option_list = options
        .iter()
        .map(|(f, ty, _)| format!("{f}: {}", quote!(#ty).to_string().replace(' ', "")))
        .collect::<Vec<_>>()
        .join(", ");
    let set_arms = options.iter().map(|(f, ty, _)| {
        let key = f.to_string();
        quote!(#key => pass.#f = <#ty as ::gpt_rs_ir::attr::Attr>::parse(c)?,)
    });
    let print_changed = options.iter().map(|(f, ty, _)| {
        let key = f.to_string();
        quote! {
            if self.#f != defaults.#f {
                let mut text = ::std::string::String::from(concat!(#key, " = "));
                <#ty as ::gpt_rs_ir::attr::Attr>::print(&self.#f, &mut text);
                changed.push(text);
            }
        }
    });
    let default_fns: Vec<Ident> = options
        .iter()
        .map(|(f, _, _)| {
            format_ident!(
                "__GPT_RS_PASS_DEFAULT_{}_{}",
                upper,
                f.to_string().to_uppercase()
            )
        })
        .collect();
    let default_defs = options.iter().zip(&default_fns).map(|((f, ty, _), def)| {
        quote! {
            #[allow(non_snake_case)]
            fn #def() -> ::std::string::String {
                let mut text = ::std::string::String::new();
                <#ty as ::gpt_rs_ir::attr::Attr>::print(&#value.#f, &mut text);
                text
            }
        }
    });
    let option_infos = options.iter().zip(&default_fns).map(|((f, ty, doc), def)| {
        let key = f.to_string();
        let ty_text = quote!(#ty).to_string().replace(' ', "");
        quote!(#krate::OptionInfo { name: #key, ty: #ty_text, default: #def, doc: #doc })
    });
    let parse_options = if options.is_empty() {
        quote! {
            if c.eat("{") && !c.eat("}") {
                return ::std::result::Result::Err(c.error(concat!("pass `", #name, "` takes no options")));
            }
        }
    } else {
        quote! {
            if c.eat("{") {
                c.list("}", |c| {
                    let key = c.ident()?;
                    c.expect("=")?;
                    match key {
                        #(#set_arms)*
                        other => return ::std::result::Result::Err(c.error(format!(
                            "pass `{}` has no option `{other}`; its options: {}", #name, #option_list
                        ))),
                    }
                    ::std::result::Result::Ok(())
                })?;
            }
        }
    };
    let options_body = if options.is_empty() {
        quote!(::std::string::String::new())
    } else {
        quote! {
            let defaults = #value;
            let mut changed: ::std::vec::Vec<::std::string::String> = ::std::vec::Vec::new();
            #(#print_changed)*
            if changed.is_empty() {
                ::std::string::String::new()
            } else {
                format!("{{{}}}", changed.join(", "))
            }
        }
    };

    let stat_consts = args.stats.iter().map(|(stat, _)| {
        let konst = format_ident!("{}", stat.to_string().to_uppercase());
        let key = stat.to_string();
        quote!(pub const #konst: #krate::Stat = #krate::Stat::new(#key);)
    });
    let stat_infos = args.stats.iter().map(|(stat, doc)| {
        let key = stat.to_string();
        quote!(#krate::StatInfo { name: #key, doc: #doc })
    });
    let stats_impl = if args.stats.is_empty() {
        quote!()
    } else {
        quote!(impl #ident { #(#stat_consts)* })
    };

    Ok(quote! {
        #input

        impl #krate::Registered for #ident {
            fn name(&self) -> &'static str {
                #name
            }

            fn options(&self) -> ::std::string::String {
                #options_body
            }
        }

        #stats_impl

        #(#default_defs)*

        #[allow(non_snake_case)]
        fn #make_fn(
            c: &mut ::gpt_rs_ir::text::Cursor<'_>,
        ) -> ::anyhow::Result<::std::boxed::Box<dyn #krate::Pass>> {
            #[allow(unused_mut)]
            let mut pass = #value;
            #parse_options
            ::std::result::Result::Ok(::std::boxed::Box::new(pass))
        }

        #[::gpt_rs_ir::linkme::distributed_slice(#krate::PASSES)]
        #[linkme(crate = ::gpt_rs_ir::linkme)]
        static #entry: #krate::PassEntry = #krate::PassEntry {
            name: #name,
            doc: #doc,
            rust: ::std::stringify!(#ident),
            file: ::std::file!(),
            options: &[#(#option_infos),*],
            stats: &[#(#stat_infos),*],
            make: #make_fn,
        };
    })
}

// ------------------------------------------------------------------------------------------------
// Documented rewrites: #[derive(Doc)], #[pattern] and #[fold]

/// `impl Doc` for the struct of `input`, from its doc comment, which `what` needs.
fn doc_impl(input: &DeriveInput, what: &str) -> syn::Result<TokenStream2> {
    let doc = doc_text(&input.attrs);
    if doc.is_empty() {
        return Err(syn::Error::new_spanned(
            &input.ident,
            format!("{what} needs a doc comment, which the reference pages show"),
        ));
    }
    let ident = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();
    Ok(quote! {
        impl #impl_generics ::gpt_rs_ir::doc::Doc for #ident #ty_generics #where_clause {
            fn name(&self) -> &'static str {
                ::std::stringify!(#ident)
            }
            fn doc(&self) -> &'static str {
                #doc
            }
            fn file(&self) -> &'static str {
                ::std::file!()
            }
        }
    })
}

#[proc_macro_derive(Doc)]
pub fn derive_doc(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    doc_impl(&input, "an op trait")
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

/// The arguments of `#[pattern]`: none, `canonical`, or for a pattern generic over its root op,
/// `canonical(Op, ...)` with the ops it registers for.
struct PatternArgs {
    canonical: Option<Vec<SynType>>,
}

impl Parse for PatternArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        if input.is_empty() {
            return Ok(Self { canonical: None });
        }
        let key: Ident = input.parse()?;
        if key != "canonical" {
            return Err(syn::Error::new_spanned(
                &key,
                "the #[pattern] argument is `canonical` or `canonical(Op, ...)`",
            ));
        }
        let mut roots = Vec::new();
        if !input.is_empty() {
            let content;
            syn::parenthesized!(content in input);
            roots = Punctuated::<SynType, Token![,]>::parse_terminated(&content)?
                .into_iter()
                .collect();
        }
        Ok(Self {
            canonical: Some(roots),
        })
    }
}

/// `#[pattern]` on a pattern struct implements `Doc` from its doc comment. `#[pattern(canonical)]`
/// also registers the pattern with `canonicalize`. A pattern generic over its root op names the
/// ops it registers for: `#[pattern(canonical(Op, ...))]`.
#[proc_macro_attribute]
pub fn pattern(args: TokenStream, input: TokenStream) -> TokenStream {
    let args = parse_macro_input!(args as PatternArgs);
    let input = parse_macro_input!(input as DeriveInput);
    expand_pattern(args, input)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

fn expand_pattern(args: PatternArgs, input: DeriveInput) -> syn::Result<TokenStream2> {
    let doc = doc_impl(&input, "a pattern")?;
    let ident = &input.ident;
    let Data::Struct(data) = &input.data else {
        return Err(syn::Error::new_spanned(
            &input,
            "#[pattern] goes on a struct",
        ));
    };
    let Some(roots) = &args.canonical else {
        return Ok(quote!(#input #doc));
    };
    let adds: Vec<TokenStream2> = match (&data.fields, roots.is_empty()) {
        (Fields::Unit, true) if input.generics.params.is_empty() => vec![quote!(set.add(#ident);)],
        (Fields::Unnamed(fields), false) if fields.unnamed.len() == 1 => roots
            .iter()
            .map(|root| quote!(set.add(#ident::<#root>(::std::marker::PhantomData));))
            .collect(),
        _ => {
            return Err(syn::Error::new_spanned(
                ident,
                "a canonical pattern is a unit struct, or a struct of one `PhantomData` generic \
                 over its root op that names the ops: `#[pattern(canonical(Op, ...))]`",
            ))
        }
    };
    let add_fn = format_ident!("__gpt_rs_canonical_{}", snake_case(&ident.to_string()));
    let entry = format_ident!("__GPT_RS_CANONICAL_{}", ident.to_string().to_uppercase());
    Ok(quote! {
        #input
        #doc

        fn #add_fn(set: &mut ::gpt_rs_ir::rewrite::PatternSet) {
            #(#adds)*
        }

        #[::gpt_rs_ir::linkme::distributed_slice(::gpt_rs_ir::rewrite::CANONICAL_PATTERNS)]
        #[linkme(crate = ::gpt_rs_ir::linkme)]
        static #entry: fn(&mut ::gpt_rs_ir::rewrite::PatternSet) = #add_fn;
    })
}

/// `#[fold]` on `impl Fold for <op>` registers the op's folder, with the doc comment of the impl,
/// which says what the op folds to.
#[proc_macro_attribute]
pub fn fold(args: TokenStream, input: TokenStream) -> TokenStream {
    let args = TokenStream2::from(args);
    let item = parse_macro_input!(input as syn::ItemImpl);
    expand_fold(args, item)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

fn expand_fold(args: TokenStream2, item: syn::ItemImpl) -> syn::Result<TokenStream2> {
    if !args.is_empty() {
        return Err(syn::Error::new_spanned(args, "#[fold] takes no arguments"));
    }
    let is_fold = item
        .trait_
        .as_ref()
        .and_then(|(_, path, _)| path.segments.last())
        .is_some_and(|segment| segment.ident == "Fold");
    let op = match &*item.self_ty {
        SynType::Path(path) if is_fold => path.path.segments.last().map(|s| s.ident.clone()),
        _ => None,
    };
    let Some(op) = op else {
        return Err(syn::Error::new_spanned(
            &item.self_ty,
            "#[fold] goes on `impl Fold for <op>`",
        ));
    };
    let doc = doc_text(&item.attrs);
    if doc.is_empty() {
        return Err(syn::Error::new_spanned(
            &item.self_ty,
            "a folder needs a doc comment: what the op folds to, which the reference pages show",
        ));
    }
    let ty = &item.self_ty;
    let name = op.to_string();
    let entry = format_ident!("__GPT_RS_FOLD_{}", name.to_uppercase());
    Ok(quote! {
        #item

        #[::gpt_rs_ir::linkme::distributed_slice(::gpt_rs_ir::op::FOLDS)]
        #[linkme(crate = ::gpt_rs_ir::linkme)]
        static #entry: ::gpt_rs_ir::op::FoldEntry = ::gpt_rs_ir::op::FoldEntry {
            op: ::std::any::TypeId::of::<#ty>,
            rust: #name,
            file: ::std::file!(),
            doc: #doc,
            fold: ::gpt_rs_ir::op::fold_erased::<#ty>,
        };
    })
}

/// Registers `impl OpCost for <op>`: the cost of an op that the default model gets wrong.
#[proc_macro_attribute]
pub fn cost(args: TokenStream, input: TokenStream) -> TokenStream {
    let args = TokenStream2::from(args);
    let item = parse_macro_input!(input as syn::ItemImpl);
    expand_cost(args, item)
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

fn expand_cost(args: TokenStream2, item: syn::ItemImpl) -> syn::Result<TokenStream2> {
    if !args.is_empty() {
        return Err(syn::Error::new_spanned(args, "#[cost] takes no arguments"));
    }
    let is_cost = item
        .trait_
        .as_ref()
        .and_then(|(_, path, _)| path.segments.last())
        .is_some_and(|segment| segment.ident == "OpCost");
    let op = match &*item.self_ty {
        SynType::Path(path) if is_cost => path.path.segments.last().map(|s| s.ident.clone()),
        _ => None,
    };
    let Some(op) = op else {
        return Err(syn::Error::new_spanned(
            &item.self_ty,
            "#[cost] goes on `impl OpCost for <op>`",
        ));
    };
    if doc_text(&item.attrs).is_empty() {
        return Err(syn::Error::new_spanned(
            &item.self_ty,
            "a cost needs a doc comment: what the op costs, unlike the default model",
        ));
    }
    let ty = &item.self_ty;
    let name = op.to_string();
    let entry = format_ident!("__GPT_RS_COST_{}", name.to_uppercase());
    Ok(quote! {
        #item

        #[::gpt_rs_ir::linkme::distributed_slice(::gpt_rs_ir::cost::COSTS)]
        #[linkme(crate = ::gpt_rs_ir::linkme)]
        static #entry: ::gpt_rs_ir::cost::CostEntry = ::gpt_rs_ir::cost::CostEntry {
            op: ::std::any::TypeId::of::<#ty>,
            rust: #name,
            file: ::std::file!(),
            cost: ::gpt_rs_ir::cost::cost_erased::<#ty>,
        };
    })
}

/// One node of a [`matched!`] pattern.
enum DagPattern {
    /// `_`: any value.
    Any,
    /// `name`: the value, bound as a `ValueRef`.
    Value(Ident),
    /// `name @ Op(operands..)`, or `Op(operands..)` without a name: the op that defines the value,
    /// bound as a `Matched<Op>`. Without parentheses its operands go unchecked.
    Op {
        name: Option<Ident>,
        path: syn::Path,
        operands: Option<Vec<DagPattern>>,
    },
}

impl Parse for DagPattern {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        if input.peek(Token![_]) {
            input.parse::<Token![_]>()?;
            return Ok(DagPattern::Any);
        }
        let name = if input.peek(Ident) && input.peek2(Token![@]) {
            let name: Ident = input.parse()?;
            input.parse::<Token![@]>()?;
            Some(name)
        } else {
            None
        };
        let path: syn::Path = input.parse()?;
        let last = path
            .segments
            .last()
            .ok_or_else(|| input.error("expected an op or a name"))?;
        let is_op = last
            .ident
            .to_string()
            .starts_with(|c: char| c.is_ascii_uppercase());
        if !is_op {
            return match (name, path.get_ident()) {
                (None, Some(ident)) => Ok(DagPattern::Value(ident.clone())),
                _ => Err(syn::Error::new_spanned(
                    &path,
                    "expected an op type, which starts with an uppercase letter, or a value name",
                )),
            };
        }
        let operands = if input.peek(syn::token::Paren) {
            let content;
            syn::parenthesized!(content in input);
            let list = Punctuated::<DagPattern, Token![,]>::parse_terminated(&content)?;
            Some(list.into_iter().collect())
        } else {
            None
        };
        Ok(DagPattern::Op {
            name,
            path,
            operands,
        })
    }
}

/// `let <pattern> = <value> else <block>` of [`matched!`].
struct DagLet {
    pattern: DagPattern,
    value: Expr,
    otherwise: syn::Block,
}

impl Parse for DagLet {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        input.parse::<Token![let]>()?;
        let pattern = input.parse()?;
        input.parse::<Token![=]>()?;
        let value = Expr::parse_without_eager_brace(input)?;
        input.parse::<Token![else]>()?;
        let otherwise = input.parse()?;
        Ok(DagLet {
            pattern,
            value,
            otherwise,
        })
    }
}

/// Pushes the statements that match `pattern` against `value`, an expression of a `ValueRef`.
fn dag_statements(
    pattern: &DagPattern,
    value: TokenStream2,
    otherwise: &syn::Block,
    count: &mut usize,
    out: &mut Vec<TokenStream2>,
) {
    match pattern {
        DagPattern::Any => {}
        DagPattern::Value(name) => out.push(quote!(let #name = #value;)),
        DagPattern::Op {
            name,
            path,
            operands,
        } => {
            let binding = match name {
                Some(name) => name.clone(),
                None => {
                    *count += 1;
                    format_ident!(
                        "__matched_op_{}",
                        *count,
                        span = proc_macro2::Span::mixed_site()
                    )
                }
            };
            // With `return None` as the `else`, clippy would ask for `?`, which the macro leaves
            // to the caller's choice of failure.
            out.push(quote! {
                #[allow(clippy::question_mark)]
                let ::core::option::Option::Some(#binding) = (#value).defining_op::<#path>()
                else #otherwise;
            });
            for (index, operand) in operands.iter().flatten().enumerate() {
                let operand_value = quote!(#binding.operand(#index));
                dag_statements(operand, operand_value, otherwise, count, out);
            }
        }
    }
}

/// The `let`-`else` of an op DAG: binds the ops that define a value and its operands, as the
/// pattern names them, or runs the `else` block, which must not continue.
///
/// ```ignore
/// matched!(let transpose @ Transpose(dot @ DotGeneral(patches @ ExtractPatches(input), weight)) = value else {
///     return None;
/// });
/// ```
///
/// `name @ Op(..)` binds the `Matched<Op>` that defines the value, and `Op(..)` matches it without
/// a name. The patterns in parentheses match the op's operands in order, and an op without
/// parentheses leaves them unchecked. A plain lowercase `name` binds the value as a `ValueRef`,
/// and `_` matches any value.
#[proc_macro]
pub fn matched(input: TokenStream) -> TokenStream {
    let DagLet {
        pattern,
        value,
        otherwise,
    } = parse_macro_input!(input as DagLet);
    let root = format_ident!("__matched_value", span = proc_macro2::Span::mixed_site());
    let mut statements = vec![quote!(let #root = #value;)];
    let mut count = 0;
    dag_statements(
        &pattern,
        quote!(#root),
        &otherwise,
        &mut count,
        &mut statements,
    );
    quote!(#(#statements)*).into()
}
