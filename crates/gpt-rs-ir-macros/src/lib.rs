//! Derives for `gpt-rs-ir`.
//!
//! - `#[derive(Op)]` with `#[op("dialect.name", ...)]` on an op struct. It generates the op's
//!   named operand, result and region group structs and implements `OpKind` and `OpDyn`.
//! - `#[derive(Dialect)]` with `#[dialect(name = "...", op = Variant)]` on the enum listing a
//!   dialect's ops. It generates the dispatch over its ops and their place in the `Op` union, and
//!   it sets the enum's `DIALECT` constant to `name`.
//! - `#[derive(Attr)]` on a fieldless enum: one snake_case text name per variant. When the enum
//!   names the kinds of an op family, it also gives each variant the traits that its
//!   `#[traits(...)]` lists.
//! - `#[pass(name = "...", stats(...))]` on a pass struct. It gives the pass its name and
//!   statistics, makes its fields options, and registers it with its doc comment.
//! - `#[pattern]` on a pattern struct records its name and doc comment. `#[pattern(canonical)]`
//!   also registers it with `canonicalize`.
//! - `#[fold]` on `impl Fold for <op>` registers the op's folder with its doc comment.
//! - `#[derive(Doc)]` on an op trait struct records its name and doc comment.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
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

#[proc_macro_derive(Attr, attributes(traits))]
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
            "#[derive(Attr)] supports fieldless enums",
        ));
    };
    let name = &input.ident;
    let mut variants = Vec::new();
    let mut names = Vec::new();
    let mut traits = Vec::new();
    let mut docs = Vec::new();
    for variant in &data.variants {
        if !matches!(variant.fields, Fields::Unit) {
            return Err(syn::Error::new_spanned(
                variant,
                "#[derive(Attr)] supports fieldless variants only",
            ));
        }
        variants.push(variant.ident.clone());
        names.push(snake_case(&variant.ident.to_string()));
        docs.push(doc_text(&variant.attrs));
        let mut listed = Vec::new();
        for attr in variant.attrs.iter().filter(|a| a.path().is_ident("traits")) {
            listed.extend(attr.parse_args_with(Punctuated::<Expr, Token![,]>::parse_terminated)?);
        }
        traits.push(trait_slice(&listed));
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

        impl ::gpt_rs_ir::traits::KindTraits for #name {
            fn traits(self) -> &'static [&'static dyn ::gpt_rs_ir::traits::OpTrait] {
                match self { #(Self::#variants => #traits,)* }
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

/// A `'static` slice of the trait values `exprs`, each a constant expression.
fn trait_slice(exprs: &[Expr]) -> TokenStream2 {
    quote! {{
        const TRAITS: &[&dyn ::gpt_rs_ir::traits::OpTrait] = &[#(&#exprs),*];
        TRAITS
    }}
}

// ------------------------------------------------------------------------------------------------
// #[derive(Op)]

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

struct GroupSpec {
    name: Ident,
    kind: KindSpec,
    arity: AritySpec,
}

struct RegionSpec {
    name: Ident,
    terminator: SynType,
}

#[derive(Default)]
struct EffectSpec {
    opaque: bool,
    rng: bool,
    alloc: bool,
    reads: Vec<Ident>,
    writes: Vec<Ident>,
}

struct OpArgs {
    name: LitStr,
    operands: Vec<GroupSpec>,
    results: Vec<GroupSpec>,
    regions: Vec<RegionSpec>,
    effects: EffectSpec,
    traits: Vec<Expr>,
}

fn parse_kind(input: ParseStream) -> syn::Result<(KindSpec, AritySpec)> {
    let ident: Ident = input.parse()?;
    let arity = match ident.to_string().as_str() {
        "Variadic" => Some(AritySpec::Variadic),
        "Optional" => Some(AritySpec::Optional),
        _ => None,
    };
    if let Some(arity) = arity {
        input.parse::<Token![<]>()?;
        let (kind, nested) = parse_kind(input)?;
        if nested != AritySpec::One {
            return Err(syn::Error::new(
                ident.span(),
                format!("`{ident}` cannot hold another Variadic or Optional"),
            ));
        }
        input.parse::<Token![>]>()?;
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
                     Optional<..> or Variadic<..>"
                ),
            ))
        }
    };
    Ok((kind, AritySpec::One))
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

fn parse_group(input: ParseStream) -> syn::Result<GroupSpec> {
    let name: Ident = input.parse()?;
    input.parse::<Token![:]>()?;
    let (kind, arity) = parse_kind(input)?;
    Ok(GroupSpec { name, kind, arity })
}

fn parse_region(input: ParseStream) -> syn::Result<RegionSpec> {
    let name: Ident = input.parse()?;
    input.parse::<Token![:]>()?;
    let block: Ident = input.parse()?;
    if block != "Block" {
        return Err(syn::Error::new(block.span(), "expected Block<Terminator>"));
    }
    input.parse::<Token![<]>()?;
    let terminator: SynType = input.parse()?;
    input.parse::<Token![>]>()?;
    Ok(RegionSpec { name, terminator })
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
            operands: Vec::new(),
            results: Vec::new(),
            regions: Vec::new(),
            effects: EffectSpec::default(),
            traits: Vec::new(),
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
                "operands" => args.operands = parse_list(input, parse_group)?,
                "results" => args.results = parse_list(input, parse_group)?,
                "regions" => args.regions = parse_list(input, parse_region)?,
                "effects" => args.effects = parse_effects(input)?,
                "traits" => args.traits = parse_list(input, Expr::parse)?,
                other => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!(
                            "unknown #[op] clause `{other}`, expected operands, results, \
                             regions, effects or traits"
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

/// The name prefix and the field of an op family, named `"dialect.{field}"`: the field's
/// mnemonic completes the name.
fn family<'a>(
    name: &LitStr,
    fields: &'a [FieldSpec],
) -> syn::Result<Option<(String, &'a FieldSpec)>> {
    let pattern = name.value();
    let Some((prefix, rest)) = pattern.split_once('{') else {
        return Ok(None);
    };
    let Some(field) = rest.strip_suffix('}') else {
        return Err(syn::Error::new(
            name.span(),
            "the `{field}` placeholder must end the op name",
        ));
    };
    let Some(spec) = fields.iter().find(|f| f.key == field) else {
        return Err(syn::Error::new(
            name.span(),
            format!("op name placeholder `{{{field}}}` names no field"),
        ));
    };
    Ok(Some((prefix.to_string(), spec)))
}

#[proc_macro_derive(Op, attributes(op, attr))]
pub fn derive_op(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match expand_op(&input) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
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

/// Builder conversions into the `Values` view.
fn operands_from(struct_name: &Ident, groups: &[GroupSpec]) -> TokenStream2 {
    let values = quote!(::gpt_rs_ir::op::Values);
    let value = quote!(::gpt_rs_ir::module::Value);
    if let [group] = groups {
        let name = &group.name;
        let ty = match group.arity {
            AritySpec::One => value,
            AritySpec::Optional => quote!(::std::option::Option<#value>),
            AritySpec::Variadic => quote!(::std::vec::Vec<#value>),
        };
        return quote! {
            impl ::std::convert::From<#ty> for #struct_name<#values> {
                fn from(value: #ty) -> Self { #struct_name { #name: value } }
            }
        };
    }
    if groups.is_empty() || groups.iter().any(|g| g.arity != AritySpec::One) {
        return quote!();
    }
    let names = groups.iter().map(|g| &g.name);
    let tys = groups.iter().map(|_| &value);
    let tuple = quote!((#(#tys),*));
    let idx = (0..groups.len()).map(syn::Index::from);
    quote! {
        impl ::std::convert::From<#tuple> for #struct_name<#values> {
            fn from(values: #tuple) -> Self {
                #struct_name { #(#names: values.#idx),* }
            }
        }
    }
}

fn expand_op(input: &DeriveInput) -> syn::Result<TokenStream2> {
    let args: OpArgs = one_attr(
        input,
        "op",
        "#[derive(Op)] needs #[op(\"dialect.name\", ...)]",
    )?
    .parse_args()?;
    let Data::Struct(data) = &input.data else {
        return Err(syn::Error::new_spanned(
            input,
            "#[derive(Op)] supports structs",
        ));
    };
    let ident = &input.ident;
    let vis = &input.vis;
    let op_name = &args.name;
    let fields = data
        .fields
        .iter()
        .map(field_spec)
        .collect::<syn::Result<Vec<_>>>()?;
    let family = family(op_name, &fields)?;
    let family_ident = family.as_ref().map(|(_, f)| &f.ident);

    let operands_ident = format_ident!("{}Operands", ident);
    let results_ident = format_ident!("{}Results", ident);
    let regions_ident = format_ident!("{}Regions", ident);
    let operand_code = group_code(
        vis,
        &operands_ident,
        &args.operands,
        &format_ident!("OPERANDS"),
        "operand",
    );
    let result_code = group_code(
        vis,
        &results_ident,
        &args.results,
        &format_ident!("RESULTS"),
        "result",
    );
    let operands_from = operands_from(&operands_ident, &args.operands);

    let has_regions = !args.regions.is_empty();
    let region_names: Vec<String> = args
        .regions
        .iter()
        .map(|r| r.name.unraw().to_string())
        .collect();
    let region_fields: Vec<&Ident> = args.regions.iter().map(|r| &r.name).collect();
    let region_count = args.regions.len();
    let (regions_def, regions_ty, split_regions, flatten_regions) = if has_regions {
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
    } else {
        (
            quote!(),
            quote!(::gpt_rs_ir::op::NoRegions),
            quote! {
                ::anyhow::ensure!(flat.is_empty(), "expected no regions, got {}", flat.len());
                ::std::result::Result::Ok(::gpt_rs_ir::op::NoRegions)
            },
            quote! { ::std::vec::Vec::new() },
        )
    };
    let terminator_arms = args.regions.iter().enumerate().map(|(i, r)| {
        let term = &r.terminator;
        quote!(#i => <#term as ::gpt_rs_ir::op::InOp>::cast(op).is_some(),)
    });

    let (name_fn, name_match) = match &family {
        Some((prefix, f)) => {
            let (fid, ty) = (&f.ident, &f.ty);
            (
                quote! {
                    ::std::borrow::Cow::Owned(format!("{}{}", #prefix, ::gpt_rs_ir::attr::Mnemonic::mnemonic(self.#fid)))
                },
                quote! {
                    let rest = name.strip_prefix(#prefix)?;
                    let family: #ty = <#ty as ::gpt_rs_ir::attr::Mnemonic>::from_mnemonic(rest)?;
                },
            )
        }
        None => (
            quote!(::std::borrow::Cow::Borrowed(#op_name)),
            quote! {
                if name != #op_name { return ::std::option::Option::None; }
            },
        ),
    };

    let EffectSpec {
        opaque,
        rng,
        alloc,
        reads,
        writes,
    } = &args.effects;
    let group_bit = |group: &Ident| -> syn::Result<TokenStream2> {
        let index = args
            .operands
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
    let op_traits = trait_slice(&args.traits);
    let kind_traits = match family_ident {
        Some(fid) => quote!(::gpt_rs_ir::traits::KindTraits::traits(self.#fid)),
        None => quote!(&[]),
    };
    // The folder that `#[fold]` registers for this op, found once.
    let fold_body = quote! {
        static FOLDER: ::std::sync::OnceLock<::std::option::Option<::gpt_rs_ir::op::FoldFn>> =
            ::std::sync::OnceLock::new();
        let fold = (*FOLDER.get_or_init(|| {
            ::gpt_rs_ir::op::folder_of(::std::any::TypeId::of::<Self>())
        }))?;
        fold(self, ins, segments, results)
    };
    let declares = |flag: bool, name: &str| -> TokenStream2 {
        if flag {
            let marker = format_ident!("{}", name);
            quote!(impl ::gpt_rs_ir::op::#marker for #ident {})
        } else {
            quote!()
        }
    };
    let declare_regions = declares(has_regions, "DeclaresRegions");

    // Attributes: every field except the family field. `parse` reads each into a local
    // `<field>_slot`. The suffix keeps a field named `c`, `name` or `key` from shadowing a local
    // of `parse`.
    let attr_fields: Vec<&FieldSpec> = fields
        .iter()
        .filter(|f| family_ident != Some(&f.ident))
        .collect();
    let slot_of = |f: &FieldSpec| format_ident!("{}_slot", f.ident);
    let print_attrs = attr_fields.iter().map(|f| {
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
    let slot_decls = attr_fields.iter().map(|f| {
        let (slot, ty) = (slot_of(f), &f.ty);
        quote!(let mut #slot: ::std::option::Option<#ty> = ::std::option::Option::None;)
    });
    let key_arms = attr_fields.iter().map(|f| {
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
        if family_ident == Some(fid) {
            quote!(#fid: family)
        } else if f.default {
            quote!(#fid: #slot.unwrap_or_default())
        } else {
            quote!(#fid: #slot.ok_or_else(|| c.error(format!("{} needs attribute `{}`", name, #key)))?)
        }
    });

    let op_doc = doc_text(&input.attrs);
    let attr_infos = attr_fields.iter().map(|f| {
        let (key, ty, default) = (&f.key, &f.ty, f.default);
        let ty_text = quote!(#ty).to_string().replace(' ', "");
        quote!(::gpt_rs_ir::op::AttrInfo { name: #key, ty: #ty_text, default: #default })
    });
    let kinds_fn = match &family {
        Some((_, f)) => {
            let ty = &f.ty;
            quote! {{
                fn kinds() -> ::std::vec::Vec<::gpt_rs_ir::op::KindInfo> {
                    <#ty as ::gpt_rs_ir::attr::Mnemonic>::ALL
                        .iter()
                        .map(|&kind| ::gpt_rs_ir::op::KindInfo {
                            mnemonic: ::gpt_rs_ir::attr::Mnemonic::mnemonic(kind),
                            doc: ::gpt_rs_ir::attr::Mnemonic::doc(kind),
                            traits: ::gpt_rs_ir::traits::KindTraits::traits(kind),
                        })
                        .collect()
                }
                kinds
            }}
        }
        None => quote!(::std::vec::Vec::new),
    };

    let region_sigs = if has_regions {
        quote! {
            let typed = <Self as ::gpt_rs_ir::op::OpKind>::split_operands::<::gpt_rs_ir::op::Types<'_>>(ins.iter().copied(), segments)?;
            let sigs = <Self as ::gpt_rs_ir::op::RegionOpDef>::region_signatures(self, typed)?;
            ::std::result::Result::Ok(<Self as ::gpt_rs_ir::op::OpKind>::flatten_regions(sigs))
        }
    } else {
        quote!(::std::result::Result::Ok(::std::vec::Vec::new()))
    };

    let GroupCode {
        struct_def: operands_def,
        consts: operands_const,
        split: split_operands,
        flatten: flatten_operands,
    } = operand_code;
    let GroupCode {
        struct_def: results_def,
        consts: results_const,
        split: split_results,
        flatten: flatten_results,
    } = result_code;

    Ok(quote! {
        #operands_def
        #results_def
        #regions_def
        #operands_from
        #declare_regions

        impl ::gpt_rs_ir::op::OpKind for #ident {
            type Operands<V: ::gpt_rs_ir::op::View> = #operands_ident<V>;
            type Results<V: ::gpt_rs_ir::op::View> = #results_ident<V>;
            type Regions<R> = #regions_ty;

            #operands_const
            #results_const
            const REGIONS: &'static [&'static str] = &[#(#region_names),*];
            const INFO: ::gpt_rs_ir::op::OpInfo = ::gpt_rs_ir::op::OpInfo {
                name: #op_name,
                rust: ::std::stringify!(#ident),
                file: ::std::file!(),
                doc: #op_doc,
                operands: <Self as ::gpt_rs_ir::op::OpKind>::OPERANDS,
                results: <Self as ::gpt_rs_ir::op::OpKind>::RESULTS,
                regions: <Self as ::gpt_rs_ir::op::OpKind>::REGIONS,
                attrs: &[#(#attr_infos),*],
                traits: #op_traits,
                effects: ::gpt_rs_ir::op::Effects {
                    opaque: #opaque,
                    rng: #rng,
                    alloc: #alloc,
                    reads: 0 #(| #read_bits)*,
                    writes: 0 #(| #write_bits)*,
                },
                kinds: #kinds_fn,
            };

            fn name(&self) -> ::std::borrow::Cow<'static, str> { #name_fn }

            fn effects(&self) -> ::gpt_rs_ir::op::Effects {
                ::gpt_rs_ir::op::Effects {
                    opaque: #opaque,
                    rng: #rng,
                    alloc: #alloc,
                    reads: 0 #(| #read_bits)*,
                    writes: 0 #(| #write_bits)*,
                }
            }

            fn traits(&self) -> ::gpt_rs_ir::traits::Traits {
                ::gpt_rs_ir::traits::Traits::new(#op_traits, #kind_traits)
            }

            fn print_attrs(&self, out: &mut ::std::vec::Vec<(&'static str, ::std::string::String)>) {
                #(#print_attrs)*
            }

            fn parse(
                name: &str,
                c: &mut ::gpt_rs_ir::text::Cursor<'_>,
            ) -> ::std::option::Option<::anyhow::Result<Self>> {
                #name_match
                ::std::option::Option::Some((|| -> ::anyhow::Result<Self> {
                    #(#slot_decls)*
                    if c.eat("{") {
                        c.list("}", |c| {
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
                })())
            }

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

            fn is_region_terminator(index: usize, op: &::gpt_rs_ir::Op) -> bool {
                match index {
                    #(#terminator_arms)*
                    _ => false,
                }
            }
        }

        impl ::gpt_rs_ir::op::OpDyn for #ident {
            fn dyn_result_types(
                &self,
                ins: &[&::gpt_rs_ir::types::Type],
                segments: ::std::option::Option<&[usize]>,
                regions: &[::gpt_rs_ir::module::Region],
                module: &::gpt_rs_ir::module::Module,
            ) -> ::anyhow::Result<(::gpt_rs_ir::op::Flat<::gpt_rs_ir::types::Type>, ::std::option::Option<::std::vec::Vec<usize>>)> {
                let typed = <Self as ::gpt_rs_ir::op::OpKind>::split_operands::<::gpt_rs_ir::op::Types<'_>>(ins.iter().copied(), segments)?;
                let regions = <Self as ::gpt_rs_ir::op::OpKind>::split_regions(regions.to_vec())?;
                let out = <Self as ::gpt_rs_ir::op::OpDef>::result_types(
                    self,
                    typed,
                    &::gpt_rs_ir::op::TypeCx { module, regions },
                )?;
                ::std::result::Result::Ok(<Self as ::gpt_rs_ir::op::OpKind>::flatten_results::<::gpt_rs_ir::op::OwnedTypes>(out))
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
                let typed = <Self as ::gpt_rs_ir::op::OpKind>::split_operands::<::gpt_rs_ir::op::Host<'_>>(ins.iter().copied(), segments)?;
                let regions = <Self as ::gpt_rs_ir::op::OpKind>::split_regions(regions.to_vec())?;
                let mut cx = ::gpt_rs_ir::op::EvalCx { regions, results, runner };
                let out = <Self as ::gpt_rs_ir::op::Eval>::eval(self, typed, &mut cx)?;
                ::std::result::Result::Ok(<Self as ::gpt_rs_ir::op::OpKind>::flatten_results::<::gpt_rs_ir::op::OwnedHost>(out).0.into_vec())
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
    for variant in &data.variants {
        let ty = match &variant.fields {
            Fields::Unnamed(fields) if fields.unnamed.len() == 1 => &fields.unnamed[0].ty,
            _ => {
                return Err(syn::Error::new_spanned(
                    variant,
                    "dialect variants hold one op: `Name(Name)`",
                ))
            }
        };
        variants.push(&variant.ident);
        tys.push(ty);
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
                match self { #(Self::#variants(op) => #op_kind::name(op),)* }
            }

            pub fn effects(&self) -> ::gpt_rs_ir::op::Effects {
                match self { #(Self::#variants(op) => #op_kind::effects(op),)* }
            }

            pub fn traits(&self) -> ::gpt_rs_ir::traits::Traits {
                match self { #(Self::#variants(op) => #op_kind::traits(op),)* }
            }

            pub fn operand_groups(&self) -> &'static [::gpt_rs_ir::op::Group] {
                match self { #(Self::#variants(_) => <#tys as #op_kind>::OPERANDS,)* }
            }

            pub fn result_groups(&self) -> &'static [::gpt_rs_ir::op::Group] {
                match self { #(Self::#variants(_) => <#tys as #op_kind>::RESULTS,)* }
            }

            pub fn region_names(&self) -> &'static [&'static str] {
                match self { #(Self::#variants(_) => <#tys as #op_kind>::REGIONS,)* }
            }

            pub fn is_region_terminator(&self, index: usize, op: &::gpt_rs_ir::Op) -> bool {
                match self { #(Self::#variants(_) => <#tys as #op_kind>::is_region_terminator(index, op),)* }
            }

            pub fn print_attrs(&self, out: &mut ::std::vec::Vec<(&'static str, ::std::string::String)>) {
                match self { #(Self::#variants(op) => #op_kind::print_attrs(op, out),)* }
            }

            pub fn parse(
                name: &str,
                c: &mut ::gpt_rs_ir::text::Cursor<'_>,
            ) -> ::std::option::Option<::anyhow::Result<Self>> {
                #(
                    if let ::std::option::Option::Some(result) = <#tys as #op_kind>::parse(name, c) {
                        return ::std::option::Option::Some(result.map(Self::#variants));
                    }
                )*
                ::std::option::Option::None
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
        }

        #(
            impl ::std::convert::From<#tys> for ::gpt_rs_ir::Op {
                fn from(op: #tys) -> Self { ::gpt_rs_ir::Op::#op_variant(#ident::#variants(op)) }
            }

            impl ::gpt_rs_ir::op::InOp for #tys {
                fn cast(op: &::gpt_rs_ir::Op) -> ::std::option::Option<&Self> {
                    match op {
                        ::gpt_rs_ir::Op::#op_variant(#ident::#variants(inner)) => ::std::option::Option::Some(inner),
                        _ => ::std::option::Option::None,
                    }
                }
            }
        )*
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
