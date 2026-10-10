//! Expansion of `#[nn::module]` (documented on [`crate::module`]).

use proc_macro2::{Span, TokenStream};
use quote::{quote, ToTokens};
use syn::{
    parse_quote,
    punctuated::Punctuated,
    spanned::Spanned,
    visit_mut::{self, VisitMut},
    Attribute, Error, Expr, Fields, FnArg, GenericArgument, Generics, ImplItem, Item, ItemEnum,
    ItemImpl, ItemStruct, Lifetime, LitStr, Pat, PathArguments, Result, ReturnType, Token, Type,
};

const BACKEND: &str = "B";

/// Names a sublayer field cannot take: its generated method would shadow a module method.
const RESERVED: [&str; 4] = ["forward", "call", "visit_params", "visit_params_mut"];

pub(crate) fn expand(attr: TokenStream, item: TokenStream) -> Result<TokenStream> {
    if !attr.is_empty() {
        return Err(Error::new(attr.span(), "#[nn::module] takes no arguments"));
    }
    match syn::parse2::<Item>(item)? {
        Item::Struct(item) => expand_struct(item),
        Item::Enum(item) => expand_enum(item),
        Item::Impl(item) => expand_impl(item),
        item => Err(Error::new(
            item.span(),
            "#[nn::module] applies to a struct, an enum, or the impl that holds its `forward`",
        )),
    }
}

/// Field and variant options of `#[module(...)]`.
#[derive(Default)]
struct Options {
    config: bool,
    flatten: bool,
    rename: Option<Expr>,
}

impl Options {
    /// Parses and removes the `#[module(...)]` attributes.
    fn take(attrs: &mut Vec<Attribute>) -> Result<Self> {
        let mut options = Options::default();
        let mut result = Ok(());
        attrs.retain(|attr| {
            if !attr.path().is_ident("module") {
                return true;
            }
            let parsed = attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("config") {
                    options.config = true;
                } else if meta.path.is_ident("flatten") {
                    options.flatten = true;
                } else if meta.path.is_ident("rename") {
                    options.rename = Some(meta.value()?.parse()?);
                } else {
                    return Err(meta.error("expected `config`, `flatten`, or `rename = <&str>`"));
                }
                Ok(())
            });
            if let Err(err) = parsed {
                result = Err(err);
            }
            false
        });
        result.map(|()| options)
    }
}

/// What a struct field holds.
#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Tensor,
    Layer,
}

/// How a struct field wraps its tensors or sublayers.
#[derive(Clone, Copy, PartialEq)]
enum Wrap {
    Plain,
    Option,
    Vec,
}

/// A visited (non-config) struct field.
struct Visited {
    ident: syn::Ident,
    /// Parameter path segment (a `&str` expression).
    name: Expr,
    kind: Kind,
    wrap: Wrap,
    flatten: bool,
    /// Type of the tensor or sublayer, with `<B>` added.
    inner: Type,
}

/// Replaces `Tensor` in module and functional code with the device tensor of the backend `B`.
pub(crate) struct TensorSugar;

impl VisitMut for TensorSugar {
    fn visit_type_mut(&mut self, ty: &mut Type) {
        if is_bare(ty, "Tensor") {
            // `DeviceTensor` takes the position of `Tensor`, so the IDE resolves `Tensor` to it.
            let device = syn::Ident::new("DeviceTensor", ty.span());
            *ty = parse_quote!(::gpt_rs::tensor::#device<B>);
            return;
        }
        visit_mut::visit_type_mut(self, ty);
    }
}

/// Returns true when `ty` is the single-segment path `name` without generic arguments.
fn is_bare(ty: &Type, name: &str) -> bool {
    let Type::Path(path) = ty else { return false };
    path.qself.is_none()
        && path.path.leading_colon.is_none()
        && path.path.segments.len() == 1
        && path.path.segments[0].ident == name
        && path.path.segments[0].arguments.is_none()
}

/// Returns the type argument of `ty` if `ty` is `wrapper<T>`.
fn wrapped<'a>(ty: &'a mut Type, wrapper: &str) -> Option<&'a mut Type> {
    let Type::Path(path) = ty else { return None };
    if path.qself.is_some() || path.path.segments.len() != 1 {
        return None;
    }
    let segment = &mut path.path.segments[0];
    if segment.ident != wrapper {
        return None;
    }
    let PathArguments::AngleBracketed(args) = &mut segment.arguments else {
        return None;
    };
    if args.args.len() != 1 {
        return None;
    }
    match &mut args.args[0] {
        GenericArgument::Type(inner) => Some(inner),
        _ => None,
    }
}

/// Rewrites a sublayer or tensor type `T` to its backend-generic form (`T<B>`, `DeviceTensor<B>`).
fn backend_type(ty: &mut Type) -> Result<Kind> {
    if is_bare(ty, "Tensor") {
        TensorSugar.visit_type_mut(ty);
        return Ok(Kind::Tensor);
    }
    let span = ty.span();
    let error = || {
        Error::new(
            span,
            "#[nn::module] fields must be `Tensor`s or sublayers, whose backend argument the \
             macro adds. Mark plain data with `#[module(config)]`",
        )
    };
    let Type::Path(path) = &mut *ty else {
        return Err(error());
    };
    if path.qself.is_some() {
        return Err(error());
    }
    let last = path.path.segments.last_mut().expect("paths are non-empty");
    match &mut last.arguments {
        PathArguments::None => last.arguments = PathArguments::AngleBracketed(parse_quote!(<B>)),
        PathArguments::AngleBracketed(args) => args.args.insert(0, parse_quote!(B)),
        PathArguments::Parenthesized(_) => return Err(error()),
    }
    Ok(Kind::Layer)
}

/// Adds `B: PortableBackend + 'static` after the lifetime parameters of `generics`. Error messages
/// name the attribute as `macro_name`.
pub(crate) fn add_backend_param(generics: &mut Generics, macro_name: &str) -> Result<()> {
    if let Some(param) = generics.type_params().find(|p| p.ident == BACKEND) {
        return Err(Error::new(
            param.span(),
            format!("{macro_name} adds the backend parameter `B` itself"),
        ));
    }
    let position = generics.lifetimes().count();
    generics.params.insert(
        position,
        parse_quote!(B: ::gpt_rs::backend::PortableBackend + 'static),
    );
    Ok(())
}

fn expand_struct(mut item: ItemStruct) -> Result<TokenStream> {
    let Fields::Named(fields) = &mut item.fields else {
        return Err(Error::new(
            item.fields.span(),
            "#[nn::module] structs must have named fields",
        ));
    };
    let mut visited = Vec::new();
    for field in fields.named.iter_mut() {
        let options = Options::take(&mut field.attrs)?;
        let ident = field.ident.clone().expect("named field");
        if options.config {
            if options.flatten || options.rename.is_some() {
                return Err(Error::new(
                    ident.span(),
                    "`config` fields are not visited, so they take no `flatten` or `rename`",
                ));
            }
            TensorSugar.visit_type_mut(&mut field.ty);
            continue;
        }
        let (wrap, inner) = if let Some(inner) = wrapped(&mut field.ty, "Option") {
            (Wrap::Option, inner)
        } else if let Some(inner) = wrapped(&mut field.ty, "Vec") {
            (Wrap::Vec, inner)
        } else {
            (Wrap::Plain, &mut field.ty)
        };
        let kind = backend_type(inner)?;
        let inner = inner.clone();
        if kind == Kind::Layer && RESERVED.iter().any(|name| ident == name) {
            return Err(Error::new(
                ident.span(),
                format!("sublayer field `{ident}` would shadow the module method `{ident}`"),
            ));
        }
        if options.flatten && (kind != Kind::Layer || wrap != Wrap::Plain) {
            return Err(Error::new(
                ident.span(),
                "only a plain sublayer field can be `flatten`ed",
            ));
        }
        if options.flatten && options.rename.is_some() {
            return Err(Error::new(
                ident.span(),
                "a `flatten`ed field adds no path segment to rename",
            ));
        }
        let name = options.rename.unwrap_or_else(|| {
            let name = LitStr::new(&ident.to_string(), ident.span());
            parse_quote!(#name)
        });
        visited.push(Visited {
            ident,
            name,
            kind,
            wrap,
            flatten: options.flatten,
            inner,
        });
    }
    add_backend_param(&mut item.generics, "#[nn::module]")?;

    let ident = &item.ident;
    let (impl_generics, ty_generics, where_clause) = item.generics.split_for_impl();
    let visit = visited.iter().map(|field| visit_field(field, false));
    let visit_mut = visited.iter().map(|field| visit_field(field, true));
    let module = module_impl(
        ident,
        &item.generics,
        quote!(#(#visit)*),
        quote!(#(#visit_mut)*),
        visited.is_empty(),
    );
    let calls = visited.iter().filter_map(call_method);
    let paths = visited.iter().filter(|field| !field.flatten).map(|field| {
        let key = LitStr::new(&field.ident.to_string(), field.ident.span());
        let name = &field.name;
        quote!(#key => ::core::option::Option::Some(#name),)
    });
    Ok(quote! {
        #item
        #module
        impl #impl_generics #ident #ty_generics #where_clause {
            #(#calls)*

            /// The path segment that constructors enter for `field`, or `None` when it adds none.
            #[doc(hidden)]
            #[allow(dead_code)]
            fn __nn_field_path(field: &str) -> ::core::option::Option<&'static str> {
                match field {
                    #(#paths)*
                    _ => ::core::option::Option::None,
                }
            }
        }
    })
}

fn expand_enum(mut item: ItemEnum) -> Result<TokenStream> {
    let mut visit = Vec::new();
    let mut visit_mut = Vec::new();
    let mut paths = Vec::new();
    for variant in item.variants.iter_mut() {
        let options = Options::take(&mut variant.attrs)?;
        if let Some(name) = &options.rename {
            let key = LitStr::new(&variant.ident.to_string(), variant.ident.span());
            paths.push(quote!(#key => ::core::option::Option::Some(#name),));
        }
        if options.config || options.flatten {
            return Err(Error::new(
                variant.ident.span(),
                "enum variants take only `rename`, and without it they add no path segment",
            ));
        }
        let Fields::Unnamed(fields) = &mut variant.fields else {
            return Err(Error::new(
                variant.span(),
                "#[nn::module] enum variants must hold exactly one sublayer, `Variant(Layer)`",
            ));
        };
        if fields.unnamed.len() != 1 {
            return Err(Error::new(
                fields.span(),
                "#[nn::module] enum variants must hold exactly one sublayer, `Variant(Layer)`",
            ));
        }
        let field = &mut fields.unnamed[0];
        if backend_type(&mut field.ty)? != Kind::Layer {
            return Err(Error::new(
                field.ty.span(),
                "enum variants must hold sublayers",
            ));
        }
        let variant = &variant.ident;
        for (arms, visit_fn) in [
            (&mut visit, quote!(visit_params)),
            (&mut visit_mut, quote!(visit_params_mut)),
        ] {
            let body = match &options.rename {
                Some(name) => {
                    quote!(v.scoped(#name, |v| ::gpt_rs::module::Module::<B>::#visit_fn(layer, v)))
                }
                None => quote!(::gpt_rs::module::Module::<B>::#visit_fn(layer, v)),
            };
            arms.push(quote!(Self::#variant(layer) => #body,));
        }
    }
    add_backend_param(&mut item.generics, "#[nn::module]")?;
    let module = module_impl(
        &item.ident,
        &item.generics,
        quote!(match self { #(#visit)* }?;),
        quote!(match self { #(#visit_mut)* }?;),
        false,
    );
    let ident = &item.ident;
    let (impl_generics, ty_generics, where_clause) = item.generics.split_for_impl();
    Ok(quote! {
        #item
        #module
        impl #impl_generics #ident #ty_generics #where_clause {
            /// The path segment that constructors enter for `variant`, or `None` when it adds none.
            #[doc(hidden)]
            #[allow(dead_code)]
            fn __nn_variant_path(variant: &str) -> ::core::option::Option<&'static str> {
                match variant {
                    #(#paths)*
                    _ => ::core::option::Option::None,
                }
            }
        }
    })
}

/// Generates `Module<B>` with `NAME` and the given visitor bodies.
fn module_impl(
    ident: &syn::Ident,
    generics: &Generics,
    visit: TokenStream,
    visit_mut: TokenStream,
    empty: bool,
) -> TokenStream {
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
    let name = LitStr::new(&ident.to_string(), ident.span());
    let unused = empty.then(|| quote!(let _ = v;));
    quote! {
        impl #impl_generics ::gpt_rs::module::Module<B> for #ident #ty_generics #where_clause {
            const NAME: &'static str = #name;

            fn visit_params(
                &self,
                v: &mut ::gpt_rs::module::ParamVisitor<'_, B>,
            ) -> anyhow::Result<()> {
                #unused
                #visit
                Ok(())
            }

            fn visit_params_mut(
                &mut self,
                v: &mut ::gpt_rs::module::ParamVisitorMut<'_, B>,
            ) -> anyhow::Result<()> {
                #unused
                #visit_mut
                Ok(())
            }
        }
    }
}

/// Generates the visitor statement of one field.
fn visit_field(field: &Visited, mutable: bool) -> TokenStream {
    let Visited {
        ident, name, kind, ..
    } = field;
    let (access, iter, visit_fn) = if mutable {
        (
            quote!(&mut self.#ident),
            quote!(iter_mut),
            quote!(visit_params_mut),
        )
    } else {
        (quote!(&self.#ident), quote!(iter), quote!(visit_params))
    };
    let role = quote!(::gpt_rs::module::TensorRole::Parameter);
    let module = quote!(::gpt_rs::module::Module::<B>);
    let leaf = |value: TokenStream, name: TokenStream| match kind {
        Kind::Tensor => quote!(v.param(#name, #role, #value)),
        Kind::Layer => quote!(v.scoped(#name, |v| #module::#visit_fn(#value, v))),
    };
    match field.wrap {
        Wrap::Plain if field.flatten => quote!(#module::#visit_fn(#access, v)?;),
        Wrap::Plain => {
            let visit = leaf(access, name.to_token_stream());
            quote!(#visit?;)
        }
        Wrap::Option => {
            let visit = leaf(quote!(item), name.to_token_stream());
            quote! {
                if let Some(item) = #access {
                    #visit?;
                }
            }
        }
        Wrap::Vec => {
            let visit = leaf(quote!(item), quote!(&index.to_string()));
            quote! {
                v.scoped(#name, |v| {
                    for (index, item) in (#access).#iter().enumerate() {
                        #visit?;
                    }
                    Ok(())
                })?;
            }
        }
    }
}

/// Generates the method that runs a sublayer field (`self.up_proj(x)`). Only plain and optional
/// sublayers get one.
fn call_method(field: &Visited) -> Option<TokenStream> {
    if field.kind != Kind::Layer {
        return None;
    }
    let Visited { ident, inner, .. } = field;
    let layer = quote!(::gpt_rs::module::Layer<B>);
    let args = quote!(<#inner as #layer>::Args<'_>);
    let output = quote!(<#inner as #layer>::Output);
    let call = quote!(::gpt_rs::module::Layer::<B>::call);
    match field.wrap {
        Wrap::Plain => Some(quote! {
            /// Runs the sublayer inside its profiling scope (`Layer::call`).
            #[allow(dead_code)]
            fn #ident(&self, args: #args) -> #output {
                #call(&self.#ident, args)
            }
        }),
        Wrap::Option => Some(quote! {
            /// Runs the sublayer, when present, inside its profiling scope (`Layer::call`).
            #[allow(dead_code)]
            fn #ident(&self, args: #args) -> Option<#output> {
                self.#ident.as_ref().map(|layer| #call(layer, args))
            }
        }),
        Wrap::Vec => None,
    }
}

/// Gives elided argument lifetimes the lifetime of `Layer::Args`.
struct ArgsLifetime;

impl VisitMut for ArgsLifetime {
    fn visit_type_reference_mut(&mut self, reference: &mut syn::TypeReference) {
        if reference.lifetime.is_none() {
            reference.lifetime = Some(args_lifetime());
        }
        visit_mut::visit_type_reference_mut(self, reference);
    }

    fn visit_lifetime_mut(&mut self, lifetime: &mut Lifetime) {
        if lifetime.ident == "_" {
            *lifetime = args_lifetime();
        }
    }

    fn visit_type_bare_fn_mut(&mut self, _: &mut syn::TypeBareFn) {}

    fn visit_parenthesized_generic_arguments_mut(
        &mut self,
        _: &mut syn::ParenthesizedGenericArguments,
    ) {
    }
}

fn args_lifetime() -> Lifetime {
    Lifetime::new("'__args", Span::call_site())
}

fn expand_impl(mut item: ItemImpl) -> Result<TokenStream> {
    if let Some((_, path, _)) = &item.trait_ {
        return Err(Error::new(
            path.span(),
            "#[nn::module] applies to the inherent impl of a module",
        ));
    }
    let Type::Path(self_ty) = &mut *item.self_ty else {
        return Err(Error::new(item.self_ty.span(), "expected a module type"));
    };
    let last = self_ty
        .path
        .segments
        .last_mut()
        .expect("paths are non-empty");
    let self_ident = last.ident.clone();
    match &mut last.arguments {
        PathArguments::None => last.arguments = PathArguments::AngleBracketed(parse_quote!(<B>)),
        PathArguments::AngleBracketed(args) => {
            let position = args
                .args
                .iter()
                .take_while(|arg| matches!(arg, GenericArgument::Lifetime(_)))
                .count();
            args.args.insert(position, parse_quote!(B));
        }
        PathArguments::Parenthesized(args) => {
            return Err(Error::new(args.span(), "expected a module type"));
        }
    }
    add_backend_param(&mut item.generics, "#[nn::module]")?;
    TensorSugar.visit_item_impl_mut(&mut item);
    for impl_item in &mut item.items {
        if let ImplItem::Fn(function) = impl_item {
            if let Some(params) = params_argument(&function.sig) {
                ScopeFields {
                    params,
                    self_ident: self_ident.clone(),
                }
                .visit_block_mut(&mut function.block);
            }
        }
    }

    let position = item
        .items
        .iter()
        .position(|item| matches!(item, ImplItem::Fn(f) if f.sig.ident == "forward"));
    let layer = match position {
        Some(position) => {
            let ImplItem::Fn(forward) = item.items.remove(position) else {
                unreachable!("position found a fn");
            };
            Some(layer_impl(&item, forward)?)
        }
        None => None,
    };
    let inherent = (!item.items.is_empty()).then(|| item.to_token_stream());
    Ok(quote! {
        #layer
        #inherent
    })
}

/// The identifier of a `name: &mut Params<..>` argument of `sig`.
fn params_argument(sig: &syn::Signature) -> Option<syn::Ident> {
    sig.inputs.iter().find_map(|input| {
        let FnArg::Typed(input) = input else {
            return None;
        };
        let (Pat::Ident(pat), Type::Reference(reference)) = (&*input.pat, &*input.ty) else {
            return None;
        };
        let Type::Path(path) = &*reference.elem else {
            return None;
        };
        let last = path.path.segments.last()?;
        (reference.mutability.is_some() && last.ident == "Params").then(|| pat.ident.clone())
    })
}

/// Wraps each field initializer of a `Self { .. }` literal, and each `Self::Variant(..)` payload,
/// that reads the params argument, so that it runs inside that field's or variant's parameter path.
struct ScopeFields {
    params: syn::Ident,
    self_ident: syn::Ident,
}

impl ScopeFields {
    fn is_self(&self, path: &syn::Path) -> bool {
        path.segments.len() == 1
            && (path.segments[0].ident == "Self" || path.segments[0].ident == self.self_ident)
    }

    /// Whether `expr` names the params argument, so it can read parameters.
    fn reads_params(&self, expr: &Expr) -> bool {
        fn mentions(tokens: TokenStream, ident: &syn::Ident) -> bool {
            tokens.into_iter().any(|token| match token {
                proc_macro2::TokenTree::Ident(found) => found == *ident,
                proc_macro2::TokenTree::Group(group) => mentions(group.stream(), ident),
                _ => false,
            })
        }
        mentions(expr.to_token_stream(), &self.params)
    }

    fn scoped(&self, expr: &Expr, segment: TokenStream) -> Expr {
        let params = &self.params;
        parse_quote!({
            let mut __nn_scope = ::gpt_rs::nn::Params::__enter(#params, #segment);
            let #params = ::core::ops::DerefMut::deref_mut(&mut __nn_scope);
            #expr
        })
    }
}

impl VisitMut for ScopeFields {
    fn visit_expr_mut(&mut self, expr: &mut Expr) {
        visit_mut::visit_expr_mut(self, expr);
        match expr {
            Expr::Struct(literal) if self.is_self(&literal.path) => {
                for field in literal.fields.iter_mut() {
                    let syn::Member::Named(name) = &field.member else {
                        continue;
                    };
                    if field.colon_token.is_none() || !self.reads_params(&field.expr) {
                        continue;
                    }
                    let key = LitStr::new(&name.to_string(), name.span());
                    field.expr = self.scoped(&field.expr, quote!(Self::__nn_field_path(#key)));
                }
            }
            Expr::Call(call) if call.args.len() == 1 => {
                let Expr::Path(func) = &*call.func else {
                    return;
                };
                let segments = &func.path.segments;
                let variant = match segments.iter().collect::<Vec<_>>().as_slice() {
                    [owner, variant]
                        if (owner.ident == "Self" || owner.ident == self.self_ident)
                            && variant.ident.to_string().starts_with(char::is_uppercase) =>
                    {
                        variant.ident.clone()
                    }
                    _ => return,
                };
                if !self.reads_params(&call.args[0]) {
                    return;
                }
                let key = LitStr::new(&variant.to_string(), variant.span());
                let payload = call.args[0].clone();
                call.args[0] = self.scoped(&payload, quote!(Self::__nn_variant_path(#key)));
            }
            _ => {}
        }
    }
}

/// Generates `Layer<B>` of the module from its `forward`.
fn layer_impl(item: &ItemImpl, forward: syn::ImplItemFn) -> Result<TokenStream> {
    let sig = &forward.sig;
    if !matches!(forward.vis, syn::Visibility::Inherited) {
        return Err(Error::new(
            forward.vis.span(),
            "`forward` implements `Layer::forward`, so it takes no visibility",
        ));
    }
    if !sig.generics.params.is_empty() || sig.generics.where_clause.is_some() {
        return Err(Error::new(
            sig.generics.span(),
            "`forward` takes no generic parameters",
        ));
    }
    let mut inputs = sig.inputs.iter();
    match inputs.next() {
        Some(FnArg::Receiver(receiver))
            if receiver.reference.is_some() && receiver.mutability.is_none() => {}
        _ => {
            return Err(Error::new(
                sig.inputs.span(),
                "`forward` must take `&self` first",
            ))
        }
    }
    let mut pats: Punctuated<Pat, Token![,]> = Punctuated::new();
    let mut types: Punctuated<Type, Token![,]> = Punctuated::new();
    for input in inputs {
        let FnArg::Typed(input) = input else {
            unreachable!("only the first input is a receiver");
        };
        let mut ty = (*input.ty).clone();
        ArgsLifetime.visit_type_mut(&mut ty);
        pats.push((*input.pat).clone());
        types.push(ty);
    }
    let (args_pat, args_ty) = if pats.len() == 1 {
        (pats[0].to_token_stream(), types[0].to_token_stream())
    } else {
        (quote!((#pats)), quote!((#types)))
    };
    let ReturnType::Type(_, output) = &sig.output else {
        return Err(Error::new(
            sig.output.span(),
            "`forward` must return its output",
        ));
    };

    let (impl_generics, _, where_clause) = item.generics.split_for_impl();
    let self_ty = &item.self_ty;
    let attrs = &item.attrs;
    let forward_attrs = &forward.attrs;
    let block = &forward.block;
    let lifetime = args_lifetime();
    Ok(quote! {
        #(#attrs)*
        impl #impl_generics ::gpt_rs::module::Layer<B> for #self_ty #where_clause {
            type Args<#lifetime> = #args_ty;
            type Output = #output;

            #(#forward_attrs)*
            fn forward(&self, #args_pat: Self::Args<'_>) -> Self::Output #block
        }
    })
}
