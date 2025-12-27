use proc_macro::TokenStream;
use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::{format_ident, quote};
use syn::{
    parse::Parse,
    parse::ParseStream,
    parse_quote,
    spanned::Spanned,
    visit_mut::{self, VisitMut},
    AngleBracketedGenericArguments, BinOp, Expr, ExprClosure, ExprMacro, FnArg, GenericArgument,
    GenericParam, Ident, ItemFn, Lifetime, Local, Pat, PatIdent, Path, PathArguments,
    Result as SynResult, ReturnType, Stmt, Token, Type,
};

#[proc_macro_attribute]
pub fn support_runtime_overload(_attr: TokenStream, item: TokenStream) -> TokenStream {
    match expand_support_runtime_overload(item) {
        Ok(tokens) => tokens,
        Err(err) => err.to_compile_error().into(),
    }
}

#[proc_macro_attribute]
pub fn ptir_pattern(attr: TokenStream, item: TokenStream) -> TokenStream {
    match expand_ptir_pattern(attr, item) {
        Ok(tokens) => tokens,
        Err(err) => err.to_compile_error().into(),
    }
}

fn expand_support_runtime_overload(item: TokenStream) -> Result<TokenStream, syn::Error> {
    let function: ItemFn = syn::parse(item)?;

    if function.sig.generics.params.is_empty() {
        return Err(syn::Error::new(
            function.sig.span(),
            "support_runtime_overload functions must be generic over the backend",
        ));
    }

    let fn_ident = function.sig.ident.clone();
    let fn_name = fn_ident.to_string();

    let params = parse_params(&function.sig.inputs)?;
    if params.is_empty() {
        return Err(syn::Error::new(
            function.sig.span(),
            "support_runtime_overload functions must accept at least a backend parameter",
        ));
    }
    let pascal_name = to_pascal_case(&fn_name);
    let impl_ident = format_ident!("{}Implementation", pascal_name);
    let entry_ident = format_ident!("{}Entry", pascal_name);
    let ctx_ident = format_ident!("{}Context", pascal_name);

    let default_impl_ident = format_ident!("__{}_default_impl", fn_name);
    let default_forward_ident = format_ident!("__{}_default_forward", fn_name);
    let default_name_const = format_ident!("__{}_DEFAULT_NAME", fn_name.to_uppercase());

    let default_name_string = build_default_name(&fn_name, &params[1..]);
    let default_name_lit = syn::LitStr::new(&default_name_string, fn_ident.span());

    let ok_type = extract_ok_type(&function.sig.output)?;

    let mut wrapper_attrs = Vec::new();
    let mut default_impl_attrs = Vec::new();
    for attr in &function.attrs {
        if attr.path().is_ident("ptir_pattern") {
            default_impl_attrs.push(attr.clone());
        } else {
            wrapper_attrs.push(attr.clone());
        }
    }
    let attrs = wrapper_attrs;
    let vis = function.vis.clone();
    let generics = function.sig.generics.clone();

    // Create default implementation function from original.
    let mut default_fn = function.clone();
    default_fn.sig.ident = default_impl_ident.clone();
    default_fn.attrs = default_impl_attrs;

    // Prepare context struct generics.
    let mut ctx_generics = generics.clone();
    ctx_generics
        .params
        .push(GenericParam::Lifetime(parse_quote!('a)));
    let ctx_generics = strip_type_defaults(&ctx_generics);
    let ctx_struct_generics = ctx_generics.clone();
    let (ctx_impl_generics, ctx_ty_generics, ctx_where_clause) = ctx_generics.split_for_impl();

    let context_fields = params
        .iter()
        .map(|param| {
            let ident = &param.ident;
            let ty = add_lifetime(&param.ty, &Lifetime::new("'a", Span::call_site()));
            quote! { pub #ident: #ty }
        })
        .collect::<Vec<_>>();

    let default_forward_args = params
        .iter()
        .map(|param| {
            let ident = &param.ident;
            quote! { ctx.#ident }
        })
        .collect::<Vec<_>>();

    let cache_fields = if params.len() > 1 {
        params
            .iter()
            .skip(1)
            .map(|param| param.ident.clone())
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };

    let ctx_type_use = context_type_with_lifetime(&ctx_ident, &generics, quote!('_));
    let ctx_type_lifetime = context_type_with_lifetime(&ctx_ident, &generics, quote!('a));

    let cache_key_body = if cache_fields.is_empty() {
        quote! { None }
    } else {
        let updates = cache_fields.iter().map(|ident| {
            quote! {
                crate::ops::functional::runtime::accumulate_cache_key(&mut builder, &ctx.#ident);
            }
        });
        quote! {
            let mut builder = crate::ops::functional::runtime::CacheKeyBuilder::new();
            #(#updates)*
            builder.finish()
        }
    };

    let generics_no_defaults = strip_type_defaults(&generics);
    let (impl_generics, ty_generics, impl_where_clause) = generics.split_for_impl();
    let (sig_impl_generics, _sig_ty_generics, sig_where_clause) =
        generics_no_defaults.split_for_impl();

    let wrapper_body = build_wrapper_body(
        &entry_ident,
        &impl_ident,
        &ctx_ident,
        &default_impl_ident,
        &default_name_const,
        &params,
    );

    let key_expr = quote! {
        crate::ops::functional::registry::FunctionalKey::new(concat!(module_path!(), "::", stringify!(#fn_ident)))
    };

    let wrapper_params = params
        .iter()
        .map(|param| param.arg.clone())
        .collect::<Vec<_>>();

    let output = quote! {
        #default_fn

        fn #default_forward_ident #impl_generics (
            ctx: #ctx_type_use,
        ) -> anyhow::Result<#ok_type> #impl_where_clause {
            #default_impl_ident(#(#default_forward_args),*)
        }

        pub struct #ctx_ident #ctx_struct_generics #ctx_where_clause {
            #(#context_fields),*
        }

        impl #ctx_impl_generics Copy for #ctx_ident #ctx_ty_generics #ctx_where_clause {}

        impl #ctx_impl_generics Clone for #ctx_ident #ctx_ty_generics #ctx_where_clause {
            fn clone(&self) -> Self {
                *self
            }
        }

        pub struct #impl_ident #impl_generics
        where
            B: crate::backend::spec::PortableBackend + 'static,
        {
            name: &'static str,
            forward: fn(#ctx_type_use) -> anyhow::Result<#ok_type>,
            supports: fn(&#ctx_type_use) -> bool,
        }

        impl #impl_generics #impl_ident #ty_generics
        where
            B: crate::backend::spec::PortableBackend + 'static,
        {
            pub fn new(
                name: &'static str,
                forward: fn(#ctx_type_use) -> anyhow::Result<#ok_type>,
            ) -> Self {
                Self {
                    name,
                    forward,
                    supports: |_| true,
                }
            }

            pub fn with_supports(
                mut self,
                supports: fn(&#ctx_type_use) -> bool,
            ) -> Self {
                self.supports = supports;
                self
            }

            pub fn portable() -> Self {
                Self::new(#default_name_const, #default_forward_ident::<B>)
            }

            pub fn name(&self) -> &'static str {
                self.name
            }

            pub fn forward(&self, ctx: #ctx_type_use) -> anyhow::Result<#ok_type> {
                (self.forward)(ctx)
            }

            pub fn supports(&self, ctx: &#ctx_type_use) -> bool {
                (self.supports)(ctx)
            }
        }

        pub struct #entry_ident #impl_generics (std::marker::PhantomData<B>)
        where
            B: crate::backend::spec::PortableBackend + 'static;

        impl #impl_generics crate::ops::functional::registry::FunctionalRegistryEntry<B>
            for #entry_ident #ty_generics
        where
            B: crate::backend::spec::PortableBackend + 'static,
        {
            type Impl = #impl_ident<B>;
            type ForwardCtx<'a> = #ctx_type_lifetime;
            type ForwardOutput = #ok_type;

            fn key() -> crate::ops::functional::registry::FunctionalKey {
                #key_expr
            }

            fn name(implementation: &Self::Impl) -> &'static str {
                implementation.name()
            }

            fn supports<'a>(implementation: &Self::Impl, ctx: &Self::ForwardCtx<'a>) -> bool {
                implementation.supports(ctx)
            }

            fn forward<'a>(
                implementation: &Self::Impl,
                ctx: Self::ForwardCtx<'a>,
            ) -> anyhow::Result<Self::ForwardOutput> {
                implementation.forward(ctx)
            }

            fn cache_key(ctx: &Self::ForwardCtx<'_>) -> Option<crate::ops::functional::runtime::FunctionalCacheKey> {
                #cache_key_body
            }
        }

        impl #impl_generics crate::ops::functional::registry::IntoImplementation<#entry_ident<B>, B>
            for #impl_ident<B>
        where
            B: crate::backend::spec::PortableBackend + 'static,
        {
            fn into_impl(self) -> std::sync::Arc<Self> {
                std::sync::Arc::new(self)
            }
        }

        impl #impl_generics crate::ops::functional::registry::IntoImplementation<#entry_ident<B>, B>
            for std::sync::Arc<#impl_ident<B>>
        where
            B: crate::backend::spec::PortableBackend + 'static,
        {
            fn into_impl(self) -> std::sync::Arc<#impl_ident<B>> {
                self
            }
        }

        const #default_name_const: &str = #default_name_lit;

        #(#attrs)*
        #vis fn #fn_ident #sig_impl_generics (#(#wrapper_params),*) -> anyhow::Result<#ok_type> #sig_where_clause {
            #wrapper_body
        }
    };

    Ok(output.into())
}

#[derive(Clone)]
enum AnchorSpec {
    Auto,
    Name(Ident),
}

#[derive(Clone)]
struct PatternAttr {
    target: syn::LitStr,
    anchor: AnchorSpec,
    bind: Option<Vec<Ident>>,
}

impl Parse for PatternAttr {
    fn parse(input: ParseStream<'_>) -> SynResult<Self> {
        let mut target: Option<syn::LitStr> = None;
        let mut anchor: Option<AnchorSpec> = None;
        let mut bind: Option<Vec<Ident>> = None;

        while !input.is_empty() {
            let key: Ident = input.parse()?;

            if key == "target" {
                input.parse::<Token![=]>()?;
                let value: syn::LitStr = input.parse()?;
                target = Some(value);
            } else if key == "anchor" {
                input.parse::<Token![=]>()?;
                let value: Ident = input.parse()?;
                if value == "auto" {
                    anchor = Some(AnchorSpec::Auto);
                } else {
                    anchor = Some(AnchorSpec::Name(value));
                }
            } else if key == "bind" {
                let content;
                syn::parenthesized!(content in input);
                let names = content
                    .parse_terminated(Ident::parse, Token![,])?
                    .into_iter()
                    .collect::<Vec<_>>();
                bind = Some(names);
            } else {
                return Err(syn::Error::new(
                    key.span(),
                    "unknown ptir_pattern argument (expected target=..., anchor=..., bind(...))",
                ));
            }

            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
            }
        }

        let target = target.ok_or_else(|| {
            syn::Error::new(Span::call_site(), "ptir_pattern requires target = \"...\"")
        })?;

        Ok(PatternAttr {
            target,
            anchor: anchor.unwrap_or(AnchorSpec::Auto),
            bind,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ViewKind {
    Add,
    Sub,
    Mul,
    Div,
    Maximum,
    Minimum,
    Exp,
    ReduceSum,
    ReduceMax,
    DotGeneral,
    ExtractPatches,
    Reshape,
    Transpose,
    BroadcastTo,
    Slice,
    Concat,
    Take,
    DynamicUpdateSlice,
    ReduceWindow,
}

impl ViewKind {
    fn view_type_tokens(self) -> TokenStream2 {
        match self {
            ViewKind::Add => quote!(::gpt_rs::backend::pattern::AddOpView),
            ViewKind::Sub => quote!(::gpt_rs::backend::pattern::SubOpView),
            ViewKind::Mul => quote!(::gpt_rs::backend::pattern::MulOpView),
            ViewKind::Div => quote!(::gpt_rs::backend::pattern::DivOpView),
            ViewKind::Maximum => quote!(::gpt_rs::backend::pattern::MaximumOpView),
            ViewKind::Minimum => quote!(::gpt_rs::backend::pattern::MinimumOpView),
            ViewKind::Exp => quote!(::gpt_rs::backend::pattern::ExpOpView),
            ViewKind::ReduceSum => quote!(::gpt_rs::backend::pattern::ReduceSumOpView),
            ViewKind::ReduceMax => quote!(::gpt_rs::backend::pattern::ReduceMaxOpView),
            ViewKind::DotGeneral => quote!(::gpt_rs::backend::pattern::DotGeneralOpView),
            ViewKind::ExtractPatches => quote!(::gpt_rs::backend::pattern::ExtractPatchesOpView),
            ViewKind::Reshape => quote!(::gpt_rs::backend::pattern::ReshapeOpView),
            ViewKind::Transpose => quote!(::gpt_rs::backend::pattern::TransposeOpView),
            ViewKind::BroadcastTo => quote!(::gpt_rs::backend::pattern::BroadcastOpView),
            ViewKind::Slice => quote!(::gpt_rs::backend::pattern::SliceOpView),
            ViewKind::Concat => quote!(::gpt_rs::backend::pattern::ConcatOpView),
            ViewKind::Take => quote!(::gpt_rs::backend::pattern::TakeOpView),
            ViewKind::DynamicUpdateSlice => {
                quote!(::gpt_rs::backend::pattern::DynamicUpdateSliceOpView)
            }
            ViewKind::ReduceWindow => quote!(::gpt_rs::backend::pattern::ReduceWindowOpView),
        }
    }

    fn view_type_name(self) -> &'static str {
        match self {
            ViewKind::Add => "AddOpView",
            ViewKind::Sub => "SubOpView",
            ViewKind::Mul => "MulOpView",
            ViewKind::Div => "DivOpView",
            ViewKind::Maximum => "MaximumOpView",
            ViewKind::Minimum => "MinimumOpView",
            ViewKind::Exp => "ExpOpView",
            ViewKind::ReduceSum => "ReduceSumOpView",
            ViewKind::ReduceMax => "ReduceMaxOpView",
            ViewKind::DotGeneral => "DotGeneralOpView",
            ViewKind::ExtractPatches => "ExtractPatchesOpView",
            ViewKind::Reshape => "ReshapeOpView",
            ViewKind::Transpose => "TransposeOpView",
            ViewKind::BroadcastTo => "BroadcastOpView",
            ViewKind::Slice => "SliceOpView",
            ViewKind::Concat => "ConcatOpView",
            ViewKind::Take => "TakeOpView",
            ViewKind::DynamicUpdateSlice => "DynamicUpdateSliceOpView",
            ViewKind::ReduceWindow => "ReduceWindowOpView",
        }
    }

    fn matcher_tokens(self) -> TokenStream2 {
        match self {
            ViewKind::Add => quote!(::gpt_rs::backend::pattern::filters::add),
            ViewKind::Sub => quote!(::gpt_rs::backend::pattern::filters::sub),
            ViewKind::Mul => quote!(::gpt_rs::backend::pattern::filters::mul),
            ViewKind::Div => quote!(::gpt_rs::backend::pattern::filters::div),
            ViewKind::Maximum => quote!(::gpt_rs::backend::pattern::filters::maximum),
            ViewKind::Minimum => quote!(::gpt_rs::backend::pattern::filters::minimum),
            ViewKind::Exp => quote!(::gpt_rs::backend::pattern::filters::exp),
            ViewKind::ReduceSum => quote!(::gpt_rs::backend::pattern::filters::reduce_sum),
            ViewKind::ReduceMax => quote!(::gpt_rs::backend::pattern::filters::reduce_max),
            ViewKind::DotGeneral => quote!(::gpt_rs::backend::pattern::filters::dot_general),
            ViewKind::ExtractPatches => {
                quote!(::gpt_rs::backend::pattern::filters::extract_patches)
            }
            ViewKind::Reshape => quote!(::gpt_rs::backend::pattern::filters::reshape),
            ViewKind::Transpose => quote!(::gpt_rs::backend::pattern::filters::transpose),
            ViewKind::BroadcastTo => quote!(::gpt_rs::backend::pattern::filters::broadcast_to),
            ViewKind::Slice => quote!(::gpt_rs::backend::pattern::filters::slice),
            ViewKind::Concat => quote!(::gpt_rs::backend::pattern::filters::concat),
            ViewKind::Take => quote!(::gpt_rs::backend::pattern::filters::take),
            ViewKind::DynamicUpdateSlice => {
                quote!(::gpt_rs::backend::pattern::filters::dynamic_update_slice)
            }
            ViewKind::ReduceWindow => quote!(::gpt_rs::backend::pattern::filters::reduce_window),
        }
    }

    fn preferred_anchor_score(self) -> u8 {
        match self {
            ViewKind::ExtractPatches => 0,
            ViewKind::DotGeneral => 1,
            ViewKind::ReduceMax | ViewKind::ReduceSum => 2,
            ViewKind::Slice => 3,
            ViewKind::Concat | ViewKind::DynamicUpdateSlice | ViewKind::ReduceWindow => 3,
            ViewKind::Transpose | ViewKind::Reshape | ViewKind::BroadcastTo => 4,
            _ => 10,
        }
    }

    fn bind_suffix(self) -> &'static str {
        match self {
            ViewKind::Add => "add",
            ViewKind::Sub => "sub",
            ViewKind::Mul => "mul",
            ViewKind::Div => "div",
            ViewKind::Maximum => "maximum",
            ViewKind::Minimum => "minimum",
            ViewKind::Exp => "exp",
            ViewKind::ReduceSum => "reduce_sum",
            ViewKind::ReduceMax => "reduce_max",
            ViewKind::DotGeneral => "dot_general",
            ViewKind::ExtractPatches => "extract_patches",
            ViewKind::Reshape => "reshape",
            ViewKind::Transpose => "transpose",
            ViewKind::BroadcastTo => "broadcast_to",
            ViewKind::Slice => "slice",
            ViewKind::Concat => "concat",
            ViewKind::Take => "take",
            ViewKind::DynamicUpdateSlice => "dynamic_update_slice",
            ViewKind::ReduceWindow => "reduce_window",
        }
    }
}

#[derive(Debug, Clone)]
struct BindInfo {
    name: String,
    kind: ViewKind,
}

fn strip_expr_wrappers(expr: &Expr) -> &Expr {
    let mut current = expr;
    loop {
        match current {
            Expr::Paren(inner) => current = &inner.expr,
            Expr::Group(inner) => current = &inner.expr,
            Expr::Try(inner) => current = &inner.expr,
            _ => return current,
        }
    }
}

fn ident_from_expr(expr: &Expr) -> Option<Ident> {
    match strip_expr_wrappers(expr) {
        Expr::Path(path) if path.qself.is_none() && path.path.segments.len() == 1 => {
            Some(path.path.segments.first()?.ident.clone())
        }
        _ => None,
    }
}

fn expr_mentions_known_tensor(expr: &Expr, known: &std::collections::HashSet<String>) -> bool {
    if let Some(ident) = ident_from_expr(expr) {
        return known.contains(&ident.to_string());
    }
    false
}

fn expr_tree_mentions_known_tensor(expr: &Expr, known: &std::collections::HashSet<String>) -> bool {
    match strip_expr_wrappers(expr) {
        Expr::Path(_) => expr_mentions_known_tensor(expr, known),
        Expr::Binary(bin) => {
            expr_tree_mentions_known_tensor(&bin.left, known)
                || expr_tree_mentions_known_tensor(&bin.right, known)
        }
        Expr::MethodCall(call) => {
            expr_tree_mentions_known_tensor(&call.receiver, known)
                || call
                    .args
                    .iter()
                    .any(|arg| expr_tree_mentions_known_tensor(arg, known))
        }
        Expr::Call(call) => {
            expr_tree_mentions_known_tensor(&call.func, known)
                || call
                    .args
                    .iter()
                    .any(|arg| expr_tree_mentions_known_tensor(arg, known))
        }
        Expr::Reference(reference) => expr_tree_mentions_known_tensor(&reference.expr, known),
        Expr::Unary(unary) => expr_tree_mentions_known_tensor(&unary.expr, known),
        Expr::Cast(cast) => expr_tree_mentions_known_tensor(&cast.expr, known),
        Expr::Field(field) => expr_tree_mentions_known_tensor(&field.base, known),
        Expr::Index(index) => {
            expr_tree_mentions_known_tensor(&index.expr, known)
                || expr_tree_mentions_known_tensor(&index.index, known)
        }
        Expr::If(expr_if) => {
            expr_tree_mentions_known_tensor(&expr_if.cond, known)
                || expr_if.then_branch.stmts.iter().any(|stmt| match stmt {
                    Stmt::Expr(expr, _) => expr_tree_mentions_known_tensor(expr, known),
                    Stmt::Local(local) => local
                        .init
                        .as_ref()
                        .is_some_and(|init| expr_tree_mentions_known_tensor(&init.expr, known)),
                    Stmt::Item(_) | Stmt::Macro(_) => false,
                })
                || expr_if
                    .else_branch
                    .as_ref()
                    .is_some_and(|(_, expr)| expr_tree_mentions_known_tensor(expr, known))
        }
        _ => false,
    }
}

fn infer_view_kind(expr: &Expr, known: &std::collections::HashSet<String>) -> Option<ViewKind> {
    match strip_expr_wrappers(expr) {
        Expr::Call(call) => {
            let Expr::Path(path) = &*call.func else {
                return None;
            };
            let name = path.path.segments.last()?.ident.to_string();
            match name.as_str() {
                "concat" | "try_concat" => Some(ViewKind::Concat),
                _ => None,
            }
        }
        Expr::Binary(bin) => {
            if !expr_tree_mentions_known_tensor(&bin.left, known)
                && !expr_tree_mentions_known_tensor(&bin.right, known)
            {
                return None;
            }
            match &bin.op {
                BinOp::Add(_) => Some(ViewKind::Add),
                BinOp::Sub(_) => Some(ViewKind::Sub),
                BinOp::Mul(_) => Some(ViewKind::Mul),
                BinOp::Div(_) => Some(ViewKind::Div),
                _ => None,
            }
        }
        Expr::MethodCall(call) => {
            if !expr_tree_mentions_known_tensor(&call.receiver, known) {
                return None;
            }
            let name = call.method.to_string();
            match name.as_str() {
                "extract_patches" => Some(ViewKind::ExtractPatches),
                "dot_general" => Some(ViewKind::DotGeneral),
                "reshape" | "try_reshape" => Some(ViewKind::Reshape),
                "transpose" | "try_transpose" => Some(ViewKind::Transpose),
                "broadcast_to" | "try_broadcast_to" => Some(ViewKind::BroadcastTo),
                "slice" | "try_slice" => Some(ViewKind::Slice),
                "take" | "try_take" => Some(ViewKind::Take),
                "dynamic_update_slice" | "try_dynamic_update_slice" => {
                    Some(ViewKind::DynamicUpdateSlice)
                }
                "reduce_window" | "try_reduce_window" => Some(ViewKind::ReduceWindow),
                "maximum" | "try_maximum" => Some(ViewKind::Maximum),
                "minimum" | "try_minimum" => Some(ViewKind::Minimum),
                "reduce_sum" | "try_reduce_sum" => Some(ViewKind::ReduceSum),
                "reduce_max" | "try_reduce_max" => Some(ViewKind::ReduceMax),
                "exp" => Some(ViewKind::Exp),
                _ => None,
            }
        }
        _ => None,
    }
}

fn bind_name_for(
    original: &Ident,
    kind: ViewKind,
    base_counts: &mut std::collections::HashMap<String, u32>,
    counts: &mut std::collections::HashMap<String, u32>,
) -> String {
    let base = original.to_string();
    let base_entry = base_counts.entry(base.clone()).or_insert(0);
    let is_shadowed = *base_entry > 0;
    *base_entry = base_entry.saturating_add(1);

    let candidate = if is_shadowed {
        format!("{base}_{}", kind.bind_suffix())
    } else {
        base
    };

    let entry = counts.entry(candidate.clone()).or_insert(0);
    let suffix = *entry;
    *entry = entry.saturating_add(1);
    if suffix == 0 {
        candidate
    } else {
        format!("{candidate}_{suffix}")
    }
}

fn inject_binds_into_capture(
    closure: &mut ExprClosure,
    bindings: &[Binding],
) -> SynResult<Vec<BindInfo>> {
    let body = match &mut *closure.body {
        Expr::Block(block) => block,
        other => {
            return Err(syn::Error::new(
                other.span(),
                "ptir_pattern expects capture_ptir closure body to be a block",
            ))
        }
    };

    let mut known_tensors: std::collections::HashSet<String> = std::collections::HashSet::new();
    for binding in bindings {
        known_tensors.insert(binding.name.to_string());
    }

    let mut base_counts: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
    let mut counts: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
    let mut binds = Vec::new();

    for stmt in &mut body.block.stmts {
        let Stmt::Local(Local {
            pat,
            init: Some(init),
            ..
        }) = stmt
        else {
            continue;
        };
        let Pat::Ident(PatIdent { ident, .. }) = pat else {
            continue;
        };

        let Some(kind) = infer_view_kind(&init.expr, &known_tensors) else {
            continue;
        };

        let bind_name = bind_name_for(ident, kind, &mut base_counts, &mut counts);
        let bind_lit = syn::LitStr::new(&bind_name, ident.span());
        let orig_expr = (*init.expr).clone();
        init.expr = Box::new(parse_quote! { (#orig_expr).ptir_bind(#bind_lit) });

        known_tensors.insert(ident.to_string());
        binds.push(BindInfo {
            name: bind_name,
            kind,
        });
    }

    Ok(binds)
}

#[derive(Default)]
struct CaptureSiteCollector {
    next_site: u32,
    sites: Vec<Vec<BindInfo>>,
    error: Option<syn::Error>,
}

impl VisitMut for CaptureSiteCollector {
    fn visit_expr_mut(&mut self, expr: &mut Expr) {
        if self.error.is_some() {
            return;
        }

        if let Expr::Macro(expr_macro) = expr {
            if expr_macro
                .mac
                .path
                .segments
                .last()
                .map(|segment| segment.ident == "capture_ptir")
                .unwrap_or(false)
            {
                let site_id = self.next_site;
                self.next_site = self.next_site.saturating_add(1);

                match rewrite_capture_ptir(expr_macro, site_id) {
                    Ok((replacement, binds)) => {
                        self.sites.push(binds);
                        *expr = replacement;
                        return;
                    }
                    Err(err) => {
                        self.error = Some(err);
                        return;
                    }
                }
            }
        }

        visit_mut::visit_expr_mut(self, expr);
    }
}

fn rewrite_capture_ptir(expr: &ExprMacro, site_id: u32) -> SynResult<(Expr, Vec<BindInfo>)> {
    let mut capture: CaptureInput = syn::parse2(expr.mac.tokens.clone()).map_err(|err| {
        syn::Error::new(
            err.span(),
            "ptir_pattern failed to parse capture_ptir! invocation",
        )
    })?;

    let binds = inject_binds_into_capture(&mut capture.closure, &capture.bindings)?;

    let mac_path = expr.mac.path.clone();
    let graph_part = capture.graph.as_ref().map(|graph| quote!(graph = #graph;));
    let binding_parts = capture.bindings.iter().map(|binding| {
        let name = &binding.name;
        if let Some(expr) = &binding.expr {
            quote!(#name = #expr)
        } else {
            quote!(#name)
        }
    });
    let closure = capture.closure;

    let call_tokens = quote! {
        #graph_part
        { #(#binding_parts),* },
        #closure
    };

    let site_lit = syn::LitInt::new(&site_id.to_string(), Span::call_site());
    let wrapped_expr: Expr = parse_quote! {{
        let _ptir_site_guard = ::gpt_rs::backend::pattern::PatternCaptureSiteGuard::push(#site_lit);
        let _ptir_capture = ::gpt_rs::backend::pattern::CaptureCallGuard::begin();
        let __ptir_res = #mac_path!(#call_tokens);
        __ptir_res.map(|(__ptir_graph, __ptir_value)| {
            _ptir_capture.finish(&__ptir_value);
            (__ptir_graph, __ptir_value)
        })
    }};

    Ok((wrapped_expr, binds))
}

fn expand_ptir_pattern(attr: TokenStream, item: TokenStream) -> SynResult<TokenStream> {
    let attr: PatternAttr = syn::parse(attr)?;
    let mut function: ItemFn = syn::parse(item)?;

    let fn_ident = function.sig.ident.clone();
    let fn_name = fn_ident.to_string();
    let base_name = fn_name
        .strip_prefix("__")
        .and_then(|rest| rest.strip_suffix("_default_impl"))
        .unwrap_or(&fn_name);

    let pattern_name = to_pascal_case(base_name);
    let view_name = if pattern_name.ends_with("Pattern") {
        pattern_name.clone()
    } else {
        format!("{pattern_name}Pattern")
    };
    let view_ident = format_ident!("{view_name}", span = fn_ident.span());

    let record_ident = format_ident!(
        "__{}_ptir_pattern_record",
        base_name,
        span = fn_ident.span()
    );
    let site_captured_ident = format_ident!(
        "__{}_ptir_pattern_site_captured",
        base_name,
        span = fn_ident.span()
    );

    let mut collector = CaptureSiteCollector::default();
    collector.visit_block_mut(&mut function.block);
    if let Some(err) = collector.error.take() {
        return Err(err);
    }

    let site_count = collector.sites.len();
    if site_count == 0 {
        return Err(syn::Error::new(
            function.sig.span(),
            "ptir_pattern requires at least one capture_ptir! invocation",
        ));
    }

    // Determine anchor bind name.
    let anchor_name = match attr.anchor {
        AnchorSpec::Name(name) => name.to_string(),
        AnchorSpec::Auto => {
            let first = collector
                .sites
                .first()
                .ok_or_else(|| syn::Error::new(function.sig.span(), "missing capture sites"))?;
            let best = first
                .iter()
                .min_by_key(|bind| (bind.kind.preferred_anchor_score(), bind.name.clone()))
                .ok_or_else(|| {
                    syn::Error::new(
                        function.sig.span(),
                        "ptir_pattern anchor=auto requires at least one PTIR-producing `let` inside capture_ptir closure",
                    )
                })?;
            best.name.clone()
        }
    };

    for (idx, site) in collector.sites.iter().enumerate() {
        if !site.iter().any(|b| b.name == anchor_name) {
            return Err(syn::Error::new(
                function.sig.span(),
                format!("ptir_pattern anchor `{anchor_name}` is missing from capture site {idx}"),
            ));
        }
    }

    // Build union of binds across capture sites.
    let mut union: std::collections::BTreeMap<String, ViewKind> = std::collections::BTreeMap::new();
    let mut present_in_all: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    for site in &collector.sites {
        let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for bind in site {
            if let Some(existing) = union.get(&bind.name).copied() {
                if existing != bind.kind {
                    return Err(syn::Error::new(
                        function.sig.span(),
                        format!(
                            "ptir_pattern bind `{}` has conflicting kinds across capture sites",
                            bind.name
                        ),
                    ));
                }
            } else {
                union.insert(bind.name.clone(), bind.kind);
            }
            if seen.insert(&bind.name) {
                *present_in_all.entry(bind.name.clone()).or_insert(0) += 1;
            }
        }
    }

    // Apply explicit bind(...) restriction if provided.
    let bind_filter: Option<std::collections::HashSet<String>> = attr
        .bind
        .map(|list| list.into_iter().map(|id| id.to_string()).collect());
    if let Some(filter) = &bind_filter {
        for name in filter {
            if !union.contains_key(name) {
                return Err(syn::Error::new(
                    function.sig.span(),
                    format!("ptir_pattern bind `{name}` does not exist in any capture site"),
                ));
            }
        }
    }

    let binds = union
        .iter()
        .filter(|(name, _)| bind_filter.as_ref().is_none_or(|f| f.contains(*name)))
        .map(|(name, kind)| (name.clone(), *kind))
        .collect::<Vec<_>>();

    let required_names = binds
        .iter()
        .filter(|(name, _)| present_in_all.get(name).copied() == Some(site_count))
        .map(|(name, _)| name.clone())
        .collect::<std::collections::HashSet<_>>();

    let anchor_kind = union
        .get(&anchor_name)
        .copied()
        .ok_or_else(|| syn::Error::new(function.sig.span(), "anchor kind missing"))?;

    // Prepend capture guard to the function body.
    let record_path: Path = parse_quote!(#record_ident);
    let site_captured_path: Path = parse_quote!(#site_captured_ident);
    function
        .block
        .stmts
        .insert(0, parse_quote! { let _ptir_pattern_guard = ::gpt_rs::backend::pattern::PatternCaptureGuard::push(#record_path, #site_captured_path); });

    // Generate per-site storage.
    let variant_ident = format_ident!("__{}PatternVariant", pattern_name, span = fn_ident.span());
    let variant_fields = binds.iter().map(|(name, _)| {
        let ident = format_ident!("{name}");
        if required_names.contains(name) {
            quote!(pub #ident: ::gpt_rs::backend::pattern::TemplateNodeId)
        } else {
            quote!(pub #ident: ::core::option::Option<::gpt_rs::backend::pattern::TemplateNodeId>)
        }
    });

    let pattern_upper = pattern_name.to_uppercase();
    let site_locks = (0..site_count)
        .map(|idx| format_ident!("__PTIR_PATTERN_{}_SITE_{}", pattern_upper, idx))
        .collect::<Vec<_>>();
    let site_static_defs = site_locks.iter().map(|ident| {
        quote!(static #ident: ::std::sync::OnceLock<#variant_ident> = ::std::sync::OnceLock::new();)
    });

    let site_match_arms = site_locks
        .iter()
        .enumerate()
        .map(|(idx, lock_ident)| {
            let idx_lit = syn::LitInt::new(&idx.to_string(), Span::call_site());
            quote!(#idx_lit => #lock_ident.get().is_some(),)
        })
        .collect::<Vec<_>>();

    let anchor_lit = syn::LitStr::new(&anchor_name, fn_ident.span());

    let set_arms = site_locks.iter().enumerate().map(|(idx, lock_ident)| {
        let idx_lit = syn::LitInt::new(&idx.to_string(), Span::call_site());
        quote!(#idx_lit => { let _ = #lock_ident.set(variant); },)
    });

    let field_value_builders = binds.iter().map(|(name, _)| {
        let bind_lit = syn::LitStr::new(name, fn_ident.span());
        let field_ident = format_ident!("{name}");
        if required_names.contains(name) {
            quote! {
                let #field_ident = match binds.iter().rev().find(|b| b.name == #bind_lit).map(|b| b.value).and_then(|v| value_to_node.get(&v).copied()) {
                    Some(id) => id,
                    None => return,
                };
            }
        } else {
            quote! {
                let #field_ident = binds
                    .iter()
                    .rev()
                    .find(|b| b.name == #bind_lit)
                    .map(|b| b.value)
                    .and_then(|v| value_to_node.get(&v).copied());
            }
        }
    });

    let variant_field_inits = binds.iter().map(|(name, _)| {
        let ident = format_ident!("{name}");
        quote!(#ident: #ident)
    });

    let view_struct_fields = binds.iter().map(|(name, kind)| {
        let ident = format_ident!("{name}");
        let view_ty = kind.view_type_tokens();
        if required_names.contains(name) {
            quote!(pub #ident: #view_ty)
        } else {
            quote!(pub #ident: ::core::option::Option<#view_ty>)
        }
    });

    let view_field_extractors = binds
        .iter()
        .map(|(name, kind)| {
        let field_ident = format_ident!("{name}");
        let view_ty = kind.view_type_tokens();
        if required_names.contains(name) {
            quote! {
                let #field_ident = {
                    let inst = matched.inst(variant.#field_ident)?;
                    <#view_ty as ::gpt_rs::backend::pattern::OperationView>::extract(inst, rewriter)?
                };
            }
        } else {
            quote! {
                let #field_ident = if let Some(node) = variant.#field_ident {
                    let inst = matched.inst(node)?;
                    Some(<#view_ty as ::gpt_rs::backend::pattern::OperationView>::extract(inst, rewriter)?)
                } else {
                    None
                };
            }
        }
    })
        .collect::<Vec<_>>();

    let view_field_inits = binds
        .iter()
        .map(|(name, _)| {
            let ident = format_ident!("{name}");
            quote!(#ident: #ident)
        })
        .collect::<Vec<_>>();

    let matcher_tokens = anchor_kind.matcher_tokens();
    let target_lit = attr.target;
    let site_extract_blocks = site_locks
        .iter()
        .map(|lock_ident| {
            quote! {
                if let Some(variant) = #lock_ident.get() {
                    if let Some(matched) = variant.template.match_from_anchor(root, rewriter) {
                        #(#view_field_extractors)*
                        return Some(Self {
                            match_: matched,
                            #(#view_field_inits),*
                        });
                    }
                }
            }
        })
        .collect::<Vec<_>>();

    let record_fn = quote! {
        fn #record_ident(
            site: u32,
            nodes: &[::gpt_rs::backend::pattern::CapturedNode],
            binds: &[::gpt_rs::backend::pattern::BindRecord],
            output: &dyn ::core::any::Any,
        ) {
            let already = match site {
                #(#site_match_arms)*
                _ => true,
            };
            if already {
                return;
            }

            let output = if let Some(output) = output
                .downcast_ref::<::gpt_rs::backend::spec::ValueId>()
                .copied()
            {
                output
            } else if let Some(output) = output
                .downcast_ref::<(
                    ::gpt_rs::backend::spec::ValueId,
                    ::gpt_rs::backend::spec::ValueId,
                )>()
                .map(|t| t.0)
            {
                output
            } else if let Some(output) = output
                .downcast_ref::<(
                    ::gpt_rs::backend::spec::ValueId,
                    ::gpt_rs::backend::spec::ValueId,
                    ::gpt_rs::backend::spec::ValueId,
                )>()
                .map(|t| t.0)
            {
                output
            } else if let Some(output) = output
                .downcast_ref::<(
                    ::gpt_rs::backend::spec::ValueId,
                    ::gpt_rs::backend::spec::ValueId,
                    ::gpt_rs::backend::spec::ValueId,
                    ::gpt_rs::backend::spec::ValueId,
                )>()
                .map(|t| t.0)
            {
                output
            } else if let Some(output) = output
                .downcast_ref::<(
                    ::gpt_rs::backend::spec::ValueId,
                    ::gpt_rs::backend::spec::ValueId,
                    ::gpt_rs::backend::spec::ValueId,
                    ::gpt_rs::backend::spec::ValueId,
                    ::gpt_rs::backend::spec::ValueId,
                )>()
                .map(|t| t.0)
            {
                output
            } else if let Some(output) = output
                .downcast_ref::<(
                    ::gpt_rs::backend::spec::ValueId,
                    ::gpt_rs::backend::spec::ValueId,
                    ::gpt_rs::backend::spec::ValueId,
                    ::gpt_rs::backend::spec::ValueId,
                    ::gpt_rs::backend::spec::ValueId,
                    ::gpt_rs::backend::spec::ValueId,
                )>()
                .map(|t| t.0)
            {
                output
            } else if let Some(output) = output
                .downcast_ref::<(
                    ::gpt_rs::backend::spec::ValueId,
                    ::gpt_rs::backend::spec::ValueId,
                    ::gpt_rs::backend::spec::ValueId,
                    ::gpt_rs::backend::spec::ValueId,
                    ::gpt_rs::backend::spec::ValueId,
                    ::gpt_rs::backend::spec::ValueId,
                    ::gpt_rs::backend::spec::ValueId,
                )>()
                .map(|t| t.0)
            {
                output
            } else if let Some(output) = output
                .downcast_ref::<(
                    ::gpt_rs::backend::spec::ValueId,
                    ::gpt_rs::backend::spec::ValueId,
                    ::gpt_rs::backend::spec::ValueId,
                    ::gpt_rs::backend::spec::ValueId,
                    ::gpt_rs::backend::spec::ValueId,
                    ::gpt_rs::backend::spec::ValueId,
                    ::gpt_rs::backend::spec::ValueId,
                    ::gpt_rs::backend::spec::ValueId,
                )>()
                .map(|t| t.0)
            {
                output
            } else {
                return;
            };
            let Some(anchor) = binds.iter().rev().find(|b| b.name == #anchor_lit).map(|b| b.value) else {
                return;
            };

            let Some(built) = ::gpt_rs::backend::pattern::build_template(nodes, output, anchor) else {
                return;
            };
            let ::gpt_rs::backend::pattern::BuiltTemplate { template, value_to_node } = built;

            #(#field_value_builders)*

            let variant = #variant_ident {
                template,
                #(#variant_field_inits),*
            };

            match site {
                #(#set_arms)*
                _ => {}
            }
        }
    };

    let site_captured_fn = quote! {
        fn #site_captured_ident(site: u32) -> bool {
            match site {
                #(#site_match_arms)*
                _ => true,
            }
        }
    };

    let def_ident = format_ident!(
        "__PTIR_PATTERN_DEF_{}",
        base_name.to_uppercase(),
        span = fn_ident.span()
    );

    let view_impl = quote! {
        #[derive(Clone)]
        pub struct #view_ident {
            match_: ::gpt_rs::backend::pattern::TemplateMatch,
            #(#view_struct_fields),*
        }

        impl #view_ident {
            pub const TARGET: &'static str = #target_lit;

            pub fn output(&self) -> ::gpt_rs::backend::spec::ValueId {
                self.match_.output
            }

            pub fn anchor(&self) -> ::gpt_rs::backend::index::InstId {
                self.match_.anchor
            }

            pub fn closure_report(
                &self,
                rewriter: &::gpt_rs::backend::rewriter::ProgramRewriter,
            ) -> ::gpt_rs::backend::pattern::ClosureReport {
                self.match_.closure_report(rewriter)
            }

            pub fn input(
                &self,
                index: u32,
            ) -> ::core::option::Option<::gpt_rs::backend::spec::ValueId> {
                self.match_.input(index)
            }

            pub fn input_count(&self) -> usize {
                self.match_.input_count()
            }
        }

        impl ::gpt_rs::backend::pattern::OperationView for #view_ident {
            const MATCHER: ::gpt_rs::backend::pattern::OperationMatcher = #matcher_tokens;

            fn extract(
                root: ::gpt_rs::backend::index::InstId,
                rewriter: &::gpt_rs::backend::rewriter::ProgramRewriter,
            ) -> ::core::option::Option<Self> {
                #(#site_extract_blocks)*
                None
            }
        }
    };

    let variant_def = quote! {
        #[derive(Clone)]
        struct #variant_ident {
            template: ::gpt_rs::backend::pattern::PatternTemplate,
            #(#variant_fields),*
        }

        #(#site_static_defs)*

        #site_captured_fn

        #record_fn
    };

    let pattern_field_entries = binds.iter().map(|(name, kind)| {
        let name_lit = syn::LitStr::new(name, Span::call_site());
        let view_lit = syn::LitStr::new(kind.view_type_name(), Span::call_site());
        let optional = !required_names.contains(name);
        quote! {
            ::gpt_rs::backend::pattern::PatternField {
                name: #name_lit,
                view: #view_lit,
                optional: #optional,
            }
        }
    });

    let registry_def = quote! {
        #[::gpt_rs::linkme::distributed_slice(::gpt_rs::backend::pattern::PATTERN_DEFS)]
        static #def_ident: ::gpt_rs::backend::pattern::PatternDef = ::gpt_rs::backend::pattern::PatternDef {
            target: <#view_ident>::TARGET,
            module_path: module_path!(),
            view_name: stringify!(#view_ident),
            fields: &[#(#pattern_field_entries),*],
        };
    };

    let output = quote! {
        #function
        #variant_def
        #view_impl
        #registry_def
    };

    Ok(output.into())
}

#[proc_macro]
pub fn capture_ptir(input: TokenStream) -> TokenStream {
    match expand_capture_ptir(input) {
        Ok(tokens) => tokens,
        Err(err) => err.to_compile_error().into(),
    }
}

struct CaptureInput {
    graph: Option<Expr>,
    bindings: Vec<Binding>,
    closure: ExprClosure,
}

struct Binding {
    name: Ident,
    expr: Option<Expr>,
}

impl Parse for Binding {
    fn parse(input: ParseStream<'_>) -> SynResult<Self> {
        let name: Ident = input.parse()?;
        if input.peek(Token![=]) {
            input.parse::<Token![=]>()?;
            let expr: Expr = input.parse()?;
            Ok(Binding {
                name,
                expr: Some(expr),
            })
        } else {
            Ok(Binding { name, expr: None })
        }
    }
}

impl Parse for CaptureInput {
    fn parse(input: ParseStream<'_>) -> SynResult<Self> {
        let mut graph = None;
        if input.peek(Ident) {
            let lookahead = input.fork();
            let ident: Ident = lookahead.parse()?;
            if ident == "graph" {
                input.parse::<Ident>()?; // consume `graph`
                input.parse::<Token![=]>()?;
                let expr: Expr = input.parse()?;
                input.parse::<Token![;]>()?;
                graph = Some(expr);
            }
        }

        let content;
        syn::braced!(content in input);
        let bindings = if content.is_empty() {
            Vec::new()
        } else {
            content
                .parse_terminated(Binding::parse, Token![,])?
                .into_iter()
                .collect()
        };

        if bindings.is_empty() {
            return Err(syn::Error::new(
                Span::call_site(),
                "capture_ptir requires at least one binding",
            ));
        }

        if input.peek(Token![,]) {
            input.parse::<Token![,]>()?;
        }

        let closure: ExprClosure = input.parse()?;

        Ok(CaptureInput {
            graph,
            bindings,
            closure,
        })
    }
}

fn expand_capture_ptir(input: TokenStream) -> SynResult<TokenStream> {
    let capture = syn::parse::<CaptureInput>(input)?;

    if capture.closure.inputs.len() != 1 {
        return Err(syn::Error::new(
            capture.closure.span(),
            "capture_ptir closures must take exactly one argument",
        ));
    }

    let session_pat = capture.closure.inputs.first().unwrap().clone();
    let session_ident = match session_pat.clone() {
        Pat::Ident(ref ident) => ident.ident.clone(),
        other => {
            return Err(syn::Error::new(
                other.span(),
                "capture_ptir closure parameter must be an identifier",
            ))
        }
    };

    let bindings = capture.bindings;
    let graph_override = capture.graph;

    let mut ref_idents = Vec::new();
    let mut ref_inits = Vec::new();
    let mut pre_imports = Vec::new();
    let mut session_imports = Vec::new();

    for (index, binding) in bindings.iter().enumerate() {
        let ref_ident = format_ident!("__capture_operand_{}", index);
        let expr = binding.expr.clone().map_or_else(
            || Expr::Path(expr_path_from_ident(&binding.name)),
            |expr| expr,
        );
        ref_inits.push(quote! {
            let #ref_ident = #expr;
        });
        let id_ident = format_ident!("__capture_id_{}", index);
        let spec_ident = format_ident!("__capture_spec_{}", index);
        pre_imports.push(quote! {
            let #id_ident = ctx.import(#ref_ident)?;
            let #spec_ident = ::gpt_rs::ops::functional::tensor_spec_from_device(#ref_ident);
        });
        let tensor_ident = &binding.name;
        session_imports.push(quote! {
            let #tensor_ident = #session_ident.import_spec(
                stringify!(#tensor_ident),
                #id_ident,
                #spec_ident,
            );
        });
        ref_idents.push(ref_ident.clone());
    }

    let first_ref = ref_idents
        .first()
        .ok_or_else(|| syn::Error::new(Span::call_site(), "capture_ptir requires operands"))?;

    let body = capture.closure.body;
    let value_expr = quote! {{ #body }};

    let graph_block = if let Some(expr) = graph_override {
        quote! { #expr }
    } else {
        quote! {
            let __capture_refs = vec![#(#ref_idents),*];
            ::gpt_rs::ops::functional::resolve_graph_from_tensors(&__capture_refs)
                .or_else(|| ::gpt_rs::ops::graph::context::current_arena())
                .unwrap_or_else(|| ::gpt_rs::ops::graph::GraphArena::new(#first_ref.backend()))
        }
    };

    let expanded = quote! {{
        #(#ref_inits)*
        let __capture_graph = { #graph_block };
        let __capture_value = __capture_graph.capture(|ctx| {
            #(#pre_imports)*
            let #session_pat = ::gpt_rs::ops::ptir::PtirSession::new(ctx);
            #(#session_imports)*
            #value_expr
        })?;
        ::core::result::Result::Ok::<_, ::anyhow::Error>((__capture_graph, __capture_value))
    }};

    Ok(expanded.into())
}

fn expr_path_from_ident(ident: &Ident) -> syn::ExprPath {
    syn::ExprPath {
        attrs: Vec::new(),
        qself: None,
        path: Path::from(ident.clone()),
    }
}

#[derive(Clone)]
struct ParamInfo {
    ident: Ident,
    ty: Type,
    arg: FnArg,
}

fn parse_params(
    inputs: &syn::punctuated::Punctuated<FnArg, syn::token::Comma>,
) -> Result<Vec<ParamInfo>, syn::Error> {
    inputs
        .iter()
        .map(|arg| match arg {
            FnArg::Typed(pat_type) => match &*pat_type.pat {
                Pat::Ident(PatIdent { ident, .. }) => Ok(ParamInfo {
                    ident: ident.clone(),
                    ty: (*pat_type.ty).clone(),
                    arg: FnArg::Typed(pat_type.clone()),
                }),
                _ => Err(syn::Error::new(
                    pat_type.pat.span(),
                    "unsupported parameter pattern",
                )),
            },
            FnArg::Receiver(_) => Err(syn::Error::new(arg.span(), "methods are not supported")),
        })
        .collect()
}

fn to_pascal_case(name: &str) -> String {
    name.split('_')
        .filter(|segment| !segment.is_empty())
        .map(|segment| {
            let mut chars = segment.chars();
            match chars.next() {
                Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
                None => String::new(),
            }
        })
        .collect()
}

fn build_default_name(fn_name: &str, params: &[ParamInfo]) -> String {
    if params.is_empty() {
        return fn_name.to_string();
    }
    let mut parts = vec![fn_name.to_string()];
    for param in params {
        parts.push(param.ident.to_string());
    }
    parts.join("_")
}

fn extract_ok_type(output: &ReturnType) -> Result<Type, syn::Error> {
    let ty = match output {
        ReturnType::Type(_, ty) => ty,
        ReturnType::Default => {
            return Err(syn::Error::new(
                output.span(),
                "support_runtime_overload functions must return Result",
            ))
        }
    };

    let type_path = match &**ty {
        Type::Path(type_path) => type_path,
        _ => {
            return Err(syn::Error::new(
                ty.span(),
                "support_runtime_overload functions must return Result",
            ))
        }
    };

    let last_segment = type_path.path.segments.last().ok_or_else(|| {
        syn::Error::new(
            ty.span(),
            "support_runtime_overload functions must return Result",
        )
    })?;

    if last_segment.ident != "Result" {
        return Err(syn::Error::new(
            ty.span(),
            "support_runtime_overload functions must return Result",
        ));
    }

    let args = match &last_segment.arguments {
        PathArguments::AngleBracketed(AngleBracketedGenericArguments { args, .. }) => args,
        _ => {
            return Err(syn::Error::new(
                ty.span(),
                "Result must have generic arguments",
            ))
        }
    };

    let ok_type = match args.first() {
        Some(GenericArgument::Type(inner)) => inner.clone(),
        _ => {
            return Err(syn::Error::new(
                ty.span(),
                "Result must have a type parameter",
            ))
        }
    };

    Ok(ok_type)
}

fn add_lifetime(ty: &Type, lifetime: &Lifetime) -> Type {
    match ty {
        Type::Reference(reference) => {
            let mut new_ref = reference.clone();
            if new_ref.lifetime.is_none() {
                new_ref.lifetime = Some(lifetime.clone());
            }
            Type::Reference(new_ref)
        }
        Type::Path(path) => {
            let mut new_path = path.clone();
            for segment in &mut new_path.path.segments {
                if let PathArguments::AngleBracketed(args) = &mut segment.arguments {
                    for arg in &mut args.args {
                        if let GenericArgument::Type(inner_ty) = arg {
                            *inner_ty = add_lifetime(inner_ty, lifetime);
                        }
                    }
                }
            }
            Type::Path(new_path)
        }
        _ => ty.clone(),
    }
}

fn strip_type_defaults(generics: &syn::Generics) -> syn::Generics {
    let mut cloned = generics.clone();
    for param in cloned.params.iter_mut() {
        if let GenericParam::Type(ty_param) = param {
            ty_param.default = None;
        }
    }
    cloned
}

fn build_wrapper_body(
    entry_ident: &Ident,
    impl_ident: &Ident,
    ctx_ident: &Ident,
    default_impl_ident: &Ident,
    default_name_const: &Ident,
    params: &[ParamInfo],
) -> TokenStream2 {
    let context_inits = params.iter().map(|param| {
        let ident = &param.ident;
        quote! { #ident: #ident }
    });
    let param_idents = params.iter().map(|param| param.ident.clone());

    quote! {
        if let Some(registry) = crate::ops::functional::runtime::current_registry::<B>() {
            registry.register_default::<#entry_ident<B>, _, _>(
                #default_name_const,
                || #impl_ident::<B>::portable(),
            );
            let ctx = #ctx_ident {
                #(#context_inits),*
            };
            let (output, _) = registry.call_forward::<#entry_ident<B>>(ctx)?;
            return Ok(output);
        }

        #default_impl_ident(#(#param_idents),*)
    }
}

fn context_type_with_lifetime(
    ctx_ident: &Ident,
    generics: &syn::Generics,
    lifetime: TokenStream2,
) -> TokenStream2 {
    let mut args = Vec::new();
    for param in generics.params.iter() {
        match param {
            GenericParam::Type(ty_param) => {
                let ident = &ty_param.ident;
                args.push(quote! { #ident });
            }
            GenericParam::Const(const_param) => {
                let ident = &const_param.ident;
                args.push(quote! { #ident });
            }
            GenericParam::Lifetime(_) => {}
        }
    }

    if args.is_empty() {
        quote! { #ctx_ident<#lifetime> }
    } else {
        quote! { #ctx_ident<#lifetime, #(#args),*> }
    }
}
