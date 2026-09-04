//! Generates the PTIR pattern view of a `#[functional]` from the ops that its `capture!` bodies
//! bind.

use std::collections::{BTreeMap, HashMap, HashSet};

use proc_macro2::{Span, TokenStream};
use quote::{format_ident, quote};
use syn::{parse_quote, BinOp, Error, Expr, Ident, Local, Pat, PatIdent, Result, Stmt};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ViewKind {
    Any,
    Add,
    Sub,
    Mul,
    Div,
    Maximum,
    Minimum,
    Exp,
    Unary,
    Cast,
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
    fn view_type_name(self) -> &'static str {
        match self {
            ViewKind::Any => "AnyOpView",
            ViewKind::Add => "AddOpView",
            ViewKind::Sub => "SubOpView",
            ViewKind::Mul => "MulOpView",
            ViewKind::Div => "DivOpView",
            ViewKind::Maximum => "MaximumOpView",
            ViewKind::Minimum => "MinimumOpView",
            ViewKind::Exp => "ExpOpView",
            ViewKind::Unary => "ElementwiseUnaryOpView",
            ViewKind::Cast => "CastOpView",
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

    fn view_type_tokens(self) -> TokenStream {
        let ident = Ident::new(self.view_type_name(), Span::call_site());
        quote!(::gpt_rs::backend::pattern::#ident)
    }

    fn matcher_tokens(self) -> TokenStream {
        let ident = Ident::new(
            match self {
                ViewKind::Any => "any",
                other => other.bind_suffix(),
            },
            Span::call_site(),
        );
        quote!(::gpt_rs::backend::pattern::filters::#ident)
    }

    /// Anchor preference: the rarest, most specific ops first.
    fn anchor_score(self) -> u8 {
        match self {
            ViewKind::Any => 20,
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
            ViewKind::Any => "op",
            ViewKind::Add => "add",
            ViewKind::Sub => "sub",
            ViewKind::Mul => "mul",
            ViewKind::Div => "div",
            ViewKind::Maximum => "maximum",
            ViewKind::Minimum => "minimum",
            ViewKind::Exp => "exp",
            ViewKind::Unary => "elementwise_unary",
            ViewKind::Cast => "cast",
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

/// A PTIR value that a capture body binds by name: a `let` or the body's result.
#[derive(Debug, Clone)]
pub(crate) struct BindInfo {
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

fn is_known_tensor(expr: &Expr, known: &HashSet<String>) -> bool {
    match strip_expr_wrappers(expr) {
        Expr::Path(path) if path.qself.is_none() && path.path.segments.len() == 1 => {
            known.contains(&path.path.segments[0].ident.to_string())
        }
        _ => false,
    }
}

fn any_mentions<'a>(exprs: impl IntoIterator<Item = &'a Expr>, known: &HashSet<String>) -> bool {
    exprs
        .into_iter()
        .any(|expr| mentions_known_tensor(expr, known))
}

fn mentions_known_tensor(expr: &Expr, known: &HashSet<String>) -> bool {
    let any = |exprs: Vec<&Expr>| any_mentions(exprs, known);
    match strip_expr_wrappers(expr) {
        Expr::Path(_) => is_known_tensor(expr, known),
        Expr::Binary(bin) => any(vec![&*bin.left, &*bin.right]),
        Expr::MethodCall(call) => {
            mentions_known_tensor(&call.receiver, known) || any_mentions(&call.args, known)
        }
        Expr::Call(call) => {
            mentions_known_tensor(&call.func, known) || any_mentions(&call.args, known)
        }
        Expr::Reference(reference) => mentions_known_tensor(&reference.expr, known),
        Expr::Unary(unary) => mentions_known_tensor(&unary.expr, known),
        Expr::Cast(cast) => mentions_known_tensor(&cast.expr, known),
        Expr::Field(field) => mentions_known_tensor(&field.base, known),
        Expr::Index(index) => any(vec![&*index.expr, &*index.index]),
        Expr::Array(array) => any_mentions(&array.elems, known),
        Expr::Tuple(tuple) => any_mentions(&tuple.elems, known),
        Expr::If(expr_if) => {
            mentions_known_tensor(&expr_if.cond, known)
                || expr_if.then_branch.stmts.iter().any(|stmt| match stmt {
                    Stmt::Expr(expr, _) => mentions_known_tensor(expr, known),
                    Stmt::Local(local) => local
                        .init
                        .as_ref()
                        .is_some_and(|init| mentions_known_tensor(&init.expr, known)),
                    Stmt::Item(_) | Stmt::Macro(_) => false,
                })
                || expr_if
                    .else_branch
                    .as_ref()
                    .is_some_and(|(_, expr)| mentions_known_tensor(expr, known))
        }
        _ => false,
    }
}

/// Returns the last op that a PTIR expression emits, if the expression reads a known PTIR tensor.
fn infer_view_kind(expr: &Expr, known: &HashSet<String>) -> Option<ViewKind> {
    match strip_expr_wrappers(expr) {
        Expr::Call(call) => {
            let Expr::Path(path) = &*call.func else {
                return None;
            };
            let name = path.path.segments.last()?.ident.to_string();
            match name.as_str() {
                "concat" | "try_concat" => Some(ViewKind::Concat),
                _ if call
                    .args
                    .iter()
                    .any(|arg| mentions_known_tensor(arg, known)) =>
                {
                    Some(ViewKind::Any)
                }
                _ => None,
            }
        }
        Expr::Binary(bin) => {
            if !mentions_known_tensor(&bin.left, known) && !mentions_known_tensor(&bin.right, known)
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
            if !mentions_known_tensor(&call.receiver, known) {
                return None;
            }
            match call.method.to_string().as_str() {
                "add" | "try_add" | "add_scalar" | "try_add_scalar" => Some(ViewKind::Add),
                "sub" | "try_sub" | "sub_scalar" | "try_sub_scalar" => Some(ViewKind::Sub),
                "mul" | "try_mul" | "mul_scalar" | "try_mul_scalar" => Some(ViewKind::Mul),
                "div" | "try_div" | "div_scalar" | "try_div_scalar" => Some(ViewKind::Div),
                "neg" | "try_neg" | "abs" | "try_abs" | "tanh" | "try_tanh" | "log" | "try_log"
                | "erf" | "try_erf" | "reciprocal" | "try_reciprocal" | "sqrt" | "try_sqrt"
                | "rsqrt" | "try_rsqrt" => Some(ViewKind::Unary),
                "cast" | "try_cast" => Some(ViewKind::Cast),
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
                "exp" | "try_exp" | "powf" | "try_powf" => Some(ViewKind::Exp),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Bind names: the `let` name, `<name>_<op>` when it shadows an earlier bind, and a numeric
/// suffix for other collisions.
#[derive(Default)]
struct BindNames {
    base_counts: HashMap<String, u32>,
    counts: HashMap<String, u32>,
}

impl BindNames {
    fn next(&mut self, base: &str, kind: ViewKind) -> String {
        let base_entry = self.base_counts.entry(base.to_string()).or_insert(0);
        let is_shadowed = *base_entry > 0;
        *base_entry += 1;
        let candidate = if is_shadowed {
            format!("{base}_{}", kind.bind_suffix())
        } else {
            base.to_string()
        };
        let entry = self.counts.entry(candidate.clone()).or_insert(0);
        let suffix = *entry;
        *entry += 1;
        if suffix == 0 {
            candidate
        } else {
            format!("{candidate}_{suffix}")
        }
    }
}

fn bind(expr: &mut Expr, name: &str) {
    let original = expr.clone();
    let name = syn::LitStr::new(name, Span::call_site());
    *expr = parse_quote! { (#original).ptir_bind(#name) };
}

/// Binds the PTIR values that a capture body names. `operands` are the imported tensors.
pub(crate) fn bind_capture_body(body: &mut Expr, operands: &[Ident]) -> Vec<BindInfo> {
    let mut known: HashSet<String> = operands.iter().map(ToString::to_string).collect();
    let mut names = BindNames::default();
    let mut binds = Vec::new();

    let tail = match body {
        Expr::Block(block) => {
            let stmts = &mut block.block.stmts;
            for stmt in stmts.iter_mut() {
                let Stmt::Local(Local {
                    pat,
                    init: Some(init),
                    ..
                }) = stmt
                else {
                    continue;
                };
                // `let (a, b) = f(known, ..)`: `a` and `b` are PTIR tensors, but their ops are
                // unknown.
                if let Pat::Tuple(tuple) = pat {
                    if mentions_known_tensor(&init.expr, &known) {
                        for elem in &tuple.elems {
                            if let Pat::Ident(PatIdent { ident, .. }) = elem {
                                known.insert(ident.to_string());
                            }
                        }
                    }
                    continue;
                }
                let Pat::Ident(PatIdent { ident, .. }) = pat else {
                    continue;
                };
                let Some(kind) = infer_view_kind(&init.expr, &known) else {
                    continue;
                };
                let name = names.next(&ident.to_string(), kind);
                bind(&mut init.expr, &name);
                known.insert(ident.to_string());
                binds.push(BindInfo { name, kind });
            }
            match stmts.last_mut() {
                Some(Stmt::Expr(tail, None)) => Some(tail),
                _ => None,
            }
        }
        other => Some(other),
    };
    if let Some(tail) = tail {
        if let Some(kind) = infer_view_kind(tail, &known) {
            let name = names.next("output", kind);
            bind(tail, &name);
            binds.push(BindInfo { name, kind });
        }
    }
    binds
}

/// Generates the pattern view of the functional `fn_ident` from the binds of its capture sites
/// (in site order). Also generates the statement that records the site templates while the
/// functional runs.
pub(crate) fn generate(fn_ident: &Ident, sites: &[Vec<BindInfo>]) -> Result<(Stmt, TokenStream)> {
    let span = fn_ident.span();
    let error = |message: String| Error::new(span, message);
    let fn_name = fn_ident.to_string();
    let pascal = to_pascal_case(&fn_name);
    let view_ident = format_ident!("{pascal}Pattern", span = span);
    let target = syn::LitStr::new(&format!("gpt_rs.{fn_name}"), span);
    let site_count = sites.len();

    // The view treats a bind whose op differs between sites as any op.
    let mut union: BTreeMap<String, ViewKind> = BTreeMap::new();
    let mut sites_with: HashMap<String, usize> = HashMap::new();
    for site in sites {
        let mut seen = HashSet::new();
        for bind in site {
            union
                .entry(bind.name.clone())
                .and_modify(|kind| {
                    if *kind != bind.kind {
                        *kind = ViewKind::Any;
                    }
                })
                .or_insert(bind.kind);
            if seen.insert(bind.name.as_str()) {
                *sites_with.entry(bind.name.clone()).or_insert(0) += 1;
            }
        }
    }
    let required: HashSet<String> = union
        .keys()
        .filter(|name| sites_with.get(*name) == Some(&site_count))
        .cloned()
        .collect();
    let (anchor_name, anchor_kind) = union
        .iter()
        .filter(|(name, _)| required.contains(*name))
        .min_by_key(|(name, kind)| (kind.anchor_score(), (*name).clone()))
        .map(|(name, kind)| (name.clone(), *kind))
        .ok_or_else(|| {
            error(format!(
                "the capture! bodies of `{fn_name}` share no bound PTIR op. A #[functional] \
                 needs one to anchor its pattern view"
            ))
        })?;

    let record_ident = format_ident!("__{}_pattern_record", fn_name, span = span);
    let site_captured_ident = format_ident!("__{}_pattern_site_captured", fn_name, span = span);
    let variant_ident = format_ident!("__{}PatternVariant", pascal, span = span);
    let site_locks = (0..site_count)
        .map(|index| format_ident!("__{}_PATTERN_SITE_{}", fn_name.to_uppercase(), index))
        .collect::<Vec<_>>();
    let site_indices = (0..site_count as u32).collect::<Vec<_>>();

    let field = |name: &str| format_ident!("{name}");
    let bind_fields = union.keys().map(|name| field(name)).collect::<Vec<_>>();
    let bind_lits = union
        .keys()
        .map(|name| syn::LitStr::new(name, span))
        .collect::<Vec<_>>();

    let variant_fields = union.keys().map(|name| {
        let ident = field(name);
        if required.contains(name) {
            quote!(#ident: ::gpt_rs::backend::pattern::TemplateNodeId)
        } else {
            quote!(#ident: ::core::option::Option<::gpt_rs::backend::pattern::TemplateNodeId>)
        }
    });
    let node_lookups = union.keys().zip(&bind_lits).map(|(name, lit)| {
        let ident = field(name);
        let lookup = quote! {
            binds
                .iter()
                .rev()
                .find(|bind| bind.name == #lit)
                .and_then(|bind| value_to_node.get(&bind.value).copied())
        };
        if required.contains(name) {
            quote!(let ::core::option::Option::Some(#ident) = #lookup else { return; };)
        } else {
            quote!(let #ident = #lookup;)
        }
    });
    let view_fields = union.iter().map(|(name, kind)| {
        let ident = field(name);
        let view = kind.view_type_tokens();
        if required.contains(name) {
            quote!(pub #ident: #view)
        } else {
            quote!(pub #ident: ::core::option::Option<#view>)
        }
    });
    let view_extracts = union
        .iter()
        .map(|(name, kind)| {
            let ident = field(name);
            let view = kind.view_type_tokens();
            let extract = quote! {
                <#view as ::gpt_rs::backend::pattern::OperationView>::extract(
                    matched.inst(node)?,
                    rewriter,
                )
            };
            if required.contains(name) {
                quote!(let #ident = { let node = variant.#ident; #extract? };)
            } else {
                quote! {
                    let #ident = match variant.#ident {
                        ::core::option::Option::Some(node) => ::core::option::Option::Some(#extract?),
                        ::core::option::Option::None => ::core::option::Option::None,
                    };
                }
            }
        })
        .collect::<Vec<_>>();
    let site_extracts = site_locks.iter().map(|lock| {
        quote! {
            if let ::core::option::Option::Some(variant) = #lock.get() {
                if let ::core::option::Option::Some(matched) =
                    variant.template.match_from_anchor(root, rewriter)
                {
                    #(#view_extracts)*
                    return ::core::option::Option::Some(Self {
                        match_: matched,
                        #(#bind_fields),*
                    });
                }
            }
        }
    });
    let pattern_fields = union.iter().map(|(name, kind)| {
        let view = kind.view_type_name();
        let optional = !required.contains(name);
        quote! {
            ::gpt_rs::backend::pattern::PatternField { name: #name, view: #view, optional: #optional }
        }
    });
    let anchor_lit = syn::LitStr::new(&anchor_name, span);
    let matcher = anchor_kind.matcher_tokens();
    let def_ident = format_ident!("__{}_PATTERN_DEF", fn_name.to_uppercase(), span = span);
    let view_doc = format!(
        "Pattern view of the PTIR that [`{fn_name}`] captures. It has one field per value that \
         its `capture!` bodies bind. A field is optional when only some capture sites bind it."
    );

    let guard: Stmt = parse_quote! {
        let _pattern = ::gpt_rs::backend::pattern::PatternCaptureGuard::push(
            #record_ident,
            #site_captured_ident,
        );
    };

    let items = quote! {
        #[derive(Clone)]
        struct #variant_ident {
            template: ::gpt_rs::backend::pattern::PatternTemplate,
            #(#variant_fields),*
        }

        #(
            static #site_locks: ::std::sync::OnceLock<#variant_ident> = ::std::sync::OnceLock::new();
        )*

        fn #site_captured_ident(site: u32) -> bool {
            match site {
                #(#site_indices => #site_locks.get().is_some(),)*
                _ => true,
            }
        }

        fn #record_ident(
            site: u32,
            nodes: &[::gpt_rs::backend::pattern::CapturedNode],
            binds: &[::gpt_rs::backend::pattern::BindRecord],
            outputs: &[::gpt_rs::backend::spec::ValueId],
        ) {
            if #site_captured_ident(site) {
                return;
            }
            let ::core::option::Option::Some(anchor) =
                binds.iter().rev().find(|bind| bind.name == #anchor_lit).map(|bind| bind.value)
            else {
                return;
            };
            let ::core::option::Option::Some(::gpt_rs::backend::pattern::BuiltTemplate {
                template,
                value_to_node,
            }) = ::gpt_rs::backend::pattern::build_template(nodes, outputs, anchor)
            else {
                return;
            };
            #(#node_lookups)*
            let variant = #variant_ident { template, #(#bind_fields),* };
            match site {
                #(#site_indices => { let _ = #site_locks.set(variant); })*
                _ => {}
            }
        }

        #[doc = #view_doc]
        #[derive(Clone)]
        pub struct #view_ident {
            match_: ::gpt_rs::backend::pattern::TemplateMatch,
            #(#view_fields),*
        }

        impl #view_ident {
            pub const TARGET: &'static str = #target;

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

            pub fn input(&self, index: u32) -> ::core::option::Option<::gpt_rs::backend::spec::ValueId> {
                self.match_.input(index)
            }

            pub fn input_count(&self) -> usize {
                self.match_.input_count()
            }
        }

        impl ::gpt_rs::backend::pattern::OperationView for #view_ident {
            const MATCHER: ::gpt_rs::backend::pattern::OperationMatcher = #matcher;

            fn extract(
                root: ::gpt_rs::backend::index::InstId,
                rewriter: &::gpt_rs::backend::rewriter::ProgramRewriter,
            ) -> ::core::option::Option<Self> {
                #(#site_extracts)*
                ::core::option::Option::None
            }
        }

        #[::gpt_rs::linkme::distributed_slice(::gpt_rs::backend::pattern::PATTERN_DEFS)]
        static #def_ident: ::gpt_rs::backend::pattern::PatternDef =
            ::gpt_rs::backend::pattern::PatternDef {
                target: #view_ident::TARGET,
                module_path: module_path!(),
                view_name: stringify!(#view_ident),
                fields: &[#(#pattern_fields),*],
            };
    };
    Ok((guard, items))
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
