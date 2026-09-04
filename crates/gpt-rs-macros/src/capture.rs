//! Expansion of `capture!` (documented on [`crate::capture`]).

use proc_macro2::{Span, TokenStream};
use quote::{format_ident, quote};
use syn::{
    parse::{Parse, ParseStream},
    spanned::Spanned,
    Error, Expr, ExprClosure, Ident, Pat, Result, ReturnType, Token,
};

/// A parsed `capture!([session,] |a, b, ...| body)`. Inside a `#[functional]`, the attribute
/// puts `@site <index>;` in front of the invocation. The index is the position of this capture
/// among the captures of the function.
pub(crate) struct CaptureInput {
    /// Index of the capture site in its `#[functional]`.
    pub site: Option<u32>,
    /// Name of the PTIR session in the body, if the body uses the session.
    pub session: Option<Ident>,
    /// The device tensors imported into the session. In the body, each name refers to the PTIR
    /// tensor.
    pub operands: Vec<Ident>,
    pub body: Expr,
}

impl Parse for CaptureInput {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let site = if input.peek(Token![@]) {
            input.parse::<Token![@]>()?;
            let keyword: Ident = input.parse()?;
            if keyword != "site" {
                return Err(Error::new(keyword.span(), "expected `@site <index>;`"));
            }
            let index: syn::LitInt = input.parse()?;
            input.parse::<Token![;]>()?;
            Some(index.base10_parse()?)
        } else {
            None
        };
        let session = if input.peek(Ident) && input.peek2(Token![,]) {
            let session: Ident = input.parse()?;
            input.parse::<Token![,]>()?;
            Some(session)
        } else {
            None
        };
        let closure: ExprClosure = input.parse()?;
        input.parse::<Option<Token![,]>>()?;
        if !input.is_empty() {
            return Err(input.error("expected `capture!([session,] |tensor, ...| body)`"));
        }
        if closure.movability.is_some()
            || closure.asyncness.is_some()
            || closure.capture.is_some()
            || closure.constness.is_some()
            || closure.lifetimes.is_some()
        {
            return Err(Error::new(
                closure.span(),
                "capture! takes a plain closure `|tensor, ...| body`",
            ));
        }
        if !matches!(closure.output, ReturnType::Default) {
            return Err(Error::new(
                closure.output.span(),
                "a capture! closure takes no return type: the body returns PTIR tensors",
            ));
        }
        let operands = closure
            .inputs
            .iter()
            .map(|input| match input {
                Pat::Ident(pat)
                    if pat.by_ref.is_none() && pat.mutability.is_none() && pat.subpat.is_none() =>
                {
                    Ok(pat.ident.clone())
                }
                other => Err(Error::new(
                    other.span(),
                    "capture! parameters must be plain names of device tensors in scope",
                )),
            })
            .collect::<Result<Vec<_>>>()?;
        if operands.is_empty() {
            return Err(Error::new(
                closure.span(),
                "capture! needs at least one device tensor",
            ));
        }
        Ok(CaptureInput {
            site,
            session,
            operands,
            body: *closure.body,
        })
    }
}

impl CaptureInput {
    /// The invocation tokens, with the site prefix when set.
    pub(crate) fn to_tokens(&self) -> TokenStream {
        let site = self.site.map(|site| quote!(@site #site;));
        let session = self.session.as_ref().map(|session| quote!(#session,));
        let operands = &self.operands;
        let body = &self.body;
        quote!(#site #session |#(#operands),*| #body)
    }
}

/// Expands a capture to a `Result` of the lazy device tensors that the body returns. A capture
/// site of a `#[functional]` also records its nodes for the pattern view.
pub(crate) fn expand(capture: &CaptureInput) -> TokenStream {
    let hidden = |name: &str| Ident::new(name, Span::mixed_site());
    let operand_refs = hidden("operands");
    let graph = hidden("graph");
    let builder = hidden("builder");
    let result = hidden("captured");
    let call = hidden("pattern_call");
    let session = capture.session.clone().unwrap_or_else(|| hidden("session"));
    let support = quote!(::gpt_rs::ops::functional::capture);

    let count = capture.operands.len();
    let operands = &capture.operands;
    let indices = (0..count).collect::<Vec<_>>();
    let ids = indices
        .iter()
        .map(|index| format_ident!("id_{}", index, span = Span::mixed_site()))
        .collect::<Vec<_>>();
    let body = &capture.body;

    let (begin, finish) = match capture.site {
        Some(site) => (
            quote! {
                let _site = ::gpt_rs::backend::pattern::PatternCaptureSiteGuard::push(#site);
                let #call = ::gpt_rs::backend::pattern::CaptureCallGuard::begin();
            },
            quote! {
                if let ::core::result::Result::Ok(ids) = &#result {
                    #call.finish(#support::CapturedIds::values(ids).as_ref());
                }
            },
        ),
        None => (TokenStream::new(), TokenStream::new()),
    };

    quote! {{
        let #operand_refs: [&::gpt_rs::tensor::DeviceTensor<_>; #count] = [#(&#operands),*];
        let #graph = #support::arena(&#operand_refs);
        #begin
        let #result = #graph.capture(|#builder| {
            #(let #ids = #builder.import(#operand_refs[#indices])?;)*
            let #session = ::gpt_rs::ops::ptir::PtirSession::new(#builder);
            #(
                let #operands = #session.import_spec(
                    stringify!(#operands),
                    #ids,
                    #support::tensor_spec(#operand_refs[#indices]),
                );
            )*
            ::core::result::Result::Ok(#support::CaptureOutput::into_ids(#body))
        });
        #finish
        #result.and_then(|ids| #support::CapturedIds::into_device_tensors(ids, &#graph))
    }}
}
