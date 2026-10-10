//! Expansion of `#[functional]` (documented on [`crate::functional`]).

use proc_macro2::TokenStream;
use quote::quote;
use syn::spanned::Spanned;
use syn::visit_mut::VisitMut;
use syn::{parse_quote, Error, FnArg, ItemFn, Result, ReturnType, Stmt};

use crate::nn_module::{add_backend_param, TensorSugar};

pub(crate) fn expand(attr: TokenStream, item: TokenStream) -> Result<TokenStream> {
    if !attr.is_empty() {
        return Err(Error::new(attr.span(), "#[functional] takes no arguments"));
    }
    let mut function: ItemFn = syn::parse2(item)?;
    if let Some(FnArg::Receiver(receiver)) = function.sig.inputs.first() {
        return Err(Error::new(
            receiver.span(),
            "#[functional] applies only to free functions",
        ));
    }
    if matches!(function.sig.output, ReturnType::Default) {
        return Err(Error::new(
            function.sig.span(),
            "a #[functional] must return its outputs",
        ));
    }
    add_backend_param(&mut function.sig.generics, "#[functional]")?;

    TensorSugar.visit_item_fn_mut(&mut function);
    function.attrs.push(parse_quote!(#[track_caller]));

    let name = function.sig.ident.to_string();
    let prelude: [Stmt; 2] = [
        parse_quote! {
            #[allow(dead_code)]
            const FUNCTIONAL: &str = #name;
        },
        parse_quote! {
            let _scope = ::gpt_rs::profiling::functional_scope(FUNCTIONAL);
        },
    ];
    function.block.stmts.splice(0..0, prelude);
    Ok(quote!(#function))
}
