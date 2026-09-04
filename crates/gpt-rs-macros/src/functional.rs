//! Expansion of `#[functional]` (documented on [`crate::functional`]).

use proc_macro2::TokenStream;
use quote::quote;
use syn::{
    parse_quote, spanned::Spanned, visit_mut::VisitMut, Error, FnArg, ItemFn, Macro, Result,
    ReturnType, Stmt,
};

use crate::capture::CaptureInput;
use crate::nn_module::{add_backend_param, TensorSugar};
use crate::pattern::{self, BindInfo};

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
            "a #[functional] must return a `Result` of its outputs",
        ));
    }
    add_backend_param(&mut function.sig.generics, "#[functional]")?;

    let mut sites = CaptureSites::default();
    sites.visit_block_mut(&mut function.block);
    if let Some(error) = sites.error {
        return Err(error);
    }
    if sites.binds.is_empty() {
        return Err(Error::new(
            function.sig.span(),
            "a #[functional] must capture its PTIR with `capture!`",
        ));
    }
    TensorSugar.visit_item_fn_mut(&mut function);

    let fn_ident = function.sig.ident.clone();
    let (pattern_guard, pattern_items) = pattern::generate(&fn_ident, &sites.binds)?;
    let name = fn_ident.to_string();
    let prelude: [Stmt; 2] = [
        parse_quote! {
            #[allow(dead_code)]
            const FUNCTIONAL: &str = #name;
        },
        pattern_guard,
    ];
    function.block.stmts.splice(0..0, prelude);

    Ok(quote! {
        #function
        #pattern_items
    })
}

/// Marks the `capture!` invocations of a functional body as its capture sites, in source order.
#[derive(Default)]
struct CaptureSites {
    binds: Vec<Vec<BindInfo>>,
    error: Option<Error>,
}

impl CaptureSites {
    /// Marks the invocation as the next capture site and binds the PTIR values its body names.
    fn rewrite(&mut self, mac: &mut Macro) {
        let mut capture: CaptureInput = match mac.parse_body() {
            Ok(capture) => capture,
            Err(error) => {
                self.error.get_or_insert(error);
                return;
            }
        };
        if capture.site.is_some() {
            self.error.get_or_insert(Error::new(
                mac.path.span(),
                "do not write `@site`: #[functional] sets it",
            ));
            return;
        }
        capture.site = Some(self.binds.len() as u32);
        self.binds.push(pattern::bind_capture_body(
            &mut capture.body,
            &capture.operands,
        ));
        mac.tokens = capture.to_tokens();
    }
}

fn is_capture(mac: &Macro) -> bool {
    mac.path
        .segments
        .last()
        .is_some_and(|segment| segment.ident == "capture")
}

impl VisitMut for CaptureSites {
    fn visit_macro_mut(&mut self, mac: &mut Macro) {
        if is_capture(mac) {
            self.rewrite(mac);
        }
    }
}
