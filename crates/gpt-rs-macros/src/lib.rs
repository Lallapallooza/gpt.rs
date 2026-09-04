use proc_macro::TokenStream;

mod capture;
mod functional;
mod nn_module;
mod pattern;

/// Declares a neural-network module in the style of a PyTorch `nn.Module`.
///
/// The macro is re-exported as `gpt_rs::nn::module`. Apply it to the struct (or enum) and to the
/// impl that holds its `forward`:
///
/// ```ignore
/// #[nn::module]
/// pub struct FeedForward {
///     up_proj: Linear,
///     activation: Activation,
///     down_proj: Linear,
/// }
///
/// #[nn::module]
/// impl FeedForward {
///     fn forward(&self, x: &Tensor) -> Result<Tensor> {
///         let up = self.up_proj(x)?;
///         let act = self.activation(&up)?;
///         self.down_proj(&act)
///     }
/// }
/// ```
///
/// # Backend
///
/// The macro adds the backend type parameter `B: PortableBackend + 'static` to both items, before
/// any other type parameter (`FeedForward<B>`). Every field is backend-generic unless it has
/// `#[module(config)]`:
/// - Write a sublayer type without generic arguments. The macro adds `<B>`, also inside
///   `Option<_>` and `Vec<_>`, so `Linear` becomes `Linear<B>`.
/// - A `Tensor` field is a parameter of type `DeviceTensor<B>`.
///
/// `Tensor` also means `DeviceTensor<B>` in `config` fields and in the impl. Other backend-generic
/// types name `B` explicitly (`&DecodeKvCache<B>`). A layer without parameters uses `B` through a
/// `config` field. For example, `Activation` holds a `fn(&Tensor) -> Result<Tensor>`.
///
/// # Parameters
///
/// The type implements `Module<B>`, and its `NAME` is the type's name. The visitors walk the
/// fields in declaration order and visit each field under its field name. For names that do not
/// follow Rust's snake_case field naming, use `#[module(rename = "A_log")]`. The visitors handle
/// fields as follows:
/// - A `Tensor` is the parameter `name`. The parameters of a sublayer are under `name.`.
/// - The visitors skip an `Option<_>` that is `None`.
/// - Element `i` of a `Vec<_>` is under `name.i`.
/// - A `#[module(flatten)]` sublayer adds no `name.` segment.
/// - `#[module(config)]` fields (dimensions, epsilons, flags) are not visited.
///
/// An enum holds one sublayer per variant, `Variant(Layer)`. The visitors visit the active
/// variant, under its `#[module(rename = ...)]` if it has one. A `rename` takes any `&str`
/// expression, such as a string literal or a constant.
///
/// # Forward and sublayer calls
///
/// The impl's `fn forward(&self, a: A, ...) -> Result<T>` implements `Layer<B>` with
/// `Output = T` and `Args<'a> = A`. With several arguments, `Args` is the tuple `(A, ...)`.
/// Elided lifetimes become `'a`. The other items of the impl stay inherent.
///
/// Each plain or optional sublayer field gets a method with the same name. The method runs the
/// sublayer with `Layer::call`, which opens the sublayer's profiler layer scope `NAME` around its
/// `forward`. Examples are `self.up_proj(x)`, and `self.self_attn((x, cache, positions))` for
/// several arguments. The method of an `Option<_>` field returns `Result<Option<T>>`. The result
/// is `None` when the sublayer is absent. `Vec<_>` elements and enum variants run with
/// `layer.call(args)`.
///
/// A sublayer field cannot be named `forward`, `call`, `visit_params`, or `visit_params_mut`. A
/// sublayer field with the same name as another method of the module is a duplicate definition.
#[proc_macro_attribute]
pub fn module(attr: TokenStream, item: TokenStream) -> TokenStream {
    match nn_module::expand(attr.into(), item.into()) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

/// Declares a portable functional. Its body validates the inputs, then captures PTIR.
///
/// The macro is re-exported as `gpt_rs::functional`:
///
/// ```ignore
/// #[functional]
/// pub fn gelu(x: &Tensor) -> Result<Tensor> {
///     ensure_rank_at_least!(x, 1);
///     capture!(|x| {
///         let half = 0.5f32 * x;
///         let erf = ptir::erf(x / ptir::sqrt(2.0f32));
///         half * (1.0f32 + erf)
///     })
/// }
/// ```
///
/// # Backend
///
/// The macro adds the backend type parameter `B: PortableBackend + 'static` to the function.
/// `Tensor` means `DeviceTensor<B>` in its signature and body. The function takes no backend
/// argument because its tensors carry the backend. Other backend-generic types name `B`
/// explicitly (`Result<LayerNormResult<B>>`).
///
/// # Validation
///
/// The body sees the constant `FUNCTIONAL`, which holds the function's name. The validation macros
/// put it in front of their messages, for example `gelu: x must have rank >= 1, got []`.
///
/// # Pattern view
///
/// Each `capture!` in the body is a capture site. The attribute generates the pattern view
/// `<Name>Pattern` with the target `gpt_rs.<name>` for backend rewrites, such as
/// `OpRewritePattern<GeluPattern>`. The view records the PTIR of each site the first time the site
/// runs.
///
/// The view has one field for each PTIR value that a capture body binds by name. The op that
/// produces the value sets the field type. A capture body binds these values:
/// - Each top-level `let name = <op>` binds `name`. The field is `<name>_<op>` when it shadows an
///   earlier bind.
/// - A result that is an op rather than a named value binds `output`.
///
/// A field that only some sites bind is optional. A field whose op differs between sites is an
/// `AnyOpView`. The view matches from its anchor, which is the most specific op that every site
/// binds.
#[proc_macro_attribute]
pub fn functional(attr: TokenStream, item: TokenStream) -> TokenStream {
    match functional::expand(attr.into(), item.into()) {
        Ok(tokens) => tokens.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

/// Captures PTIR over device tensors into their lazy graph. Re-exported as `gpt_rs::capture`.
///
/// `capture!(|a, b| body)` imports the device tensors `a` and `b` from the scope into a PTIR
/// session. Each one can be a `&DeviceTensor<B>` or a `DeviceTensor<B>`. The body sees them as
/// PTIR tensors with the same names. The body evaluates to a PTIR tensor or to a tuple of up to
/// four. `capture!` returns a `Result` of the matching lazy `DeviceTensor<B>` or tuple. A `?` in
/// the body returns its error from the capture.
///
/// `capture!(session, |a, b| body)` also gives the PTIR session a name. Use the session for
/// constants (`session.scalar(..)`, `session.iota(..)`) and for `session.export(value)`.
/// `export` keeps a value as a program output even when nothing reads it, for example an updated
/// cache.
///
/// The capture records into the graph of the first operand that has one. Without such an
/// operand, it uses the current default arena. Without a default arena, it creates a new graph.
#[proc_macro]
pub fn capture(input: TokenStream) -> TokenStream {
    match syn::parse::<capture::CaptureInput>(input) {
        Ok(input) => capture::expand(&input).into(),
        Err(err) => err.to_compile_error().into(),
    }
}
